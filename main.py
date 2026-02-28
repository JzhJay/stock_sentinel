#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A 股智能选股系统 — 次日买入推荐

数据来源：新浪财经(实时行情) + baostock(历史K线)
使用方法：
    python main.py
"""

import warnings
warnings.filterwarnings("ignore")

import sys
import datetime

import pandas as pd
import baostock as bs

from config import TOP_N, MIN_PRICE, MAX_PRICE
from data import get_all_stocks, filter_stocks, pre_screen, get_stock_history
from scoring import score_stock


# ==================== 结果展示 ====================

def print_header():
    print("=" * 70)
    print("          🏦 A 股智能选股系统 — 次日买入推荐")
    print("=" * 70)
    print(f"  运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据来源: 新浪财经(实时行情) + baostock(历史K线)")
    print(f"  评分体系: 9 维度加权评分（满分 100）")
    print(f"  价格区间: {MIN_PRICE} ~ {MAX_PRICE} 元")
    print(f"  推荐数量: 前 {TOP_N} 只")
    print("=" * 70)
    print()


def print_results(result_df):
    top = result_df.head(TOP_N)

    # 各维度显示 0~10 原始分，总分显示加权后 /100
    print(f"\n{'排名':>4}  {'代码':<8} {'名称':<8} {'最新价':>7} {'涨幅%':>6} "
          f"{'总分':>6} {'均线':>4} {'MACD':>4} {'量':>4} {'RSI':>4} "
          f"{'KDJ':>4} {'布林':>4} {'动量':>4} {'换手':>4} {'筹码':>4}")
    print("-" * 100)

    for rank, (_, r) in enumerate(top.iterrows(), 1):
        print(f"{rank:>4}  {r['代码']:<8} {r['名称']:<8} "
              f"{r['最新价']:>7.2f} {r['涨跌幅%']:>+6.2f} "
              f"{r['总分']:>6.1f} {r['均线趋势']:>4.0f} {r['MACD']:>4.0f} "
              f"{r['成交量']:>4.0f} {r['RSI']:>4.0f} "
              f"{r['KDJ']:>4.0f} {r['布林带']:>4.0f} {r['动量']:>4.0f} {r['换手率']:>4.0f} "
              f"{r['筹码分布']:>4.0f}")

    print("\n" + "=" * 70)
    print("  📋 详细分析")
    print("=" * 70)

    # (维度名, 原始分高阈值, 高标签, 原始分中阈值, 中标签)
    tag_rules = [
        ("均线趋势", 8, "✅ 均线多头排列",    5, "🔶 均线趋势偏多"),
        ("MACD",     8, "✅ MACD金叉/柱翻红", 5, "🔶 MACD偏多"),
        ("成交量",   8, "✅ 量价配合良好",    5, "🔶 成交量温和放大"),
        ("KDJ",      6, "✅ KDJ金叉信号",    None, None),
        ("动量",     7, "✅ 近期涨势良好",    None, None),
        ("筹码分布", 7, "✅ 上方筹码稀疏",    4,   "🔶 筹码结构尚可"),
    ]

    for rank, (_, r) in enumerate(top.iterrows(), 1):
        print(f"\n  第{rank}名: {r['代码']} {r['名称']}  "
              f"最新价: {r['最新价']:.2f}  涨跌幅: {r['涨跌幅%']:+.2f}%  "
              f"总分: {r['总分']:.1f}/100")
        details = []
        for col, th_hi, label_hi, th_lo, label_lo in tag_rules:
            if r[col] >= th_hi:
                details.append(label_hi)
            elif th_lo is not None and r[col] >= th_lo:
                details.append(label_lo)
        if details:
            print(f"    {'  |  '.join(details)}")


def save_csv(result_df):
    import os
    os.makedirs("output", exist_ok=True)
    fname = os.path.join(
        "output",
        f"stock_picks_{datetime.date.today().strftime('%Y%m%d')}.csv",
    )
    result_df.head(TOP_N).to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"\n💾 结果已保存到: {fname}")


def print_disclaimer():
    print("\n" + "=" * 70)
    print("  ⚠️  免责声明")
    print("=" * 70)
    print("  本工具仅供学习研究使用，不构成任何投资建议。")
    print("  股市有风险，投资需谨慎。任何投资决策请自行判断。")
    print("  历史数据不代表未来表现，技术指标仅供参考。")
    print("=" * 70)


# ==================== 核心流程 ====================

def fetch_and_score(candidates):
    """逐只获取历史 K 线并评分，返回结果列表。"""
    results = []
    completed = 0
    failed = 0
    total = len(candidates)

    print(f"\n📈 正在获取候选股票历史 K 线并评分（共 {total} 只）...")
    print("   这可能需要几分钟，请耐心等待...\n")

    for _, row in candidates.iterrows():
        bs_code = str(row["bs_code"])
        name = str(row["名称"])

        code, name, hist_df = get_stock_history(bs_code, name)
        completed += 1

        if hist_df is not None and len(hist_df) >= 60:
            try:
                score = score_stock(hist_df)
                if score and score["总分"] > 0:
                    close = hist_df["收盘"].astype(float)
                    results.append({
                        "代码": code,
                        "名称": name,
                        "最新价": round(close.iloc[-1], 2),
                        "涨跌幅%": round(
                            (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2
                        ),
                        **score,
                    })
            except Exception:
                failed += 1
        else:
            failed += 1

        if completed % 20 == 0 or completed == total:
            pct = completed / total * 100
            print(f"   进度: {completed}/{total} ({pct:.0f}%)  "
                  f"有效: {len(results)}  失败: {failed}")

    return results


def main():
    print_header()

    # 1) 获取实时行情
    all_stocks = get_all_stocks()
    if all_stocks is None or all_stocks.empty:
        print("❌ 无法获取行情数据，请检查网络连接")
        sys.exit(1)

    # 2) 基础过滤
    filtered = filter_stocks(all_stocks)
    if filtered.empty:
        print("❌ 过滤后没有符合条件的股票")
        sys.exit(1)

    # 3) 预筛选
    candidates = pre_screen(filtered)

    # 4) 获取历史 K 线 & 评分
    results = fetch_and_score(candidates)
    if not results:
        print("\n❌ 没有找到符合条件的股票")
        sys.exit(1)

    # 5) 输出
    result_df = pd.DataFrame(results).sort_values("总分", ascending=False)

    print("\n" + "=" * 70)
    print(f"  🏆 推荐次日买入的前 {TOP_N} 只股票")
    print("=" * 70)

    print_results(result_df)
    save_csv(result_df)
    print_disclaimer()


if __name__ == "__main__":
    lg = bs.login()
    if lg.error_code != "0":
        print(f"❌ baostock 登录失败: {lg.error_msg}")
        sys.exit(1)
    try:
        main()
    finally:
        try:
            bs.logout()
        except Exception:
            pass
