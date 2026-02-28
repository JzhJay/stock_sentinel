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
from scoring import score_stock, calc_exit_points


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

    # ---------- 止盈止损 ----------
    print("\n" + "=" * 70)
    print("  🎯 止盈止损参考")
    print("=" * 70)
    print(f"\n{'排名':>4}  {'代码':<8} {'名称':<8} {'最新价':>7}"
          f"  {'止损':>7} {'止损%':>6}"
          f"  {'止盈①':>7} {'幅度%':>6}"
          f"  {'止盈②':>7} {'幅度%':>6}")
    print("-" * 90)

    for rank, (_, r) in enumerate(top.iterrows(), 1):
        sl = r.get("止损价")
        tp1 = r.get("止盈1")
        tp2 = r.get("止盈2")
        if sl is not None:
            print(f"{rank:>4}  {r['代码']:<8} {r['名称']:<8} {r['最新价']:>7.2f}"
                  f"  {sl:>7.2f} {r['止损%']:>+5.1f}%"
                  f"  {tp1:>7.2f} {r['止盈1%']:>+5.1f}%"
                  f"  {tp2:>7.2f} {r['止盈2%']:>+5.1f}%")

    print(f"\n  📌 止损：基于 ATR 波动率 + 近期支撑位 + 布林下轨（取最保守值）")
    print(f"  📌 止盈①：短线目标（ATR 1.5倍 / 布林中轨）")
    print(f"  📌 止盈②：波段目标（ATR 3倍 / 布林上轨 / 近期阻力位）")

    # ---------- 详细分析 ----------
    print("\n" + "=" * 70)
    print("  📋 详细分析")
    print("=" * 70)

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

        sl = r.get("止损价")
        if sl is not None:
            print(f"    💰 建议止损: {sl:.2f}({r['止损%']:+.1f}%)  "
                  f"止盈①: {r['止盈1']:.2f}({r['止盈1%']:+.1f}%)  "
                  f"止盈②: {r['止盈2']:.2f}({r['止盈2%']:+.1f}%)")


def _build_signal_text(row) -> str:
    """为单行数据生成信号摘要文本。"""
    tag_rules = [
        ("均线趋势", 8, "均线多头", 5, "均线偏多"),
        ("MACD", 8, "MACD金叉", 5, "MACD偏多"),
        ("成交量", 8, "量价配合", 5, "温和放量"),
        ("KDJ", 6, "KDJ金叉", None, None),
        ("动量", 7, "涨势良好", None, None),
        ("筹码分布", 7, "上方筹码稀疏", 4, "筹码尚可"),
    ]
    tags = []
    for col, th_hi, lbl_hi, th_lo, lbl_lo in tag_rules:
        if col in row and row[col] >= th_hi:
            tags.append(lbl_hi)
        elif th_lo is not None and col in row and row[col] >= th_lo:
            tags.append(lbl_lo)
    return " | ".join(tags) if tags else ""


def save_csv(result_df):
    import os
    os.makedirs("output", exist_ok=True)
    fname = os.path.join(
        "output",
        f"stock_picks_{datetime.date.today().strftime('%Y%m%d')}.csv",
    )

    out = result_df.head(TOP_N).copy()
    out.insert(0, "排名", range(1, len(out) + 1))

    out["止损%"] = out["止损%"].apply(lambda x: f"{x:+.1f}%")
    out["止盈1%"] = out["止盈1%"].apply(lambda x: f"{x:+.1f}%")
    out["止盈2%"] = out["止盈2%"].apply(lambda x: f"{x:+.1f}%")

    out["信号摘要"] = out.apply(_build_signal_text, axis=1)

    dim_cols = ["均线趋势", "MACD", "成交量", "筹码分布", "动量",
                "布林带", "RSI", "KDJ", "换手率"]

    col_order = [
        "排名", "代码", "名称", "总分", "最新价", "涨跌幅%",
        "止损价", "止损%", "止盈1", "止盈1%", "止盈2", "止盈2%",
        "信号摘要",
    ] + dim_cols

    out = out[[c for c in col_order if c in out.columns]]

    out.to_csv(fname, index=False, encoding="utf-8-sig")
    print(f"\n💾 结果已保存到: {fname}")


def print_disclaimer():
    print("\n" + "=" * 70)
    print("  ⚠️  免责声明")
    print("=" * 70)
    print("  本工具仅供学习研究使用，不构成任何投资建议。")
    print("  股市有风险，投资需谨慎。任何投资决策请自行判断。")
    print("  历史数据不代表未来表现，技术指标仅供参考。")
    print("  止盈止损仅为参考区间，实际操作请结合盘面综合判断。")
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
                    entry = {
                        "代码": code,
                        "名称": name,
                        "最新价": round(close.iloc[-1], 2),
                        "涨跌幅%": round(
                            (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2] * 100, 2
                        ),
                        **score,
                    }
                    exits = calc_exit_points(hist_df)
                    if exits:
                        entry.update(exits)
                    results.append(entry)
            except Exception as e:
                failed += 1
                print(f"   ⚠️ 评分异常 {code} {name}: {e}")
        else:
            failed += 1
            print(f"   ⚠️ K线不足 {code} {name} (跳过)")

        if completed % 20 == 0 or completed == total:
            pct = completed / total * 100
            print(f"   进度: {completed}/{total} ({pct:.0f}%)  "
                  f"有效: {len(results)}  失败: {failed}")

    return results


def main():
    print_header()

    all_stocks = get_all_stocks()
    if all_stocks is None or all_stocks.empty:
        print("❌ 无法获取行情数据，请检查网络连接")
        sys.exit(1)

    filtered = filter_stocks(all_stocks)
    if filtered.empty:
        print("❌ 过滤后没有符合条件的股票")
        sys.exit(1)

    candidates = pre_screen(filtered)

    results = fetch_and_score(candidates)
    if not results:
        print("\n❌ 没有找到符合条件的股票")
        sys.exit(1)

    result_df = pd.DataFrame(results).sort_values(
        ["总分", "代码"], ascending=[False, True],
    )

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
