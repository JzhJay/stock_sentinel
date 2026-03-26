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

from config import TOP_N, FINAL_TOP, MIN_PRICE, MAX_PRICE, ROUND2_WEIGHTS
from data import get_all_stocks, filter_stocks, pre_screen, get_stock_history
from scoring import score_stock, calc_exit_points, score_stock_round2


# ==================== 结果展示 ====================

def print_header():
    print("=" * 70)
    print("          🏦 A 股智能选股系统 — 次日买入推荐")
    print("=" * 70)
    print(f"  运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据来源: 新浪财经(实时行情) + baostock(历史K线)")
    print(f"  评分体系: 9 维度加权评分（满分 100）+ 5 维度精选（满分 50）")
    print(f"  价格区间: {MIN_PRICE} ~ {MAX_PRICE} 元")
    print(f"  推荐数量: 初选 {TOP_N} 只 → 精选 {FINAL_TOP} 只")
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


def print_round2(top_df):
    """打印二次精选评分对比表。"""
    r2_cols = ["R2_风险收益比", "R2_趋势稳定性", "R2_乖离率", "R2_量能持续性", "R2_多周期共振"]
    print(f"\n{'排名':>4}  {'代码':<8} {'名称':<8} {'初选分':>6} {'精选分':>6} {'综合分':>6}"
          f"  {'风险比':>5} {'稳定':>4} {'乖离':>4} {'量能':>4} {'共振':>4}")
    print("-" * 90)
    for rank, (_, r) in enumerate(top_df.iterrows(), 1):
        print(f"{rank:>4}  {r['代码']:<8} {r['名称']:<8} "
              f"{r['总分']:>6.1f} {r.get('精选分', 0):>6.1f} {r.get('综合分', 0):>6.1f}"
              f"  {r.get('R2_风险收益比', 0):>5.0f} {r.get('R2_趋势稳定性', 0):>4.0f}"
              f" {r.get('R2_乖离率', 0):>4.0f} {r.get('R2_量能持续性', 0):>4.0f}"
              f" {r.get('R2_多周期共振', 0):>4.0f}")


def print_final(final_df):
    """打印最终 Top-3 推荐。"""
    for rank, (_, r) in enumerate(final_df.iterrows(), 1):
        print(f"\n  {'🥇🥈🥉'[rank-1]} 第{rank}名: {r['代码']} {r['名称']}  "
              f"综合分: {r.get('综合分', 0):.1f}/150 "
              f"(初选 {r['总分']:.1f} + 精选 {r.get('精选分', 0):.1f})")
        print(f"    最新价: {r['最新价']:.2f}  涨跌幅: {r['涨跌幅%']:+.2f}%")

        sl = r.get("止损价")
        if sl is not None:
            print(f"    止损: {sl:.2f}({r['止损%']:+.1f}%)  "
                  f"止盈①: {r['止盈1']:.2f}({r['止盈1%']:+.1f}%)  "
                  f"止盈②: {r['止盈2']:.2f}({r['止盈2%']:+.1f}%)")

        tags = []
        tag_rules = [
            ("均线趋势", 8, "均线多头"), ("MACD", 8, "MACD金叉"),
            ("成交量", 8, "量价配合"), ("KDJ", 6, "KDJ金叉"),
            ("R2_趋势稳定性", 8, "趋势平滑"), ("R2_多周期共振", 7, "多周期共振"),
            ("R2_量能持续性", 8, "持续放量"), ("R2_风险收益比", 8, "高性价比"),
        ]
        for col, th, lbl in tag_rules:
            if col in r and r[col] >= th:
                tags.append(lbl)
        if tags:
            print(f"    核心信号: {' | '.join(tags)}")


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


def save_csv(result_df, final_df=None):
    import os
    os.makedirs("output", exist_ok=True)
    date_str = datetime.date.today().strftime("%Y%m%d")

    dim_cols = ["均线趋势", "MACD", "成交量", "筹码分布", "动量",
                "布林带", "RSI", "KDJ", "换手率"]
    r2_cols = ["R2_风险收益比", "R2_趋势稳定性", "R2_乖离率", "R2_量能持续性", "R2_多周期共振"]

    def _fmt_pct(col_name, df):
        if col_name in df.columns:
            df[col_name] = df[col_name].apply(lambda x: f"{x:+.1f}%" if pd.notna(x) else "")

    # --- 初选 Top-10 ---
    fname10 = os.path.join("output", f"stock_picks_{date_str}.csv")
    out10 = result_df.head(TOP_N).copy()
    out10.insert(0, "排名", range(1, len(out10) + 1))
    for c in ["止损%", "止盈1%", "止盈2%"]:
        _fmt_pct(c, out10)
    out10["信号摘要"] = out10.apply(_build_signal_text, axis=1)

    col_order_10 = [
        "排名", "代码", "名称", "总分", "最新价", "涨跌幅%",
        "止损价", "止损%", "止盈1", "止盈1%", "止盈2", "止盈2%",
        "信号摘要",
    ] + dim_cols
    out10 = out10[[c for c in col_order_10 if c in out10.columns]]
    out10.to_csv(fname10, index=False, encoding="utf-8-sig")
    print(f"\n💾 初选结果已保存到: {fname10}")

    # --- 精选 Top-3 ---
    if final_df is not None and not final_df.empty:
        fname3 = os.path.join("output", f"stock_final_{date_str}.csv")
        out3 = final_df.head(FINAL_TOP).copy()
        out3.insert(0, "排名", range(1, len(out3) + 1))
        for c in ["止损%", "止盈1%", "止盈2%"]:
            _fmt_pct(c, out3)
        out3["信号摘要"] = out3.apply(_build_signal_text, axis=1)

        col_order_3 = [
            "排名", "代码", "名称", "综合分", "总分", "精选分",
            "最新价", "涨跌幅%",
            "止损价", "止损%", "止盈1", "止盈1%", "止盈2", "止盈2%",
            "信号摘要",
        ] + r2_cols + dim_cols
        out3 = out3[[c for c in col_order_3 if c in out3.columns]]
        out3.to_csv(fname3, index=False, encoding="utf-8-sig")
        print(f"💾 精选结果已保存到: {fname3}")


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
    """逐只获取历史 K 线并评分，返回 (结果列表, {代码: hist_df} 缓存)。"""
    results = []
    hist_cache: dict[str, pd.DataFrame] = {}
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
                    hist_cache[code] = hist_df
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

    return results, hist_cache


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

    results, hist_cache = fetch_and_score(candidates)
    if not results:
        print("\n❌ 没有找到符合条件的股票")
        sys.exit(1)

    result_df = pd.DataFrame(results).sort_values(
        ["总分", "代码"], ascending=[False, True],
    )

    print("\n" + "=" * 70)
    print(f"  🏆 第一轮：推荐初选前 {TOP_N} 只股票")
    print("=" * 70)

    print_results(result_df)

    # ==================== 二次精选 ====================
    top_df = result_df.head(TOP_N).copy()
    print("\n" + "=" * 70)
    print(f"  🔬 第二轮：从 Top-{TOP_N} 中精选 Top-{FINAL_TOP}")
    print("=" * 70)

    r2_dims = list(ROUND2_WEIGHTS.keys())
    print(f"  精选维度: {' / '.join(r2_dims)}")
    print()

    r2_scores = []
    for _, row in top_df.iterrows():
        code = row["代码"]
        hist_df = hist_cache.get(code)
        exit_pts = {k: row[k] for k in ["止损价", "止盈1", "止盈2", "止损%", "止盈1%", "止盈2%"]
                    if k in row and pd.notna(row[k])}
        r2 = score_stock_round2(hist_df, exit_pts if exit_pts else None)
        if r2:
            r2["代码"] = code
            r2_scores.append(r2)

    if r2_scores:
        r2_df = pd.DataFrame(r2_scores)
        top_df = top_df.merge(r2_df, on="代码", how="left")
        top_df["精选分"] = top_df["精选分"].fillna(0)
        top_df["综合分"] = top_df["总分"] + top_df["精选分"]
        top_df = top_df.sort_values(
            ["综合分", "代码"], ascending=[False, True],
        )

        print_round2(top_df)

        final_df = top_df.head(FINAL_TOP)
        print("\n" + "=" * 70)
        print(f"  ⭐ 最终推荐：Top-{FINAL_TOP}")
        print("=" * 70)
        print_final(final_df)
        save_csv(result_df, final_df)
    else:
        save_csv(result_df, None)

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
