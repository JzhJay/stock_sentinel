#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推荐后趋势预测模块

用途：
1) 读取 output/ 中近 N 日推荐结果（默认 stock_final_*.csv）
2) 对这些“买入点推荐股”做新的趋势延续预测
3) 输出 P_up_5 / P_dd_5 / P_up_10、TrendScore 与动作建议
4) 同时给出可回看样本的后5/10日表现统计
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import baostock as bs

from indicators import (
    calc_ma,
    calc_macd,
    calc_atr,
    calc_chip_distribution,
    calc_relative_position,
    calc_trend_stability,
    calc_volume_trend,
)


def _clip01(x: float, lo: float = 0.05, hi: float = 0.95) -> float:
    return float(min(max(x, lo), hi))


def _pure_to_bs(code6: str) -> str:
    return f"sh.{code6}" if code6.startswith("6") else f"sz.{code6}"


def _parse_file_date(path: str) -> dt.date | None:
    m = re.search(r"_(\d{8})\.csv$", os.path.basename(path))
    if not m:
        return None
    try:
        return dt.datetime.strptime(m.group(1), "%Y%m%d").date()
    except ValueError:
        return None


def _query_history_df(bs_code: str, start: dt.date, end: dt.date) -> pd.DataFrame | None:
    rs = bs.query_history_k_data_plus(
        bs_code,
        "date,open,high,low,close,volume,amount,turn",
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        frequency="d",
        adjustflag="2",
    )
    rows: list[list[str]] = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())

    if not rows:
        return None

    df = pd.DataFrame(
        rows,
        columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"],
    )
    for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["收盘"]).copy()
    if df.empty:
        return None
    df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
    df = df.dropna(subset=["日期"]).sort_values("日期").reset_index(drop=True)
    return df if not df.empty else None


def _load_recent_recs(source: str, days: int, top_per_day: int) -> pd.DataFrame:
    today = dt.date.today()
    pattern = "output/stock_final_*.csv" if source == "final" else "output/stock_picks_*.csv"
    paths = sorted(glob.glob(pattern))
    rows: list[pd.DataFrame] = []

    for p in paths:
        d = _parse_file_date(p)
        if d is None:
            continue
        delta = (today - d).days
        if delta < 0 or delta > days:
            continue

        df = pd.read_csv(p, dtype={"代码": str})
        if df.empty:
            continue

        if "排名" in df.columns and top_per_day > 0:
            df = df.sort_values("排名").head(top_per_day)
        elif top_per_day > 0:
            df = df.head(top_per_day)

        keep_cols = [c for c in ["代码", "名称", "最新价"] if c in df.columns]
        if not keep_cols:
            continue
        part = df[keep_cols].copy()
        part["代码"] = part["代码"].astype(str).str.zfill(6)
        part["推荐日期"] = pd.to_datetime(d)
        rows.append(part)

    if not rows:
        return pd.DataFrame(columns=["推荐日期", "代码", "名称", "最新价"])

    out = pd.concat(rows, ignore_index=True)
    for col in ["名称", "最新价"]:
        if col not in out.columns:
            out[col] = np.nan
    return out[["推荐日期", "代码", "名称", "最新价"]].sort_values(
        ["推荐日期", "代码"], ascending=[False, True],
    ).reset_index(drop=True)


def _calc_probs(hist: pd.DataFrame) -> dict[str, float]:
    close = hist["收盘"].astype(float)
    high = hist["最高"].astype(float)
    low = hist["最低"].astype(float)
    vol = hist["成交量"].astype(float)

    latest = len(close) - 1
    if latest < 60:
        return {"P_up_5": 0.5, "P_dd_5": 0.5, "P_up_10": 0.5, "TrendScore": 0.2}

    price = float(close.iloc[-1])
    ma5 = calc_ma(close, 5)
    ma10 = calc_ma(close, 10)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)

    ma_cond = [
        price > ma5.iloc[latest] > ma10.iloc[latest] > ma20.iloc[latest] > ma60.iloc[latest],
        ma5.iloc[latest] > ma5.iloc[max(0, latest - 3)],
        ma10.iloc[latest] > ma10.iloc[max(0, latest - 3)],
        ma20.iloc[latest] > ma20.iloc[max(0, latest - 3)],
        ma60.iloc[latest] > ma60.iloc[max(0, latest - 3)],
    ]
    ma_score = float(sum(bool(x) for x in ma_cond)) / len(ma_cond)

    ts = calc_trend_stability(close, 20)
    if ts is None:
        stab_score = 0.5
    else:
        r2 = float(ts["r_squared"])
        up = ts["slope"] > 0
        stab_score = _clip01((0.35 if up else 0.15) + 0.7 * r2, 0.0, 1.0)

    dif, dea, hist_macd = calc_macd(close)
    macd_score = 0.2
    if hist_macd.iloc[-1] > 0:
        macd_score += 0.4
    if dif.iloc[-1] > dea.iloc[-1]:
        macd_score += 0.3
    if hist_macd.iloc[-1] > hist_macd.iloc[-2]:
        macd_score += 0.1
    macd_score = _clip01(macd_score, 0.0, 1.0)

    chip = calc_chip_distribution(close, high, low, vol, period=60, num_bins=80)
    if chip is None:
        chip_score = 0.5
    else:
        above = float(chip["above_ratio"])
        if above < 0.15:
            chip_score = 1.0
        elif above < 0.30:
            chip_score = 0.82
        elif above < 0.50:
            chip_score = 0.58
        elif above < 0.70:
            chip_score = 0.35
        else:
            chip_score = 0.15

    rel = calc_relative_position(close, 120)
    rel = 0.5 if rel is None else float(rel)
    if 0.18 <= rel <= 0.45:
        lowpos_score = 1.0
    elif 0.10 <= rel < 0.18 or 0.45 < rel <= 0.60:
        lowpos_score = 0.75
    elif rel < 0.10:
        lowpos_score = 0.45
    else:
        lowpos_score = 0.25

    vt = calc_volume_trend(vol, 5)
    if vt is None:
        vol_score = 0.5
    else:
        days = int(vt["above_ma_days"])
        slope_up = vt["vol_slope"] > 0
        if days >= 4 and slope_up:
            vol_score = 1.0
        elif days >= 3 and slope_up:
            vol_score = 0.85
        elif days >= 3:
            vol_score = 0.70
        elif days >= 2:
            vol_score = 0.55
        else:
            vol_score = 0.35

    atr = calc_atr(high, low, close, 14).iloc[-1]
    atr_pct = float(atr / price * 100) if pd.notna(atr) and price > 0 else 3.0
    risk_vol = _clip01((atr_pct - 2.0) / 6.0, 0.0, 1.0)   # 越大风险越高
    overheat = _clip01((rel - 0.60) / 0.40, 0.0, 1.0)     # 高位追涨风险

    p_up_5 = _clip01(
        0.50
        + 0.18 * (ma_score - 0.5)
        + 0.16 * (stab_score - 0.5)
        + 0.14 * (macd_score - 0.5)
        + 0.12 * (chip_score - 0.5)
        + 0.10 * (vol_score - 0.5)
        + 0.08 * (lowpos_score - 0.5)
        - 0.10 * (risk_vol - 0.5),
    )

    p_up_10 = _clip01(
        0.50
        + 0.20 * (ma_score - 0.5)
        + 0.22 * (stab_score - 0.5)
        + 0.10 * (chip_score - 0.5)
        + 0.10 * (lowpos_score - 0.5)
        + 0.08 * (vol_score - 0.5)
        - 0.08 * (risk_vol - 0.5)
        - 0.08 * (overheat - 0.5),
    )

    p_dd_5 = _clip01(
        0.40
        + 0.24 * (risk_vol - 0.5)
        + 0.18 * (overheat - 0.5)
        - 0.15 * (chip_score - 0.5)
        - 0.12 * (ma_score - 0.5)
        - 0.08 * (stab_score - 0.5),
    )

    trend_score = 0.45 * p_up_5 + 0.25 * p_up_10 - 0.30 * p_dd_5
    return {
        "P_up_5": round(p_up_5, 3),
        "P_dd_5": round(p_dd_5, 3),
        "P_up_10": round(p_up_10, 3),
        "TrendScore": round(trend_score, 3),
    }


def _action(score: float) -> str:
    if score >= 0.60:
        return "持有/可小幅加仓"
    if score >= 0.45:
        return "继续观察（上移止损）"
    return "减仓/退出"


def _max_drawdown_pct(prices: pd.Series) -> float:
    if prices is None or prices.empty:
        return np.nan
    vals = prices.astype(float).values
    peak = np.maximum.accumulate(vals)
    dd = vals / peak - 1.0
    return float(np.min(dd) * 100)


def _build_row(rec_date: pd.Timestamp, code: str, name: str, rec_px: float, hist: pd.DataFrame) -> dict:
    dts = hist["日期"]
    idx_arr = np.where(dts.values >= np.datetime64(rec_date.date()))[0]
    if len(idx_arr) == 0:
        raise ValueError("推荐日后无交易数据")
    entry_idx = int(idx_arr[0])
    latest_idx = len(hist) - 1

    close = hist["收盘"].astype(float)
    entry_px = float(close.iloc[entry_idx])
    latest_px = float(close.iloc[latest_idx])
    hold_days = latest_idx - entry_idx

    period_ret = (latest_px / entry_px - 1) * 100
    period_dd = _max_drawdown_pct(close.iloc[entry_idx:latest_idx + 1])

    # 可回看指标（若未来数据不足则留空）
    f5 = np.nan
    f10 = np.nan
    dd5 = np.nan
    if entry_idx + 5 <= latest_idx:
        f5 = (float(close.iloc[entry_idx + 5]) / entry_px - 1) * 100
        dd5 = _max_drawdown_pct(close.iloc[entry_idx:entry_idx + 6])
    if entry_idx + 10 <= latest_idx:
        f10 = (float(close.iloc[entry_idx + 10]) / entry_px - 1) * 100

    probs = _calc_probs(hist)
    score = float(probs["TrendScore"])

    notes: list[str] = []
    if probs["P_up_5"] >= 0.60:
        notes.append("短线续涨概率偏高")
    if probs["P_dd_5"] >= 0.45:
        notes.append("回撤风险偏高")
    if probs["P_up_10"] >= 0.60:
        notes.append("中期趋势延续偏强")
    note = " | ".join(notes) if notes else "趋势中性"

    out = {
        "推荐日期": rec_date.strftime("%Y-%m-%d"),
        "代码": code,
        "名称": name,
        "推荐价": round(float(rec_px), 2) if pd.notna(rec_px) else round(entry_px, 2),
        "买入价(复权)": round(entry_px, 2),
        "最新价": round(latest_px, 2),
        "持有交易日": hold_days,
        "阶段涨跌%": round(period_ret, 2),
        "阶段最大回撤%": round(period_dd, 2),
        "后5日涨跌%": round(f5, 2) if pd.notna(f5) else np.nan,
        "后10日涨跌%": round(f10, 2) if pd.notna(f10) else np.nan,
        "后5日最大回撤%": round(dd5, 2) if pd.notna(dd5) else np.nan,
        "P_up_5": probs["P_up_5"],
        "P_dd_5": probs["P_dd_5"],
        "P_up_10": probs["P_up_10"],
        "TrendScore": probs["TrendScore"],
        "建议动作": _action(score),
        "说明": note,
    }
    return out


def _print_review_stats(df: pd.DataFrame) -> None:
    valid5 = df["后5日涨跌%"].notna()
    valid10 = df["后10日涨跌%"].notna()

    print("\n📊 回看统计（仅统计未来数据完整样本）")
    if valid5.any():
        n5 = int(valid5.sum())
        hit5 = (df.loc[valid5, "后5日涨跌%"] >= 2.0).mean()
        pred5 = (df.loc[valid5, "P_up_5"] >= 0.55)
        real5 = (df.loc[valid5, "后5日涨跌%"] >= 2.0)
        acc5 = (pred5 == real5).mean()
        print(f"   5日样本: {n5}  |  实际达标率(>=2%): {hit5:.1%}  |  阈值0.55判断准确率: {acc5:.1%}")
    else:
        print("   5日样本: 0")

    if valid10.any():
        n10 = int(valid10.sum())
        hit10 = (df.loc[valid10, "后10日涨跌%"] >= 4.0).mean()
        pred10 = (df.loc[valid10, "P_up_10"] >= 0.55)
        real10 = (df.loc[valid10, "后10日涨跌%"] >= 4.0)
        acc10 = (pred10 == real10).mean()
        print(f"   10日样本: {n10} |  实际达标率(>=4%): {hit10:.1%}  |  阈值0.55判断准确率: {acc10:.1%}")
    else:
        print("   10日样本: 0")


def main() -> int:
    parser = argparse.ArgumentParser(description="对近几日推荐股做趋势延续预测")
    parser.add_argument("--days", type=int, default=5, help="回看最近多少自然日的推荐文件")
    parser.add_argument("--source", choices=["final", "picks"], default="final", help="推荐来源文件类型")
    parser.add_argument("--top-per-day", type=int, default=3, help="每个推荐日最多纳入多少只股票")
    parser.add_argument("--output", type=str, default="", help="输出CSV路径（默认 output/trend_forecast_YYYYMMDD.csv）")
    args = parser.parse_args()

    recs = _load_recent_recs(source=args.source, days=args.days, top_per_day=args.top_per_day)
    if recs.empty:
        print("❌ 未找到符合条件的推荐文件或记录")
        return 1

    start_date = recs["推荐日期"].min().date() - dt.timedelta(days=220)
    end_date = dt.date.today()

    print(f"📥 读取推荐记录: {len(recs)} 条")
    print(f"   来源: output/stock_{args.source}_*.csv  最近 {args.days} 天")
    print(f"   历史区间: {start_date} ~ {end_date}")

    lg = bs.login()
    if lg.error_code != "0":
        print(f"❌ baostock 登录失败: {lg.error_msg}")
        return 1

    try:
        cache: dict[str, pd.DataFrame] = {}
        for code in sorted(recs["代码"].unique()):
            bs_code = _pure_to_bs(code)
            cache[code] = _query_history_df(bs_code, start_date, end_date)

        out_rows: list[dict] = []
        failed = 0
        for r in recs.itertuples(index=False):
            hist = cache.get(r.代码)
            if hist is None or len(hist) < 60:
                failed += 1
                continue
            try:
                out_rows.append(_build_row(r.推荐日期, r.代码, str(r.名称), r.最新价, hist))
            except Exception:
                failed += 1

        if not out_rows:
            print("❌ 未生成有效趋势预测结果（请检查样本或数据源）")
            return 1

        out = pd.DataFrame(out_rows).sort_values(
            ["TrendScore", "推荐日期", "代码"], ascending=[False, False, True],
        ).reset_index(drop=True)

        for c in ["P_up_5", "P_dd_5", "P_up_10", "TrendScore"]:
            out[c] = out[c].map(lambda x: f"{x:.3f}")

        os.makedirs("output", exist_ok=True)
        out_path = args.output.strip() if args.output.strip() else os.path.join(
            "output", f"trend_forecast_{dt.date.today().strftime('%Y%m%d')}.csv",
        )
        out.to_csv(out_path, index=False, encoding="utf-8-sig")

        print(f"\n✅ 趋势预测完成: 有效 {len(out)} 条, 失败 {failed} 条")
        print(f"💾 已保存: {out_path}")
        print("\nTop 10 结果预览：")
        print(out.head(10).to_string(index=False))
        _print_review_stats(pd.DataFrame(out_rows))
        return 0
    finally:
        try:
            bs.logout()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
