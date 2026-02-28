# -*- coding: utf-8 -*-
"""
股票多维度评分模块

每个维度原始分 0~10，通过 SCORE_WEIGHTS 加权后总分满分 100。
新增维度只需在 config.SCORE_WEIGHTS 中添加 key 并调整权重。
"""

import numpy as np
import pandas as pd

from config import SCORE_WEIGHTS
from indicators import (
    calc_ma, calc_macd, calc_rsi, calc_kdj, calc_bollinger,
    calc_chip_distribution, calc_atr,
)


def score_stock(df: pd.DataFrame) -> dict | None:
    """对单只股票进行多维度评分，返回各项原始分与加权总分。"""
    if df is None or len(df) < 60:
        return None

    close = df["收盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    volume = df["成交量"].astype(float)
    turnover = df["换手率"].astype(float) if "换手率" in df.columns else None

    latest = len(close) - 1
    raw: dict[str, float] = {}

    # ---- 1. 均线趋势 (原始分 0~10) ----
    ma5 = calc_ma(close, 5)
    ma10 = calc_ma(close, 10)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)

    s = 0.0
    if close.iloc[latest] > ma5.iloc[latest]:
        s += 1
    if close.iloc[latest] > ma10.iloc[latest]:
        s += 1
    if close.iloc[latest] > ma20.iloc[latest]:
        s += 1.5
    if not np.isnan(ma60.iloc[latest]) and close.iloc[latest] > ma60.iloc[latest]:
        s += 1.5
    if ma5.iloc[latest] > ma5.iloc[latest - 3]:
        s += 1
    if ma10.iloc[latest] > ma10.iloc[latest - 3]:
        s += 1
    if ma5.iloc[latest] > ma10.iloc[latest] > ma20.iloc[latest]:
        s += 3

    raw["均线趋势"] = max(min(s, 10), 0)

    # ---- 2. MACD (原始分 0~10) ----
    dif, dea, hist = calc_macd(close)
    s = 0.0
    if hist.iloc[latest] > 0:
        s += 3
    if hist.iloc[latest] > 0 and hist.iloc[latest - 1] <= 0:
        s += 3
    if hist.iloc[latest] > hist.iloc[latest - 1]:
        s += 2
    if dif.iloc[latest] > dea.iloc[latest]:
        s += 2

    raw["MACD"] = max(min(s, 10), 0)

    # ---- 3. 成交量 (原始分 0~10) ----
    vol_ma5 = calc_ma(volume, 5)
    vol_ma10 = calc_ma(volume, 10)
    s = 0.0
    if vol_ma5.iloc[latest] > vol_ma10.iloc[latest]:
        s += 3
    if volume.iloc[latest] > vol_ma5.iloc[latest]:
        s += 2
    if close.iloc[latest] > close.iloc[latest - 1] and volume.iloc[latest] > volume.iloc[latest - 1]:
        s += 3
    if not np.isnan(vol_ma5.iloc[latest]) and vol_ma5.iloc[latest] > 0:
        ratio = volume.iloc[latest] / vol_ma5.iloc[latest]
        if 1.0 < ratio < 3.0:
            s += 2
        elif ratio >= 3.0:
            s -= 2

    raw["成交量"] = max(min(s, 10), 0)

    # ---- 4. RSI (原始分 0~10) ----
    rsi = calc_rsi(close, 14)
    s = 0.0
    rsi_val = rsi.iloc[latest]
    if not np.isnan(rsi_val):
        if 40 <= rsi_val <= 60:
            s += 5
        elif 30 <= rsi_val < 40:
            s += 7
        elif 20 <= rsi_val < 30:
            s += 4
        elif rsi_val > 80:
            s -= 3
        if rsi.iloc[latest] > rsi.iloc[latest - 1] and rsi.iloc[latest - 1] < rsi.iloc[latest - 2]:
            s += 3

    raw["RSI"] = max(min(s, 10), 0)

    # ---- 5. KDJ (原始分 0~10) ----
    k, d, j = calc_kdj(high, low, close)
    s = 0.0
    if not np.isnan(k.iloc[latest]):
        if k.iloc[latest] > d.iloc[latest] and k.iloc[latest - 1] <= d.iloc[latest - 1]:
            s += 6
        elif k.iloc[latest] > d.iloc[latest]:
            s += 3
        if 20 <= j.iloc[latest] <= 80:
            s += 2
        if j.iloc[latest] > j.iloc[latest - 1] and j.iloc[latest - 1] < 30:
            s += 2

    raw["KDJ"] = max(min(s, 10), 0)

    # ---- 6. 布林带 (原始分 0~10) ----
    upper, mid, lower = calc_bollinger(close)
    s = 0.0
    if not np.isnan(lower.iloc[latest]):
        price = close.iloc[latest]
        boll_w = upper.iloc[latest] - lower.iloc[latest]
        if boll_w > 0:
            pos = (price - lower.iloc[latest]) / boll_w
            if 0.4 <= pos <= 0.7:
                s += 5
            elif 0.1 <= pos < 0.4:
                s += 4
            elif pos > 0.9:
                s -= 2
            if price > mid.iloc[latest] and close.iloc[latest - 1] <= mid.iloc[latest - 1]:
                s += 5

    raw["布林带"] = max(min(s, 10), 0)

    # ---- 7. 动量 (原始分 0~10) ----
    pct5 = (close.iloc[latest] - close.iloc[latest - 5]) / close.iloc[latest - 5] * 100
    pct10 = (close.iloc[latest] - close.iloc[latest - 10]) / close.iloc[latest - 10] * 100
    s = 0.0
    if 1 <= pct5 <= 8:
        s += 4
    elif 0 < pct5 < 1:
        s += 2
    elif pct5 > 8:
        s += 1
    if 2 <= pct10 <= 15:
        s += 3
    elif 0 < pct10 < 2:
        s += 1
    if close.iloc[latest] > df["开盘"].astype(float).iloc[latest]:
        s += 3

    raw["动量"] = max(min(s, 10), 0)

    # ---- 8. 换手率 (原始分 0~10) ----
    s = 0.0
    if turnover is not None and not turnover.empty:
        tr = turnover.iloc[latest]
        tr_ma5 = calc_ma(turnover, 5).iloc[latest]
        if not np.isnan(tr):
            if 2 <= tr <= 8:
                s += 5
            elif 1 <= tr < 2:
                s += 3
            elif tr > 15:
                s -= 2
            if not np.isnan(tr_ma5) and tr_ma5 > 0:
                r = tr / tr_ma5
                if 1.0 < r < 2.5:
                    s += 5
                elif r >= 2.5:
                    s += 1

    raw["换手率"] = max(min(s, 10), 0)

    # ---- 9. 筹码分布 (原始分 0~10) ----
    chip = calc_chip_distribution(close, high, low, volume, period=60)
    s = 0.0
    if chip is not None:
        above = chip["above_ratio"]
        profit = chip["profit_ratio"]
        conc = chip["concentration_90"]

        if above < 0.15:
            s += 4
        elif above < 0.30:
            s += 3
        elif above < 0.50:
            s += 1
        elif above > 0.70:
            s -= 1

        if 0.60 <= profit <= 0.85:
            s += 3
        elif 0.50 <= profit < 0.60:
            s += 2
        elif profit > 0.90:
            s += 1
        elif profit < 0.30:
            s -= 1

        if conc < 0.15:
            s += 3
        elif conc < 0.25:
            s += 2
        elif conc < 0.35:
            s += 1

    raw["筹码分布"] = max(min(s, 10), 0)

    # ---- 加权总分 (满分 100) ----
    weighted_total = sum(
        raw[dim] / 10 * SCORE_WEIGHTS[dim]
        for dim in SCORE_WEIGHTS
        if dim in raw
    )

    result = {dim: raw[dim] for dim in SCORE_WEIGHTS if dim in raw}
    result["总分"] = round(weighted_total, 1)
    return result


def calc_exit_points(df: pd.DataFrame) -> dict | None:
    """
    综合 ATR、支撑阻力位、布林带 三种方法预测止盈止损点。

    止损逻辑：取 ATR 止损、近期支撑、布林下轨 中的最高值（最保守）
    止盈逻辑：分两档
      T1（短线）：ATR 1.5 倍目标 与 布林中轨 的较高者
      T2（波段）：ATR 3.0 倍目标 与 布林上轨 / 近期阻力 的较高者

    Returns dict: 止损价, 止盈1, 止盈2, 止损幅度%, 止盈1幅度%, 止盈2幅度%
    """
    if df is None or len(df) < 20:
        return None

    close = df["收盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    price = close.iloc[-1]

    if price <= 0:
        return None

    # --- ATR ---
    atr = calc_atr(high, low, close, 14)
    atr_val = atr.iloc[-1]
    if np.isnan(atr_val) or atr_val <= 0:
        atr_val = price * 0.03

    # --- 支撑与阻力 ---
    lookback = min(20, len(df))
    recent_low = low.iloc[-lookback:].min()
    recent_high = high.iloc[-lookback:].max()

    # --- 布林带 ---
    upper, mid, lower = calc_bollinger(close, 20, 2)
    boll_upper = upper.iloc[-1] if not np.isnan(upper.iloc[-1]) else price * 1.06
    boll_mid = mid.iloc[-1] if not np.isnan(mid.iloc[-1]) else price * 1.03
    boll_lower = lower.iloc[-1] if not np.isnan(lower.iloc[-1]) else price * 0.94

    # --- 止损：取三种方法中最保守的（最高值，离现价最近） ---
    sl_atr = price - 2.0 * atr_val
    sl_support = recent_low * 0.99
    sl_boll = boll_lower

    stop_loss = round(max(sl_atr, sl_support, sl_boll), 2)
    # 止损不能高于现价
    if stop_loss >= price:
        stop_loss = round(price * 0.95, 2)

    # --- 止盈 T1（短线目标）---
    tp1_atr = price + 1.5 * atr_val
    tp1 = round(max(tp1_atr, boll_mid), 2)
    if tp1 <= price:
        tp1 = round(price * 1.03, 2)

    # --- 止盈 T2（波段目标）---
    tp2_atr = price + 3.0 * atr_val
    tp2 = round(max(tp2_atr, boll_upper, recent_high), 2)
    if tp2 <= tp1:
        tp2 = round(tp1 * 1.03, 2)

    return {
        "止损价": stop_loss,
        "止盈1": tp1,
        "止盈2": tp2,
        "止损%": round((stop_loss / price - 1) * 100, 1),
        "止盈1%": round((tp1 / price - 1) * 100, 1),
        "止盈2%": round((tp2 / price - 1) * 100, 1),
    }
