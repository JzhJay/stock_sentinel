# -*- coding: utf-8 -*-
"""
股票多维度评分模块

第一轮：10 维度加权评分（无硬门槛淘汰，满分 100），选出 Top-10。
第二轮：5 维度精选评分（满分 50），从 Top-10 中选出 Top-3。
"""

import numpy as np
import pandas as pd

from config import (
    SCORE_WEIGHTS, ROUND2_WEIGHTS,
    MA_TREND_LOOKBACK, HARD_MAX_ABOVE_CHIP,
    HARD_MIN_REL_POS, HARD_MAX_REL_POS,
    HARD_REQUIRE_MA60_RISING, HARD_MIN_RISING_MA_COUNT,
    HARD_MIN_SPREAD_WIDEN_COUNT, HARD_MA_EPS,
    EXIT_ATR_MULT_STRONG, EXIT_ATR_MULT_BASE, EXIT_ATR_MULT_WEAK,
    EXIT_TREND_R2_STRONG, EXIT_TREND_R2_WEAK,
    EXIT_MIN_STOP_PCT, EXIT_MAX_STOP_PCT,
    EXIT_TP1_ATR_MULT, EXIT_TP2_ATR_MULT,
    EXIT_RR_TP1_WEIGHT, EXIT_RR_TP2_WEIGHT,
    EXIT_TRAIL_BUFFER_ATR, EXIT_TRAIL_ATR_MULT,
)
from indicators import (
    calc_ma, calc_macd, calc_rsi, calc_kdj, calc_bollinger,
    calc_chip_distribution, calc_atr,
    calc_trend_stability, calc_bias, calc_volume_trend, calc_relative_position,
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

    # ---- 偏好条件计算（用于加分，不做硬淘汰） ----
    lb = max(1, int(MA_TREND_LOOKBACK))
    if latest - lb < 0:
        return None

    ma_vals = [ma5.iloc[latest], ma10.iloc[latest], ma20.iloc[latest], ma60.iloc[latest]]
    if any(np.isnan(v) for v in ma_vals):
        return None

    price = close.iloc[latest]
    ma_bull = (price > ma5.iloc[latest] > ma10.iloc[latest] > ma20.iloc[latest] > ma60.iloc[latest])
    eps = float(HARD_MA_EPS)
    rising_5 = ma5.iloc[latest] > ma5.iloc[latest - lb] + eps
    rising_10 = ma10.iloc[latest] > ma10.iloc[latest - lb] + eps
    rising_20 = ma20.iloc[latest] > ma20.iloc[latest - lb] + eps
    rising_60 = ma60.iloc[latest] > ma60.iloc[latest - lb] + eps
    rising_count = int(rising_5) + int(rising_10) + int(rising_20) + int(rising_60)

    # 默认要求短中期三条均线上行；MA60 是否强制由配置控制
    if HARD_REQUIRE_MA60_RISING:
        ma_rising = (
            rising_5 and rising_10 and rising_20 and rising_60
            and rising_count >= int(HARD_MIN_RISING_MA_COUNT)
        )
    else:
        ma_rising = (
            rising_5 and rising_10 and rising_20
            and rising_count >= int(HARD_MIN_RISING_MA_COUNT)
        )

    spread_now = [
        ma5.iloc[latest] - ma10.iloc[latest],
        ma10.iloc[latest] - ma20.iloc[latest],
        ma20.iloc[latest] - ma60.iloc[latest],
    ]
    spread_old = [
        ma5.iloc[latest - lb] - ma10.iloc[latest - lb],
        ma10.iloc[latest - lb] - ma20.iloc[latest - lb],
        ma20.iloc[latest - lb] - ma60.iloc[latest - lb],
    ]
    spread_pos = all(s > 0 for s in spread_now)
    spread_widen_count = sum(1 for n, o in zip(spread_now, spread_old) if n > o + eps)
    spread_non_shrink_count = sum(1 for n, o in zip(spread_now, spread_old) if n >= o - eps)
    min_widen = max(0, int(HARD_MIN_SPREAD_WIDEN_COUNT))
    ma_diverging = spread_pos and (
        spread_widen_count >= min_widen
        or spread_non_shrink_count == 3
    )

    chip = calc_chip_distribution(close, high, low, volume, period=60)
    if chip is None:
        chip = {"above_ratio": 1.0, "profit_ratio": 0.0, "concentration_90": 1.0}
    above_ratio = chip["above_ratio"]

    rel_pos = calc_relative_position(close, period=120)
    if rel_pos is None:
        rel_pos = 0.5

    s = 0.0
    if price > ma5.iloc[latest]:
        s += 1
    if price > ma10.iloc[latest]:
        s += 1
    if price > ma20.iloc[latest]:
        s += 1.5
    if price > ma60.iloc[latest]:
        s += 1.5
    if ma5.iloc[latest] > ma5.iloc[latest - 3]:
        s += 1
    if ma10.iloc[latest] > ma10.iloc[latest - 3]:
        s += 1
    if ma20.iloc[latest] > ma20.iloc[latest - 3]:
        s += 0.5
    if ma60.iloc[latest] > ma60.iloc[latest - 3]:
        s += 0.5
    if ma5.iloc[latest] > ma10.iloc[latest] > ma20.iloc[latest]:
        s += 3
    if ma20.iloc[latest] > ma60.iloc[latest]:
        s += 0.5

    widen_bonus = min(spread_widen_count * 0.5, 1.5)
    s += widen_bonus
    if ma_bull:
        s += 0.5
    if ma_rising:
        s += 0.5
    if ma_diverging:
        s += 0.5

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
    s = 0.0
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
    if above <= HARD_MAX_ABOVE_CHIP:
        s += 1

    raw["筹码分布"] = max(min(s, 10), 0)

    # ---- 10. 相对低位 (原始分 0~10) ----
    s = 0.0
    if 0.18 <= rel_pos <= 0.40:
        s = 10
    elif 0.10 <= rel_pos < 0.18:
        s = 8
    elif 0.40 < rel_pos <= 0.50:
        s = 8
    elif rel_pos < 0.10:
        s = 4
    elif 0.50 < rel_pos <= 0.60:
        s = 5
    else:
        s = 2
    if HARD_MIN_REL_POS <= rel_pos <= HARD_MAX_REL_POS:
        s += 1
    raw["相对低位"] = max(min(s, 10), 0)

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

    止损逻辑：
      1) 按趋势强弱自适应 ATR 倍数（强趋势更宽，弱趋势更紧）
      2) 与近期支撑、布林下轨取最保守值
      3) 再套用最小/最大止损幅度保护带
    止盈逻辑：分两档
      T1（短线）：ATR 1.5 倍目标 与 布林中轨 的较高者
      T2（波段）：ATR 3.0 倍目标 与 布林上轨 / 近期阻力 的较高者

    并额外给出：触发止盈①后的移动止损建议（成本+缓冲 / MA10 / ATR 跟踪）

    Returns dict: 止损价, 止盈1, 止盈2, 止损幅度%, 止盈1幅度%, 止盈2幅度% 等
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

    # --- 趋势强弱（用于自适应 ATR 止损倍数） ---
    ts = calc_trend_stability(close, 20)
    sl_atr_mult = EXIT_ATR_MULT_BASE
    if ts is not None:
        r2 = ts["r_squared"]
        up = ts["slope"] > 0
        if up and r2 >= EXIT_TREND_R2_STRONG:
            sl_atr_mult = EXIT_ATR_MULT_STRONG
        elif up and r2 >= EXIT_TREND_R2_WEAK:
            sl_atr_mult = EXIT_ATR_MULT_BASE
        else:
            sl_atr_mult = EXIT_ATR_MULT_WEAK

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
    sl_atr = price - sl_atr_mult * atr_val
    sl_support = recent_low * 0.99
    sl_boll = boll_lower

    stop_loss_raw = max(sl_atr, sl_support, sl_boll)
    # 止损保护带：限制在 [最大回撤, 最小回撤] 区间内
    min_stop_price = price * (1 - EXIT_MAX_STOP_PCT / 100)   # 最低允许（最远）
    max_stop_price = price * (1 - EXIT_MIN_STOP_PCT / 100)   # 最高允许（最近）
    stop_loss = min(max(stop_loss_raw, min_stop_price), max_stop_price)
    # 兜底
    if stop_loss >= price:
        stop_loss = price * (1 - EXIT_MIN_STOP_PCT / 100)
    stop_loss = round(stop_loss, 2)

    # --- 止盈 T1（短线目标）---
    tp1_atr = price + EXIT_TP1_ATR_MULT * atr_val
    tp1 = round(max(tp1_atr, boll_mid), 2)
    if tp1 <= price:
        tp1 = round(price * 1.03, 2)

    # --- 止盈 T2（波段目标）---
    tp2_atr = price + EXIT_TP2_ATR_MULT * atr_val
    tp2 = round(max(tp2_atr, boll_upper, recent_high), 2)
    if tp2 <= tp1:
        tp2 = round(tp1 * 1.03, 2)

    # --- 移动止损建议（触发止盈①后执行） ---
    ma10 = calc_ma(close, 10).iloc[-1]
    trail_from_ma = ma10 if not np.isnan(ma10) else price
    trail_from_atr = price + EXIT_TRAIL_BUFFER_ATR * atr_val
    trail_from_follow = price - EXIT_TRAIL_ATR_MULT * atr_val
    trail_stop = round(max(trail_from_ma, trail_from_atr, trail_from_follow), 2)
    if trail_stop > tp1:
        trail_stop = round(tp1, 2)

    return {
        "止损价": stop_loss,
        "止盈1": tp1,
        "止盈2": tp2,
        "止损%": round((stop_loss / price - 1) * 100, 1),
        "止盈1%": round((tp1 / price - 1) * 100, 1),
        "止盈2%": round((tp2 / price - 1) * 100, 1),
        "移动止损触发价": tp1,
        "移动止损价": trail_stop,
        "移动止损%": round((trail_stop / price - 1) * 100, 1),
        "ATR止损系数": round(sl_atr_mult, 2),
    }


# ===================================================================
#  二次精选评分（从 Top-10 → Top-3）
# ===================================================================

def score_stock_round2(df: pd.DataFrame, exit_points: dict | None) -> dict | None:
    """
    五维度精选评分，每维 0~10，加权满分 50。
    需要传入已计算好的 exit_points（止盈止损字典）。
    """
    if df is None or len(df) < 60:
        return None

    close = df["收盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    volume = df["成交量"].astype(float)

    raw: dict[str, float] = {}

    # ---- 1. 风险收益比 (0~10) ----
    s = 5.0
    if exit_points:
        tp1_pct = abs(exit_points.get("止盈1%", 0))
        tp2_pct = abs(exit_points.get("止盈2%", tp1_pct))
        sl_pct = abs(exit_points.get("止损%", 0))
        if sl_pct > 0:
            tp_mix_pct = EXIT_RR_TP1_WEIGHT * tp1_pct + EXIT_RR_TP2_WEIGHT * tp2_pct
            ratio = tp_mix_pct / sl_pct
            if ratio >= 3.0:
                s = 10
            elif ratio >= 2.5:
                s = 8
            elif ratio >= 2.0:
                s = 6
            elif ratio >= 1.5:
                s = 4
            elif ratio >= 1.0:
                s = 2
            else:
                s = 0
    raw["风险收益比"] = s

    # ---- 2. 趋势稳定性 (0~10) ----
    s = 0.0
    ts = calc_trend_stability(close, 20)
    if ts is not None:
        r2 = ts["r_squared"]
        up = ts["slope"] > 0
        if up and r2 >= 0.80:
            s = 10
        elif up and r2 >= 0.60:
            s = 8
        elif up and r2 >= 0.40:
            s = 6
        elif up and r2 >= 0.20:
            s = 4
        elif up:
            s = 2
        elif r2 >= 0.60:
            s = 1

        higher_lows = sum(
            1 for i in range(-1, -min(10, len(low)), -1)
            if low.iloc[i] > low.iloc[i - 2]
        )
        s += min(higher_lows * 0.5, 2)

    raw["趋势稳定性"] = max(min(s, 10), 0)

    # ---- 3. 乖离率 BIAS (0~10) ----
    s = 0.0
    bias = calc_bias(close, 20)
    if bias is not None:
        if 0 <= bias <= 3:
            s = 10
        elif 3 < bias <= 5:
            s = 7
        elif 5 < bias <= 8:
            s = 4
        elif bias > 8:
            s = 1
        elif -2 <= bias < 0:
            s = 5
        elif -5 <= bias < -2:
            s = 3
        else:
            s = 1
    raw["乖离率"] = s

    # ---- 4. 量能持续性 (0~10) ----
    s = 0.0
    vt = calc_volume_trend(volume, 5)
    if vt is not None:
        above = vt["above_ma_days"]
        vol_up = vt["vol_slope"] > 0
        if above >= 4 and vol_up:
            s = 10
        elif above >= 3 and vol_up:
            s = 8
        elif above >= 3:
            s = 6
        elif above >= 2 and vol_up:
            s = 5
        elif above >= 2:
            s = 3
        else:
            s = 1
    raw["量能持续性"] = max(min(s, 10), 0)

    # ---- 5. 多周期共振 (0~10) ----
    s = 0.0
    ma5 = calc_ma(close, 5)
    ma10 = calc_ma(close, 10)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)

    price = close.iloc[-1]
    daily_up = price > ma5.iloc[-1] > ma10.iloc[-1]
    weekly_up = (
        not np.isnan(ma60.iloc[-1])
        and price > ma20.iloc[-1] > ma60.iloc[-1]
    )
    ma20_rising = ma20.iloc[-1] > ma20.iloc[-4] if len(ma20) > 4 else False
    ma60_rising = (
        not np.isnan(ma60.iloc[-4])
        and ma60.iloc[-1] > ma60.iloc[-4]
    ) if len(ma60) > 4 else False

    if daily_up and weekly_up:
        s = 7
    elif daily_up:
        s = 4
    elif weekly_up:
        s = 3
    if ma20_rising:
        s += 1.5
    if ma60_rising:
        s += 1.5

    raw["多周期共振"] = max(min(s, 10), 0)

    # ---- 加权总分 (满分 50) ----
    weighted = sum(
        raw[dim] / 10 * ROUND2_WEIGHTS[dim]
        for dim in ROUND2_WEIGHTS
        if dim in raw
    )

    result = {f"R2_{dim}": raw[dim] for dim in ROUND2_WEIGHTS if dim in raw}
    result["精选分"] = round(weighted, 1)
    return result
