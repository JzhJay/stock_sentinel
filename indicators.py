# -*- coding: utf-8 -*-
"""技术指标计算模块"""

import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import polyfit


def calc_ma(series: pd.Series, period: int) -> pd.Series:
    """简单移动平均线"""
    return series.rolling(window=period, min_periods=period).mean()


def calc_ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均线"""
    return series.ewm(span=period, adjust=False).mean()


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """MACD 指标，返回 (DIF, DEA, MACD柱)"""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def calc_rsi(close: pd.Series, period=14) -> pd.Series:
    """RSI 指标"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series,
             n=9, m1=3, m2=3):
    """KDJ 指标，返回 (K, D, J)"""
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_bollinger(close: pd.Series, period=20, num_std=2):
    """布林带，返回 (上轨, 中轨, 下轨)"""
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series,
             period: int = 14) -> pd.Series:
    """
    Average True Range — 衡量价格波动幅度。
    用于计算基于波动率的止盈止损区间。
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def calc_chip_distribution(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    period: int = 60,
    num_bins: int = 100,
    decay: float = 0.97,
) -> dict | None:
    """
    估算筹码分布。

    用最近 *period* 个交易日的量价数据，将每日成交量按三角分布
    分配到 [最低价, 最高价] 的价格分箱中，同时用 *decay* 因子
    模拟旧筹码因换手而逐日衰减。

    Returns
    -------
    dict  (若数据不足返回 None)
        above_ratio      : 当前价上方筹码占比 (0~1)，越低上涨阻力越小
        profit_ratio     : 获利盘比例（当前价下方筹码占比）
        concentration_90 : 90% 筹码价格宽度 / 当前价，越小筹码越集中
    """
    n = len(close)
    if n < period:
        return None

    c = close.iloc[n - period:].values
    h = high.iloc[n - period:].values
    lo = low.iloc[n - period:].values
    v = volume.iloc[n - period:].values

    current_price = c[-1]
    if current_price <= 0:
        return None

    price_min = lo.min() * 0.99
    price_max = h.max() * 1.01
    bins = np.linspace(price_min, price_max, num_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    bw = bins[1] - bins[0]
    chip = np.zeros(num_bins)

    for i in range(period):
        age = period - 1 - i
        w = decay ** age
        day_vol = v[i] * w
        if day_vol <= 0:
            continue

        dl, dh, dc = lo[i], h[i], c[i]
        if dh <= dl:
            idx = min(np.searchsorted(centers, dc), num_bins - 1)
            chip[idx] += day_vol
        else:
            mask = (centers >= dl) & (centers <= dh)
            idxs = np.where(mask)[0]
            if len(idxs) == 0:
                idx = min(np.searchsorted(centers, dc), num_bins - 1)
                chip[idx] += day_vol
            else:
                max_dist = max(dh - dc, dc - dl, bw)
                wts = np.maximum(1 - np.abs(centers[idxs] - dc) / max_dist, 0.1)
                wts /= wts.sum()
                chip[idxs] += day_vol * wts

    total_chip = chip.sum()
    if total_chip <= 0:
        return None

    pct = chip / total_chip

    above_ratio = pct[centers > current_price].sum()
    profit_ratio = 1 - above_ratio

    cumsum = np.cumsum(pct)
    p5 = min(np.searchsorted(cumsum, 0.05), num_bins - 1)
    p95 = min(np.searchsorted(cumsum, 0.95), num_bins - 1)
    conc_range = centers[p95] - centers[p5]
    concentration_90 = conc_range / current_price

    return {
        "above_ratio": above_ratio,
        "profit_ratio": profit_ratio,
        "concentration_90": concentration_90,
    }


# ---------------------------------------------------------------------------
#  二次精选辅助指标
# ---------------------------------------------------------------------------

def calc_trend_stability(close: pd.Series, period: int = 20) -> dict | None:
    """
    趋势稳定性：用最近 *period* 日收盘价做线性回归，
    返回 R²（拟合度）和 slope（斜率方向）。
    R² 越高说明走势越平滑、趋势越清晰。
    """
    if len(close) < period:
        return None
    y = close.iloc[-period:].values.astype(float)
    x = np.arange(period, dtype=float)
    coeffs = polyfit(x, y, deg=1)      # coeffs[0]=截距, coeffs[1]=斜率
    slope = coeffs[1]
    y_pred = coeffs[0] + coeffs[1] * x
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    return {"r_squared": max(r_squared, 0), "slope": slope}


def calc_bias(close: pd.Series, period: int = 20) -> float | None:
    """乖离率 = (收盘价 − MA) / MA × 100%"""
    ma = calc_ma(close, period)
    ma_val = ma.iloc[-1]
    if np.isnan(ma_val) or ma_val <= 0:
        return None
    return (close.iloc[-1] - ma_val) / ma_val * 100


def calc_volume_trend(volume: pd.Series, period: int = 5) -> dict | None:
    """
    量能持续性：统计最近 *period* 日成交量高于 MA5 的天数，
    以及量能线性回归斜率（正 = 持续放量）。
    """
    if len(volume) < period + 5:
        return None
    vol = volume.astype(float)
    ma5 = calc_ma(vol, 5)
    recent_vol = vol.iloc[-period:].values
    recent_ma = ma5.iloc[-period:].values
    above_count = int(np.sum(recent_vol > recent_ma))

    x = np.arange(period, dtype=float)
    coeffs = polyfit(x, recent_vol, deg=1)
    slope = coeffs[1]
    return {"above_ma_days": above_count, "vol_slope": slope}
