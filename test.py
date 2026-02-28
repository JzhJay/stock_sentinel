#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A股智能选股脚本
================
基于公开行情数据，综合多种技术指标对A股进行评分，
选出推荐次日买入的前10支股票。

评分维度：
  1. 均线趋势 (MA5/MA10/MA20/MA60 多头排列)
  2. MACD 金叉 / 柱状线翻红
  3. 成交量趋势 (温和放量)
  4. RSI 指标 (超卖反弹区间)
  5. KDJ 金叉信号
  6. 布林带位置 (下轨附近反弹)
  7. 近期涨幅动量
  8. 换手率活跃度

"""

import os
os.environ["NO_PROXY"] = "*"

import warnings
warnings.filterwarnings("ignore")

import re
import sys
import time
import datetime
import numpy as np
import pandas as pd
import requests
import baostock as bs

# ==================== 配置参数 ====================
TOP_N = 10                  # 最终推荐股票数量
HISTORY_DAYS = 120          # 获取历史数据的天数
MIN_PRICE = 3.0             # 最低股价过滤
MAX_PRICE = 100.0           # 最高股价过滤（排除高价股）
RETRY_TIMES = 2             # 数据获取重试次数
SINA_BATCH_SIZE = 800       # 新浪接口每批请求股票数

# ==================== 技术指标计算 ====================

def calc_ma(close: pd.Series, period: int) -> pd.Series:
    """计算移动平均线"""
    return close.rolling(window=period, min_periods=period).mean()


def calc_ema(close: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线"""
    return close.ewm(span=period, adjust=False).mean()


def calc_macd(close: pd.Series, fast=12, slow=26, signal=9):
    """计算 MACD 指标"""
    ema_fast = calc_ema(close, fast)
    ema_slow = calc_ema(close, slow)
    dif = ema_fast - ema_slow
    dea = calc_ema(dif, signal)
    macd_hist = 2 * (dif - dea)
    return dif, dea, macd_hist


def calc_rsi(close: pd.Series, period=14) -> pd.Series:
    """计算 RSI 指标"""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calc_kdj(high: pd.Series, low: pd.Series, close: pd.Series, n=9, m1=3, m2=3):
    """计算 KDJ 指标"""
    low_n = low.rolling(window=n, min_periods=n).min()
    high_n = high.rolling(window=n, min_periods=n).max()
    rsv = (close - low_n) / (high_n - low_n).replace(0, np.nan) * 100
    k = rsv.ewm(com=m1 - 1, adjust=False).mean()
    d = k.ewm(com=m2 - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return k, d, j


def calc_bollinger(close: pd.Series, period=20, num_std=2):
    """计算布林带"""
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return upper, mid, lower


# ==================== 评分函数 ====================

def score_stock(df: pd.DataFrame) -> dict:
    """
    对单只股票进行多维度评分
    返回各项分数和总分
    """
    if df is None or len(df) < 60:
        return None

    close = df["收盘"].astype(float)
    high = df["最高"].astype(float)
    low = df["最低"].astype(float)
    volume = df["成交量"].astype(float)
    turnover_rate = df["换手率"].astype(float) if "换手率" in df.columns else None

    scores = {}
    total = 0.0

    # ---------- 1. 均线趋势评分 (满分 20) ----------
    ma5 = calc_ma(close, 5)
    ma10 = calc_ma(close, 10)
    ma20 = calc_ma(close, 20)
    ma60 = calc_ma(close, 60)

    ma_score = 0
    latest = len(close) - 1

    # 价格站上各均线
    if close.iloc[latest] > ma5.iloc[latest]:
        ma_score += 3
    if close.iloc[latest] > ma10.iloc[latest]:
        ma_score += 3
    if close.iloc[latest] > ma20.iloc[latest]:
        ma_score += 3
    if not np.isnan(ma60.iloc[latest]) and close.iloc[latest] > ma60.iloc[latest]:
        ma_score += 3

    # 短期均线向上
    if ma5.iloc[latest] > ma5.iloc[latest - 3]:
        ma_score += 2
    if ma10.iloc[latest] > ma10.iloc[latest - 3]:
        ma_score += 2

    # 多头排列: MA5 > MA10 > MA20
    if ma5.iloc[latest] > ma10.iloc[latest] > ma20.iloc[latest]:
        ma_score += 4

    ma_score = min(ma_score, 20)
    scores["均线趋势"] = ma_score
    total += ma_score

    # ---------- 2. MACD 评分 (满分 15) ----------
    dif, dea, macd_hist = calc_macd(close)
    macd_score = 0

    # MACD 柱状线为正（多头力量）
    if macd_hist.iloc[latest] > 0:
        macd_score += 4

    # MACD 柱状线由负转正（刚翻红）
    if macd_hist.iloc[latest] > 0 and macd_hist.iloc[latest - 1] <= 0:
        macd_score += 5

    # MACD 柱状线递增
    if macd_hist.iloc[latest] > macd_hist.iloc[latest - 1]:
        macd_score += 3

    # DIF > DEA (金叉状态)
    if dif.iloc[latest] > dea.iloc[latest]:
        macd_score += 3

    macd_score = min(macd_score, 15)
    scores["MACD"] = macd_score
    total += macd_score

    # ---------- 3. 成交量趋势评分 (满分 15) ----------
    vol_score = 0
    vol_ma5 = calc_ma(volume, 5)
    vol_ma10 = calc_ma(volume, 10)

    # 近5日成交量均值 > 近10日均值 (温和放量)
    if vol_ma5.iloc[latest] > vol_ma10.iloc[latest]:
        vol_score += 4

    # 今日量比 > 1 (当日放量)
    if volume.iloc[latest] > vol_ma5.iloc[latest]:
        vol_score += 3

    # 量价配合: 价涨量增
    price_up = close.iloc[latest] > close.iloc[latest - 1]
    vol_up = volume.iloc[latest] > volume.iloc[latest - 1]
    if price_up and vol_up:
        vol_score += 4

    # 温和放量(量比1~3倍)，不是暴量
    if not np.isnan(vol_ma5.iloc[latest]) and vol_ma5.iloc[latest] > 0:
        vol_ratio = volume.iloc[latest] / vol_ma5.iloc[latest]
        if 1.0 < vol_ratio < 3.0:
            vol_score += 4
        elif vol_ratio >= 3.0:
            vol_score -= 2  # 暴量扣分

    vol_score = max(min(vol_score, 15), 0)
    scores["成交量"] = vol_score
    total += vol_score

    # ---------- 4. RSI 评分 (满分 10) ----------
    rsi = calc_rsi(close, 14)
    rsi_score = 0
    rsi_val = rsi.iloc[latest]

    if not np.isnan(rsi_val):
        # RSI 在 40~60 区间（健康区间）
        if 40 <= rsi_val <= 60:
            rsi_score += 5
        # RSI 在 30~40 区间（超卖回升机会）
        elif 30 <= rsi_val < 40:
            rsi_score += 7
        # RSI 从超卖区回升
        elif 20 <= rsi_val < 30:
            rsi_score += 4
        # RSI > 70 超买（扣分）
        elif rsi_val > 80:
            rsi_score -= 3

        # RSI 向上拐头
        if rsi.iloc[latest] > rsi.iloc[latest - 1] and rsi.iloc[latest - 1] < rsi.iloc[latest - 2]:
            rsi_score += 3

    rsi_score = max(min(rsi_score, 10), 0)
    scores["RSI"] = rsi_score
    total += rsi_score

    # ---------- 5. KDJ 评分 (满分 10) ----------
    k, d, j = calc_kdj(high, low, close)
    kdj_score = 0

    if not np.isnan(k.iloc[latest]):
        # K上穿D (金叉)
        if k.iloc[latest] > d.iloc[latest] and k.iloc[latest - 1] <= d.iloc[latest - 1]:
            kdj_score += 6
        # K > D (多头状态)
        elif k.iloc[latest] > d.iloc[latest]:
            kdj_score += 3

        # J值在合理范围 (20~80)
        if 20 <= j.iloc[latest] <= 80:
            kdj_score += 2

        # J值从低位回升
        if j.iloc[latest] > j.iloc[latest - 1] and j.iloc[latest - 1] < 30:
            kdj_score += 2

    kdj_score = max(min(kdj_score, 10), 0)
    scores["KDJ"] = kdj_score
    total += kdj_score

    # ---------- 6. 布林带评分 (满分 10) ----------
    upper, mid, lower = calc_bollinger(close)
    boll_score = 0

    if not np.isnan(lower.iloc[latest]):
        price = close.iloc[latest]
        boll_width = upper.iloc[latest] - lower.iloc[latest]
        if boll_width > 0:
            position = (price - lower.iloc[latest]) / boll_width

            # 价格在布林中轨附近或略上方
            if 0.4 <= position <= 0.7:
                boll_score += 5
            # 价格从下轨反弹
            elif 0.1 <= position < 0.4:
                boll_score += 4
            # 价格接近上轨 (压力)
            elif position > 0.9:
                boll_score -= 2

            # 价格突破中轨
            if price > mid.iloc[latest] and close.iloc[latest - 1] <= mid.iloc[latest - 1]:
                boll_score += 5

    boll_score = max(min(boll_score, 10), 0)
    scores["布林带"] = boll_score
    total += boll_score

    # ---------- 7. 近期动量评分 (满分 10) ----------
    momentum_score = 0

    # 5日涨幅
    pct_5d = (close.iloc[latest] - close.iloc[latest - 5]) / close.iloc[latest - 5] * 100
    # 10日涨幅
    pct_10d = (close.iloc[latest] - close.iloc[latest - 10]) / close.iloc[latest - 10] * 100

    # 5日温和上涨 (1%~8%)
    if 1 <= pct_5d <= 8:
        momentum_score += 4
    elif 0 < pct_5d < 1:
        momentum_score += 2
    elif pct_5d > 8:
        momentum_score += 1  # 涨太多，降低评分

    # 10日涨幅温和
    if 2 <= pct_10d <= 15:
        momentum_score += 3
    elif 0 < pct_10d < 2:
        momentum_score += 1

    # 今日阳线
    if close.iloc[latest] > df["开盘"].astype(float).iloc[latest]:
        momentum_score += 3

    momentum_score = max(min(momentum_score, 10), 0)
    scores["动量"] = momentum_score
    total += momentum_score

    # ---------- 8. 换手率评分 (满分 10) ----------
    tr_score = 0
    if turnover_rate is not None and not turnover_rate.empty:
        tr = turnover_rate.iloc[latest]
        tr_ma5 = calc_ma(turnover_rate, 5).iloc[latest]
        if not np.isnan(tr):
            # 换手率适中 (2%~8%)
            if 2 <= tr <= 8:
                tr_score += 5
            elif 1 <= tr < 2:
                tr_score += 3
            elif tr > 15:
                tr_score -= 2  # 换手过高

            # 换手率放大但不暴增
            if not np.isnan(tr_ma5) and tr_ma5 > 0:
                tr_ratio = tr / tr_ma5
                if 1.0 < tr_ratio < 2.5:
                    tr_score += 5
                elif tr_ratio >= 2.5:
                    tr_score += 1

    tr_score = max(min(tr_score, 10), 0)
    scores["换手率"] = tr_score
    total += tr_score

    scores["总分"] = total
    return scores


# ==================== 数据获取（新浪财经 + baostock） ====================

SINA_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Referer": "https://finance.sina.com.cn",
}


def _code_to_sina(code: str) -> str:
    """baostock 代码 (sh.600000) -> 新浪代码 (sh600000)"""
    return code.replace(".", "")


def _code_to_pure(code: str) -> str:
    """baostock 代码 (sh.600000) -> 纯数字代码 (600000)"""
    return code.split(".")[-1]


def _parse_sina_line(line: str) -> dict | None:
    """解析新浪行情接口返回的单行数据"""
    m = re.match(r'var hq_str_(\w+)="(.*)";', line.strip())
    if not m or not m.group(2):
        return None
    sina_code = m.group(1)
    fields = m.group(2).split(",")
    if len(fields) < 32:
        return None
    try:
        yesterday_close = float(fields[2])
        current_price = float(fields[3])
        if yesterday_close <= 0 or current_price <= 0:
            return None
        change_pct = (current_price - yesterday_close) / yesterday_close * 100
        return {
            "代码": sina_code[2:],
            "名称": fields[0],
            "最新价": current_price,
            "今开": float(fields[1]),
            "昨收": yesterday_close,
            "最高": float(fields[4]),
            "最低": float(fields[5]),
            "成交量": float(fields[8]),
            "成交额": float(fields[9]),
            "涨跌幅": round(change_pct, 2),
            "sina_code": sina_code,
        }
    except (ValueError, IndexError):
        return None


def get_all_stocks():
    """获取A股所有股票列表及实时行情（baostock 股票列表 + 新浪实时行情）"""
    print("📊 正在获取A股股票列表...")
    try:
        lg = bs.login()
        if lg.error_code != "0":
            print(f"❌ baostock 登录失败: {lg.error_msg}")
            return None

        today = datetime.date.today().strftime("%Y-%m-%d")
        rs = bs.query_all_stock(day=today)
        if rs.error_code != "0":
            yesterday = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            rs = bs.query_all_stock(day=yesterday)

        stock_list = []
        while rs.error_code == "0" and rs.next():
            row = rs.get_row_data()
            code, trade_status, name = row[0], row[1], row[2]
            pure_code = _code_to_pure(code)
            if trade_status != "1":
                continue
            if pure_code.startswith("6") or pure_code.startswith("0") or pure_code.startswith("3"):
                if not pure_code.startswith("688"):
                    stock_list.append({"bs_code": code, "代码": pure_code, "名称": name})

        print(f"   获取到 {len(stock_list)} 只A股（已排除科创板/北交所）")

        if not stock_list:
            return None

        print("📡 正在通过新浪财经获取实时行情...")
        sina_codes = [_code_to_sina(s["bs_code"]) for s in stock_list]
        realtime_map = {}

        for i in range(0, len(sina_codes), SINA_BATCH_SIZE):
            batch = sina_codes[i:i + SINA_BATCH_SIZE]
            url = "https://hq.sinajs.cn/list=" + ",".join(batch)
            for attempt in range(RETRY_TIMES):
                try:
                    r = requests.get(url, headers=SINA_HEADERS, timeout=15)
                    if r.status_code == 200:
                        for line in r.text.strip().split("\n"):
                            parsed = _parse_sina_line(line)
                            if parsed:
                                realtime_map[parsed["代码"]] = parsed
                        break
                except Exception:
                    if attempt < RETRY_TIMES - 1:
                        time.sleep(1)
            if (i // SINA_BATCH_SIZE + 1) % 3 == 0:
                pct = min((i + SINA_BATCH_SIZE) / len(sina_codes) * 100, 100)
                print(f"   行情获取进度: {pct:.0f}%")
            time.sleep(0.3)

        rows = []
        for s in stock_list:
            rt = realtime_map.get(s["代码"])
            if rt:
                rt["bs_code"] = s["bs_code"]
                rows.append(rt)

        print(f"   成功获取 {len(rows)} 只股票的实时行情")
        return pd.DataFrame(rows)

    except Exception as e:
        print(f"❌ 获取股票数据失败: {e}")
        return None


def filter_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """过滤股票：排除ST、停牌、价格不在范围内的股票"""
    original_count = len(df)

    mask = ~df["名称"].str.contains("ST|st|退", na=False)
    df = df[mask]

    df = df[(df["最新价"] >= MIN_PRICE) & (df["最新价"] <= MAX_PRICE)]
    df = df[df["成交量"] > 0]
    df = df[(df["涨跌幅"] < 9.8) & (df["涨跌幅"] > -9.8)]

    filtered_count = len(df)
    print(f"   过滤后剩余 {filtered_count} 只股票（过滤了 {original_count - filtered_count} 只）")
    return df


def get_stock_history(bs_code: str, name: str, days: int = HISTORY_DAYS) -> tuple:
    """通过 baostock 获取单只股票的历史K线数据"""
    pure_code = _code_to_pure(bs_code)
    for attempt in range(RETRY_TIMES):
        try:
            end_date = datetime.date.today().strftime("%Y-%m-%d")
            start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",  # 前复权
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())

            if len(rows) >= 60:
                df = pd.DataFrame(rows, columns=["日期", "开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"])
                for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["收盘"])
                if len(df) >= 60:
                    return pure_code, name, df

            return pure_code, name, None
        except Exception:
            if attempt < RETRY_TIMES - 1:
                time.sleep(0.5)
            continue
    return pure_code, name, None


# ==================== 预筛选（使用实时数据快速过滤） ====================

def pre_screen(df: pd.DataFrame, top_n: int = 200) -> pd.DataFrame:
    """使用实时行情数据进行初步筛选，减少后续需要下载历史数据的股票数量"""
    print("🔍 正在进行预筛选...")
    df = df.copy()

    df["pre_score"] = 0.0

    mask = (df["涨跌幅"] >= -2) & (df["涨跌幅"] <= 5)
    df.loc[mask, "pre_score"] += 3

    mask = df["涨跌幅"] > 0
    df.loc[mask, "pre_score"] += 2

    # 成交额大于5000万（流动性好）
    mask = df["成交额"] > 5e7
    df.loc[mask, "pre_score"] += 2

    # 今日阳线
    mask = df["最新价"] > df["今开"]
    df.loc[mask, "pre_score"] += 2

    # 没有大幅高开（排除跳空风险）
    mask = (df["今开"] / df["昨收"] - 1).abs() < 0.03
    df.loc[mask, "pre_score"] += 1

    df = df.sort_values("pre_score", ascending=False).head(top_n)
    print(f"   预筛选出 {len(df)} 只候选股票")
    return df


# ==================== 主流程 ====================

def main():
    print("=" * 70)
    print("          🏦 A股智能选股系统 — 次日买入推荐")
    print("=" * 70)
    print(f"  运行时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据来源: 新浪财经(实时行情) + baostock(历史K线)")
    print(f"  评分维度: 均线趋势/MACD/成交量/RSI/KDJ/布林带/动量/换手率")
    print(f"  价格区间: {MIN_PRICE} ~ {MAX_PRICE} 元")
    print(f"  推荐数量: 前 {TOP_N} 只")
    print("=" * 70)
    print()

    # Step 1: 获取所有A股实时行情
    all_stocks = get_all_stocks()
    if all_stocks is None or all_stocks.empty:
        print("❌ 无法获取行情数据，请检查网络连接")
        sys.exit(1)

    # Step 2: 基础过滤
    filtered = filter_stocks(all_stocks)
    if filtered.empty:
        print("❌ 过滤后没有符合条件的股票")
        sys.exit(1)

    # Step 3: 预筛选
    candidates = pre_screen(filtered, top_n=200)

    # Step 4: 获取候选股票的历史数据并评分
    print(f"\n📈 正在获取候选股票的历史K线数据并评分（共 {len(candidates)} 只）...")
    print("   这可能需要几分钟，请耐心等待...\n")

    results = []
    completed = 0
    failed = 0
    total = len(candidates)

    for _, row in candidates.iterrows():
        bs_code = str(row["bs_code"])
        code = str(row["代码"])
        name = str(row["名称"])

        code, name, hist_df = get_stock_history(bs_code, name)
        completed += 1

        if hist_df is not None and len(hist_df) >= 60:
            try:
                score_result = score_stock(hist_df)
                if score_result and score_result["总分"] > 0:
                    latest_price = hist_df["收盘"].iloc[-1]
                    latest_change = (
                        (hist_df["收盘"].iloc[-1] - hist_df["收盘"].iloc[-2])
                        / hist_df["收盘"].iloc[-2] * 100
                    )
                    results.append({
                        "代码": code,
                        "名称": name,
                        "最新价": round(latest_price, 2),
                        "涨跌幅%": round(latest_change, 2),
                        **score_result
                    })
            except Exception:
                failed += 1
        else:
            failed += 1

        if completed % 20 == 0 or completed == total:
            progress = completed / total * 100
            print(f"   进度: {completed}/{total} ({progress:.0f}%)  "
                  f"有效: {len(results)}  失败: {failed}")

    if not results:
        print("\n❌ 没有找到符合条件的股票")
        sys.exit(1)

    # Step 5: 排序并输出结果
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("总分", ascending=False)

    print("\n" + "=" * 70)
    print(f"  🏆 推荐次日买入的前 {TOP_N} 只股票")
    print("=" * 70)

    top_stocks = result_df.head(TOP_N)

    # 格式化输出
    print(f"\n{'排名':>4}  {'代码':<8} {'名称':<8} {'最新价':>7} {'涨跌幅%':>7} "
          f"{'总分':>5} {'均线':>4} {'MACD':>5} {'成交量':>5} {'RSI':>4} "
          f"{'KDJ':>4} {'布林':>4} {'动量':>4} {'换手':>4}")
    print("-" * 100)

    for rank, (_, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"{rank:>4}  {row['代码']:<8} {row['名称']:<8} "
              f"{row['最新价']:>7.2f} {row['涨跌幅%']:>+7.2f} "
              f"{row['总分']:>5.0f} {row['均线趋势']:>4.0f} {row['MACD']:>5.0f} "
              f"{row['成交量']:>5.0f} {row['RSI']:>4.0f} "
              f"{row['KDJ']:>4.0f} {row['布林带']:>4.0f} {row['动量']:>4.0f} {row['换手率']:>4.0f}")

    # 输出详细分析
    print("\n" + "=" * 70)
    print("  📋 详细分析")
    print("=" * 70)

    for rank, (_, row) in enumerate(top_stocks.iterrows(), 1):
        print(f"\n  第{rank}名: {row['代码']} {row['名称']}  "
              f"最新价: {row['最新价']:.2f}  涨跌幅: {row['涨跌幅%']:+.2f}%  "
              f"总分: {row['总分']:.0f}/100")

        details = []
        if row["均线趋势"] >= 15:
            details.append("✅ 均线多头排列")
        elif row["均线趋势"] >= 10:
            details.append("🔶 均线趋势偏多")

        if row["MACD"] >= 10:
            details.append("✅ MACD金叉/柱状线翻红")
        elif row["MACD"] >= 7:
            details.append("🔶 MACD偏多")

        if row["成交量"] >= 10:
            details.append("✅ 量价配合良好")
        elif row["成交量"] >= 7:
            details.append("🔶 成交量温和放大")

        if row["KDJ"] >= 6:
            details.append("✅ KDJ金叉信号")

        if row["动量"] >= 7:
            details.append("✅ 近期涨势良好")

        if details:
            print(f"    {'  |  '.join(details)}")

    # 保存结果到CSV
    output_file = f"stock_picks_{datetime.date.today().strftime('%Y%m%d')}.csv"
    result_df.head(TOP_N).to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"\n💾 结果已保存到: {output_file}")

    # 免责声明
    print("\n" + "=" * 70)
    print("  ⚠️  免责声明")
    print("=" * 70)
    print("  本工具仅供学习研究使用，不构成任何投资建议。")
    print("  股市有风险，投资需谨慎。任何投资决策请自行判断。")
    print("  历史数据不代表未来表现，技术指标仅供参考。")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            bs.logout()
        except Exception:
            pass
