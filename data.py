# -*- coding: utf-8 -*-
"""
数据获取模块

股票列表 + 历史 K 线：baostock（纯本地 TCP 协议，无反爬问题）
实时行情：新浪财经 hq.sinajs.cn（逐只/批量获取，轻量可靠）
"""

import re
import time
import datetime

import requests
import pandas as pd
import baostock as bs

from config import (
    RETRY_TIMES, HISTORY_DAYS, MIN_PRICE, MAX_PRICE,
    PRE_SCREEN_TOP, SINA_BATCH_SIZE, SINA_HEADERS,
)


# ---------------------------------------------------------------------------
#  代码转换工具
# ---------------------------------------------------------------------------

def _bs_to_sina(bs_code: str) -> str:
    """baostock 代码 (sh.600000) → 新浪代码 (sh600000)"""
    return bs_code.replace(".", "")


def _bs_to_pure(bs_code: str) -> str:
    """baostock 代码 (sh.600000) → 纯 6 位数字 (600000)"""
    return bs_code.split(".")[-1]


# ---------------------------------------------------------------------------
#  新浪 hq.sinajs.cn 行情解析
# ---------------------------------------------------------------------------

def _parse_sina_line(line: str) -> dict | None:
    """解析新浪行情接口返回的单行数据。"""
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
            "代码": sina_code[2:],      # 去掉 sh/sz 前缀
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


# ---------------------------------------------------------------------------
#  获取实时行情（baostock 股票列表 + 新浪实时报价）
# ---------------------------------------------------------------------------

def get_all_stocks() -> pd.DataFrame | None:
    """
    1) 用 baostock 拿到当日可交易的 A 股列表（排除科创板/北交所）
    2) 分批请求新浪 hq.sinajs.cn 获取实时行情
    返回 DataFrame，至少包含：代码 / 名称 / 最新价 / 涨跌幅 / 成交量
    """
    print("📊 正在获取 A 股股票列表...")

    # --- baostock 获取股票列表（向前回退最多 7 天寻找最近交易日） ---
    stock_list: list[dict] = []
    for offset in range(8):
        day = (datetime.date.today() - datetime.timedelta(days=offset)).strftime("%Y-%m-%d")
        try:
            rs = bs.query_all_stock(day=day)
        except Exception as e:
            print(f"   query_all_stock({day}) 异常: {e}")
            continue

        if rs.error_code != "0":
            continue

        candidates: list[dict] = []
        while rs.next():
            try:
                row = rs.get_row_data()
            except (UnicodeDecodeError, Exception):
                continue
            code, trade_status, name = row[0], row[1], row[2]
            pure = _bs_to_pure(code)
            if trade_status != "1":
                continue
            if pure.startswith(("6", "0", "3")) and not pure.startswith("688"):
                candidates.append({"bs_code": code, "代码": pure, "名称": name})

        if candidates:
            stock_list = candidates
            print(f"   使用交易日 {day}，获取到 {len(stock_list)} 只 A 股（已排除科创板/北交所）")
            break
        print(f"   {day} 无数据，继续回退...")

    if not stock_list:
        print("   ❌ 回退 7 天仍无法获取股票列表")
        return None

    # --- 新浪 hq.sinajs.cn 批量获取实时行情 ---
    print("📡 正在通过新浪财经获取实时行情...")
    sina_codes = [_bs_to_sina(s["bs_code"]) for s in stock_list]
    realtime_map: dict[str, dict] = {}

    total_batches = (len(sina_codes) + SINA_BATCH_SIZE - 1) // SINA_BATCH_SIZE
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

        batch_idx = i // SINA_BATCH_SIZE + 1
        if batch_idx % 3 == 0 or batch_idx == total_batches:
            pct = min((i + SINA_BATCH_SIZE) / len(sina_codes) * 100, 100)
            print(f"   行情获取进度: {pct:.0f}%")
        time.sleep(0.3)

    # 合并
    rows = []
    for s in stock_list:
        rt = realtime_map.get(s["代码"])
        if rt:
            rt["bs_code"] = s["bs_code"]
            rows.append(rt)

    print(f"   成功获取 {len(rows)} 只股票的实时行情")
    return pd.DataFrame(rows) if rows else None


# ---------------------------------------------------------------------------
#  股票过滤
# ---------------------------------------------------------------------------

def filter_stocks(df: pd.DataFrame) -> pd.DataFrame:
    """排除 ST / 停牌 / 涨跌停 / 价格越界 等股票。"""
    n0 = len(df)

    df = df[~df["名称"].str.contains("ST|st|退", na=False)]
    df = df[(df["最新价"] >= MIN_PRICE) & (df["最新价"] <= MAX_PRICE)]
    df = df[df["成交量"] > 0]
    df = df[(df["涨跌幅"] < 9.8) & (df["涨跌幅"] > -9.8)]

    print(f"   过滤后剩余 {len(df)} 只（排除 {n0 - len(df)} 只）")
    return df


# ---------------------------------------------------------------------------
#  预筛选
# ---------------------------------------------------------------------------

def pre_screen(df: pd.DataFrame, top_n: int = PRE_SCREEN_TOP) -> pd.DataFrame:
    """利用实时行情字段做快速打分，保留 top_n 只候选。"""
    print("🔍 正在进行预筛选...")
    df = df.copy()
    df["pre_score"] = 0.0

    df.loc[(df["涨跌幅"] >= -2) & (df["涨跌幅"] <= 5), "pre_score"] += 3
    df.loc[df["涨跌幅"] > 0, "pre_score"] += 2

    # 成交额 > 5000 万（流动性好）
    if "成交额" in df.columns:
        df.loc[df["成交额"] > 5e7, "pre_score"] += 2

    # 今日阳线
    if "今开" in df.columns:
        df.loc[df["最新价"] > df["今开"], "pre_score"] += 2

    # 没有大幅高开（排除跳空风险）
    if "今开" in df.columns and "昨收" in df.columns:
        df.loc[(df["今开"] / df["昨收"] - 1).abs() < 0.03, "pre_score"] += 1

    df = df.sort_values("pre_score", ascending=False).head(top_n)
    print(f"   预筛选出 {len(df)} 只候选股票")
    return df


# ---------------------------------------------------------------------------
#  历史 K 线（baostock）
# ---------------------------------------------------------------------------

def get_stock_history(bs_code: str, name: str,
                      days: int = HISTORY_DAYS) -> tuple:
    """
    通过 baostock 获取单只股票的前复权日 K 线。
    返回 (纯代码, 名称, DataFrame | None)。
    """
    pure_code = _bs_to_pure(bs_code)
    end_date = datetime.date.today().strftime("%Y-%m-%d")
    start_date = (datetime.date.today() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

    for attempt in range(RETRY_TIMES):
        try:
            rs = bs.query_history_k_data_plus(
                bs_code,
                "date,open,high,low,close,volume,amount,turn",
                start_date=start_date,
                end_date=end_date,
                frequency="d",
                adjustflag="2",     # 前复权
            )
            rows = []
            while rs.error_code == "0" and rs.next():
                rows.append(rs.get_row_data())

            if len(rows) >= 60:
                df = pd.DataFrame(
                    rows,
                    columns=["日期", "开盘", "最高", "最低", "收盘",
                             "成交量", "成交额", "换手率"],
                )
                for col in ["开盘", "最高", "最低", "收盘", "成交量", "成交额", "换手率"]:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                df = df.dropna(subset=["收盘"])
                if len(df) >= 60:
                    return pure_code, name, df

            return pure_code, name, None
        except Exception:
            if attempt < RETRY_TIMES - 1:
                time.sleep(0.5)
    return pure_code, name, None
