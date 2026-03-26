# -*- coding: utf-8 -*-
"""全局配置参数"""

import os

# 禁用系统代理，避免 baostock / 新浪请求走代理后失败
os.environ.setdefault("NO_PROXY", "*")

TOP_N = 10              # 最终推荐股票数量
HISTORY_DAYS = 120      # 获取历史数据的天数
MIN_PRICE = 3.0         # 最低股价过滤
MAX_PRICE = 100.0       # 最高股价过滤（排除高价股）
MAX_WORKERS = 8         # 并发线程数（仅用于历史 K 线批量获取）
RETRY_TIMES = 2         # 数据获取重试次数
PRE_SCREEN_TOP = 200    # 预筛选保留的候选数量

# ---------- 止盈止损参数 ----------
# 止损：按趋势强度自适应 ATR 倍数
EXIT_ATR_MULT_STRONG = 2.8       # 强趋势 (R²高且向上)
EXIT_ATR_MULT_BASE = 2.0         # 普通趋势
EXIT_ATR_MULT_WEAK = 1.6         # 弱趋势/震荡
EXIT_TREND_R2_STRONG = 0.75
EXIT_TREND_R2_WEAK = 0.35

# 止损保护带：避免止损过近或过远
EXIT_MIN_STOP_PCT = 3.0          # 最小止损幅度（%）
EXIT_MAX_STOP_PCT = 12.0         # 最大止损幅度（%）

# 止盈目标（ATR法）
EXIT_TP1_ATR_MULT = 1.5
EXIT_TP2_ATR_MULT = 3.0

# 风险收益比：TP1/TP2 加权
EXIT_RR_TP1_WEIGHT = 0.4
EXIT_RR_TP2_WEIGHT = 0.6

# 触发止盈①后的移动止损建议
EXIT_TRAIL_BUFFER_ATR = 0.8      # 成本价上方锁盈缓冲（ATR 倍数）
EXIT_TRAIL_ATR_MULT = 2.0        # 跟踪止损 ATR 倍数（与 MA10 取更高）

# ---------- 评分权重（总和必须 = 100） ----------
# 每个维度原始分 0~10，加权后 sum = 100
# 核心偏好维度占大头，其余维度作为加分项
SCORE_WEIGHTS = {
    "均线趋势":  30,     # 核心：均线向上发散 + 多头结构
    "筹码分布":  25,     # 核心：上方筹码越稀疏越优
    "相对低位":  15,     # 核心：横向比较处于相对低位
    "MACD":       8,     # 其余维度作为加分项
    "成交量":     7,
    "布林带":     5,
    "动量":       4,
    "换手率":     3,
    "RSI":        2,
    "KDJ":        1,
}
assert sum(SCORE_WEIGHTS.values()) == 100, "SCORE_WEIGHTS 总和必须为 100"

# ---------- 二次精选权重（总和必须 = 50） ----------
# 从 Top-10 中精选 Top-3 的补充维度，每个维度原始分 0~10
ROUND2_WEIGHTS = {
    "风险收益比":  12,    # 止盈空间 / 止损空间，衡量入场性价比
    "趋势稳定性":  10,    # R² 线性回归拟合度，越平滑趋势越可靠
    "乖离率":       9,    # 偏离 MA20 程度，过高有回调风险
    "量能持续性":  10,    # 近 5 日放量持续性，优于脉冲式放量
    "多周期共振":   9,    # 日线 + 周线级别趋势方向一致
}
assert sum(ROUND2_WEIGHTS.values()) == 50, "ROUND2_WEIGHTS 总和必须为 50"
assert EXIT_MIN_STOP_PCT < EXIT_MAX_STOP_PCT, "止损幅度上下限配置错误"
assert abs((EXIT_RR_TP1_WEIGHT + EXIT_RR_TP2_WEIGHT) - 1.0) < 1e-9, "风险收益比权重和必须为 1"

FINAL_TOP = 3   # 二次精选最终推荐数量

# 新浪 hq.sinajs.cn 每批请求的股票数量
SINA_BATCH_SIZE = 800

# 新浪财经请求头
SINA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    ),
    "Referer": "https://finance.sina.com.cn",
}
