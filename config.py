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

# ---------- 评分权重（总和必须 = 100） ----------
# 每个维度原始分 0~10，加权后 sum = 100
# 新增维度时：在此字典中添加 key，并调低其他权重使总和仍为 100
SCORE_WEIGHTS = {
    "均线趋势":  15,     # 趋势为王，多头排列是最强持续信号
    "MACD":      12,     # 趋势 + 动量双重确认
    "成交量":    13,     # 量在价先，放量验证不可或缺
    "筹码分布":  12,     # 上方抛压直接影响买点时机
    "动量":      10,     # 近期涨势强弱
    "布林带":    10,     # 位置分析 / 突破回踩
    "RSI":        9,     # 超买超卖辅助
    "KDJ":        9,     # 金叉 / 超卖信号辅助（与 RSI 角色相近）
    "换手率":    10,     # 活跃度 / 流动性确认
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
