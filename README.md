# A 股智能选股系统

基于公开行情数据，综合多种技术指标对沪深 A 股进行量化评分，自动选出推荐次日买入的前 10 只股票。

## 选股思路

```
股票列表(baostock) + 实时行情(新浪财经) → 基础过滤 → 预筛选 → 历史K线(baostock) → 技术指标评分 → 排名输出
```

### 评分维度（加权满分 100）

每个维度原始分 0~10，通过权重映射到总分 100。新增维度时调整权重即可保持总分不变。

| 维度 | 权重 | 原始分 | 核心逻辑 |
|------|------|--------|----------|
| 均线趋势 | 15 | 0~10 | MA5/10/20/60 多头排列、价格站上均线 |
| 成交量 | 13 | 0~10 | 温和放量、量价配合 |
| MACD | 12 | 0~10 | 金叉信号、柱状线翻红及递增 |
| 筹码分布 | 12 | 0~10 | 上方套牢盘稀疏、获利盘适中、筹码集中度高 |
| 动量 | 10 | 0~10 | 5 日/10 日温和上涨、当日阳线 |
| 布林带 | 10 | 0~10 | 中轨附近或下轨反弹、突破中轨 |
| 换手率 | 10 | 0~10 | 适中换手（2%-8%）、温和放大 |
| RSI | 9 | 0~10 | 超卖反弹区间（30-40 高分）、拐头信号 |
| KDJ | 9 | 0~10 | K 上穿 D 金叉、J 值低位回升 |

### 过滤规则

- 排除 ST、退市股
- 排除北交所（8 开头）和科创板（688 开头）
- 股价区间：3 ~ 100 元
- 排除停牌（成交量为 0）和涨跌停（±9.8%）

## 项目结构

```
stock/
├── main.py            # 主程序入口
├── config.py          # 配置参数（价格区间、线程数等）
├── data.py            # 数据获取（baostock 股票列表/历史K线 + 新浪实时行情）
├── indicators.py      # 技术指标计算（MA/MACD/RSI/KDJ/布林带/筹码分布）
├── scoring.py         # 多维度评分逻辑
├── requirements.txt   # Python 依赖
└── README.md
```

## 快速使用

### 1. 创建并激活虚拟环境

```bash
# 创建虚拟环境
python -m venv stock_sentinel
```

**Windows PowerShell：**

```powershell
# PowerShell 默认禁止运行脚本，需要先修改执行策略（仅当前会话生效）
Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope Process

# 激活虚拟环境
.\stock_sentinel\Scripts\activate
```

> 若想永久修改执行策略，可以用**管理员权限**打开 PowerShell 运行：
> `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope LocalMachine`

**Windows CMD：**

```cmd
stock_sentinel\Scripts\activate.bat
```

**macOS / Linux：**

```bash
source stock_sentinel/bin/activate
```

激活成功后，终端提示符前会出现 `(stock_sentinel)` 前缀。

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 运行选股

```bash
python main.py
```

运行结束后会在终端打印推荐排名，并在 `output/` 目录下自动保存 CSV 文件（如 `output/stock_picks_20260224.csv`）。

## 配置说明

编辑 `config.py` 可调整以下参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `TOP_N` | 10 | 推荐股票数量 |
| `MIN_PRICE` / `MAX_PRICE` | 3.0 / 100.0 | 股价过滤区间（元） |
| `HISTORY_DAYS` | 120 | 历史 K 线天数 |
| `RETRY_TIMES` | 2 | 网络请求重试次数 |
| `SINA_BATCH_SIZE` | 800 | 新浪行情每批请求股票数 |
| `SCORE_WEIGHTS` | (见文件) | 各评分维度权重（总和 = 100） |

## 免责声明

本工具仅供学习研究使用，不构成任何投资建议。股市有风险，投资需谨慎。
