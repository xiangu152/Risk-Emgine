# Risk-Engine 风控引擎

> **v1.4.0** | 基于规则引擎 + 机器学习的混合反欺诈风险评分系统

面向金融交易场景的风控引擎，从用户行为日志（登录、交易、设备、IP）中提取 **23 维特征向量**，通过 **16 条反欺诈规则** 和 **ML 模型（Random Forest / XGBoost / Ensemble）** 的混合评分，输出 0~100 风险分及 LOW / MEDIUM / HIGH 等级。

---

## 架构概览

```
CSV 行为日志
       │
       ▼
┌─────────────────────┐
│  feature_extraction  │  23 维特征向量 / 用户
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    scoring_model     │  规则引擎 (16 rules)
│                      │  ML 模型 (Calibrated RF / Ensemble)
│                      │  混合融合 (α 加权)
└─────────┬───────────┘
          │
          ▼
   Risk Score (0~100)  +  Level (LOW / MEDIUM / HIGH)
          │
          ├──▶ CLI 命令行 (main.py)
          ├──▶ Web 仪表盘 (dashboard/)
          └──▶ 评估报告 (evaluate.py)
```

---

## 核心特性

- **23 维特征工程** — 基础 / 速度 / 间隔金额 / 时间循环 / 交互特征，5 大类覆盖常见欺诈模式
- **16 条反欺诈规则** — 设备牧场、撞库、暴力破解、盗号、洗钱、自动化脚本等
- **混合评分** — 规则分数 × α + ML 概率 × (1-α)，默认 α=0.2，支持 `alpha="auto"` 动态调节
- **开箱即用的 ML 模型** — Platt 校准的 Random Forest（593KB），预置在 `models/`
- **Web 仪表盘** — FastAPI + ECharts，CSV 上传即出风险概览、用户详情、特征分布
- **合成数据生成** — 30 种角色模板 + LLM 增强，自动生成带标签的行为日志

---

## 快速开始

### 环境要求

| 项目 | 说明 |
|------|------|
| Python | ≥ 3.12 |
| 包管理 | [uv](https://docs.astral.sh/uv/)（`pip install uv`） |

### 安装

```bash
git clone <repo-url>
cd Risk-Emgine
uv sync
```

### CLI 风险评分

```bash
# 批量模式 — 所有用户聚合为一个特征向量
uv run python main.py data/20260601_20260630_risk.csv

# Per-user 模式 — 每个账户独立评分
uv run python main.py --per-user data/20260601_20260630_risk.csv

# 单用户查询
uv run python main.py --per-user -u ACC_00001 data/20260601_20260630_risk.csv
```

### Python API

```python
from feature_extraction import extract_features, extract_features_per_user
from scoring_model import score_risk, score_risk_hybrid

# 特征提取
features = extract_features("data/20260601_20260630_risk.csv")

# 纯规则评分
result = score_risk(features)
# → {"score": 36.4, "level": "MEDIUM", "level_range": "30~70"}

# ML 混合评分（默认 Calibrated RF）
result = score_risk_hybrid(features, alpha=0.2)
# → {"score": 31.5, "level": "MEDIUM", "rule_score": 36.4, "ml_score": 30.3}
```

### Web 仪表盘

```bash
cd dashboard
uv run uvicorn server:app --reload --port 8000
# 打开 http://localhost:8000
```

### Docker 部署

```bash
# 一键启动
docker compose up -d

# 打开 http://localhost:8000
```

---

## 特征体系（23 维）

### 基础特征（7 个，v1.0）

| 特征键 | 值域 | 含义 |
|--------|------|------|
| `device_reuse_ratio` | 0~1 | 设备复用率 |
| `ip_change_freq` | 0~1 | IP 变更频率 |
| `tx_freq` | ≥0 | 交易次数 |
| `login_fail_ratio` | 0~1 | 登录失败率 |
| `amount_anomaly_score` | 0~1 | 金额异常（MAD Z-score） |
| `behavior_time_anomaly` | 0~1 | 行为耗时异常（<5s 占比） |
| `multi_region_risk` | 0/1 | 多地区风险 |

### 速度特征（4 个，v1.2）

| 特征键 | 值域 | 含义 |
|--------|------|------|
| `tx_velocity_5min` | ≥0 | 5 分钟窗口最大交易数 |
| `tx_velocity_1h` | ≥0 | 1 小时窗口最大交易数 |
| `device_switch_24h` | ≥0 | 24h 窗口最大不同设备数 |
| `ip_switch_24h` | ≥0 | 24h 窗口最大不同 IP 数 |

### 间隔与金额特征（4 个，v1.2）

| 特征键 | 值域 | 含义 |
|--------|------|------|
| `tx_interval_mean_sec` | ≥0 | 平均交易间隔（秒） |
| `tx_burst_ratio` | 0~1 | 60s 内连续交易占比 |
| `amount_user_deviation` | ≥0 | 用户均值偏离全局中位数的 Z-score |
| `amount_user_max_ratio` | ≥0 | 最大单笔 / 用户均值 |

### 时间循环特征（2 个，v1.2）

| 特征键 | 值域 | 含义 |
|--------|------|------|
| `night_tx_ratio` | 0~1 | 凌晨 1am~5am 交易占比 |
| `odd_hour_tx_ratio` | 0~1 | 非工作时间交易占比 |

### 交互特征（6 个，v1.3）⚠️ 最重要

| 特征键 | 值域 | 含义 |
|--------|------|------|
| `device_ip_risk` | 0~1 | 设备 + IP 同时切换的复合风险 |
| `night_automation` | 0~1 | 凌晨 + 快速操作的自动化脚本风险 |
| `velocity_amount` | ≥0 | 高频交易 × 金额异常（RF 重要性 #1） |
| `burst_max_ratio` | 0~1 | 交易突发 × 大额比例 |
| `velocity_ratio` | ≥0 | 短窗口 / 长窗口交易比例 |
| `amount_exposure` | ≥0 | 总交易金额暴露度 |

---

## 风险等级

| 等级 | 分数 | level 值 | 建议动作 |
|------|------|----------|---------|
| 低风险 | 0~29 | `"LOW"` | 放行 |
| 中风险 | 30~70 | `"MEDIUM"` | 人工复核 |
| 高风险 | 71~100 | `"HIGH"` | 拦截 / 告警 |

---

## 模型性能（v1.4.0）

测试集 15 账户，验证集 16 账户，按账户 69/16/15 分层划分。

### 模型预测

| 模型 | Val Acc | Test Acc | Test Spearman | Cost | 大小 |
|------|:-------:|:--------:|:-------------:|:----:|:----:|
| **Calibrated RF（默认）** | 75.0% | **80.0%** | **0.825** | **4** | 593KB |
| Ensemble (16 特征) | 68.8% | 80.0% | 0.854 | 7 | 760KB |
| Ensemble (23 特征) | 68.8% | 80.0% | 0.854 | 7 | 1.5MB |

### 端到端评分

| 评分方式 | Val Acc | Test Acc | Test Kappa | Cost |
|----------|:-------:|:--------:|:----------:|:----:|
| 纯规则 `score_risk()` | **87.5%** | 60.0% | 0.237 | 12 |
| 混合 α=0.2（默认） | 87.5% | **80.0%** | **0.605** | 8 |
| 混合 α=auto | 87.5% | 73.3% | 0.492 | 9 |

> 代价矩阵: 漏报 HIGH=10, MEDIUM=2; 误报 LOW=1, MEDIUM=3

### 速度

| 操作 | 耗时 |
|------|------|
| 全量特征提取 (100 账户) | ~840 ms |
| 单账户特征提取 | ~8 ms |
| 规则打分 | **5 µs** |
| ML 推理 (Calibrated RF) | **26 ms** |
| 单账户端到端 (规则) | **8 ms** |
| 单账户端到端 (ML) | **34 ms** |

---

## 项目结构

```
Risk-Emgine/
├── main.py                    # CLI 入口
├── feature_extraction.py      # 23 维特征提取
├── scoring_model.py           # 规则引擎 + ML 混合评分
├── train_ml_model.py          # ML 模型训练管线
├── evaluate.py                # 统一评估 CLI
│
├── models/                    # 预训练模型（开箱即用）
│   ├── calibrated_rf.pkl      # ← 默认模型
│   ├── ensemble_selected.pkl  # 16 特征 Ensemble
│   ├── ensemble.pkl           # 23 特征 Ensemble
│   ├── scaler.pkl             # StandardScaler
│   └── rfecv_selector.pkl     # 特征选择器
│
├── scripts/                   # 开发工具脚本
│   ├── benchmark.py           # 性能基准测试
│   ├── calibrate_thresholds.py # 阈值校准
│   ├── optimize_thresholds.py # 阈值优化
│   ├── shap_analysis.py       # SHAP 特征重要性分析
│   ├── run_eval.py            # Per-account 评估对比
│   └── generate_behavior_data.py # 测试数据生成
│
├── dashboard/                 # Web 可视化面板
│   ├── server.py              # FastAPI 后端 (6 个 API)
│   └── static/index.html      # ECharts 前端
│
├── module/data_creater/       # 合成数据生成子模块
│   ├── main.py                # 生成入口
│   ├── prompts/role_templates.py  # 30 种角色模板
│   ├── api/llm.py             # LLM API 客户端
│   └── script/generate.py     # 异步生成编排
│
├── tests/                     # 测试套件 (40+ cases)
│   ├── test_feature_extraction.py
│   ├── test_scoring_model.py
│   └── test_api_integration.py
│
├── notebooks/                 # Jupyter 分析笔记本
└── API_README.md              # API 详细文档
```

---

## 仪表盘 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/upload` | 上传 CSV，提取特征并评分 |
| `GET` | `/api/overview` | 全局统计：账户数、分数分布、等级分布 |
| `GET` | `/api/users` | 所有用户的特征、规则/ML/混合分数 |
| `GET` | `/api/users/{id}` | 单用户详情：特征、规则命中、行为时间线 |
| `GET` | `/api/features/weights` | 23 维特征权重 |
| `GET` | `/api/features/distribution` | 按风险等级分组的特征分布统计 |

---

## 训练与评估

```bash
# 训练 ML 模型（需 data/splits/{train,val,test}.csv）
uv run python train_ml_model.py

# 统一评估
uv run python evaluate.py full          # 全量评估
uv run python evaluate.py splits        # 分集评估
uv run python evaluate.py hybrid        # α 参数扫描
uv run python evaluate.py calibrate     # 阈值网格搜索
uv run python evaluate.py all           # 全部评估，输出 HTML/JSON

# 运行测试
uv run pytest tests/ -v
```

### 开发工具脚本

`scripts/` 目录包含开发和分析工具：

```bash
# 性能基准测试
uv run python scripts/benchmark.py

# 阈值校准与优化
uv run python scripts/calibrate_thresholds.py
uv run python scripts/optimize_thresholds.py

# SHAP 特征分析
uv run python scripts/shap_analysis.py

# 模型评估对比
uv run python scripts/run_eval.py
```

---

## 合成数据生成

```bash
cd module/data_creater
# 需要 .env 配置 LLM API 凭据（见 module/data_creater/README.md）
python main.py 20260601 20260630 ./data train 100
```

30 种角色模板（12 低风险 / 10 中风险 / 8 高风险），LLM 增强表面细节（姓名、设备型号、行为描述）。

---

## 已知局限

| 局限 | 原因 | 缓解措施 |
|------|------|---------|
| 训练集仅 69 账户 | 模拟数据共 100 账户 | Platt 校准 + 特征选择 23→16 缓解过拟合 |
| 7 个特征被 RFECV 剔除 | 模拟数据特征分布简单 | 保留在代码中，真实数据会激活 |
| HIGH 类仅 7 样本 | 数据生成的角色故事特性 | Cost-sensitive 评估补偿 |
| 无实时打分能力 | 当前为批处理模式 | 单次 <35ms，API 封装即可 |

---

## 许可证

私有项目。
