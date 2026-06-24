# 风控引擎 API 使用手册

> **版本 v1.4.0** | 特征提取 + 风险打分模块  
> 给前后端组员的集成参考 | 23 特征 · 16 规则 · Calibrated RF 默认模型

## 环境

```bash
# 安装依赖（仅首次）
uv sync

# 运行测试
uv run pytest tests/ -v
```

| 项目 | 说明 |
|------|------|
| Python | ≥3.12 |
| 依赖管理 | `uv`（`pip install uv` 安装） |
| 依赖声明 | `pyproject.toml` |

---

## 依赖

`pyproject.toml` 中声明，`uv sync` 自动安装：

```
numpy, pandas, scikit-learn, joblib, shap, xgboost, optuna, imbalanced-learn
```

---

## 快速开始

```python
from feature_extraction import extract_features, extract_features_per_user
from scoring_model import score_risk, score_risk_hybrid

# 1. 特征提取（传入 CSV 路径，返回 23 维特征向量）
features = extract_features("data/20260601_20260630_risk.csv")

# 2. 风险打分（0~100）
result = score_risk(features)
# → {"score": 36.4, "level": "MEDIUM", "level_range": "30~70"}

# 3. ML 混合模式（默认 Calibrated RF，开箱即用）
result = score_risk_hybrid(features, alpha=0.2)
# → {"score": 31.5, "level": "MEDIUM", "rule_score": 36.4, "ml_score": 30.3}
```

---

## API 详解

### 特征提取 `extract_features()`

```python
def extract_features(csv_path: str) -> dict[str, float]:
```

**输入：** CSV 文件路径（13 字段，由 @廖文远 的数据模块生成）

**输出：** 23 个特征键值对

#### 基础特征（7 个，v1.0）

| 特征键 | 类型 | 值域 | 含义 |
|--------|------|------|------|
| `device_reuse_ratio` | float | 0~1 | 设备复用率 |
| `ip_change_freq` | float | 0~1 | IP 变更频率 |
| `tx_freq` | float | ≥0 | 交易次数 |
| `login_fail_ratio` | float | 0~1 | 登录失败率 |
| `amount_anomaly_score` | float | 0~1 | 金额异常（MAD Z-score） |
| `behavior_time_anomaly` | float | 0~1 | 行为耗时异常（<5s 占比） |
| `multi_region_risk` | float | 0/1 | 多地区风险 |

#### 速度特征（4 个，v1.2）

| 特征键 | 类型 | 值域 | 含义 |
|--------|------|------|------|
| `tx_velocity_5min` | float | ≥0 | 5 分钟窗口最大交易数 |
| `tx_velocity_1h` | float | ≥0 | 1 小时窗口最大交易数 |
| `device_switch_24h` | float | ≥0 | 24h 窗口最大不同设备数 |
| `ip_switch_24h` | float | ≥0 | 24h 窗口最大不同 IP 数 |

#### 间隔与金额特征（4 个，v1.2）

| 特征键 | 类型 | 值域 | 含义 |
|--------|------|------|------|
| `tx_interval_mean_sec` | float | ≥0 | 平均交易间隔（秒） |
| `tx_burst_ratio` | float | 0~1 | 60s 内连续交易占比 |
| `amount_user_deviation` | float | ≥0 | 用户均值偏离全局中位数的 Z-score |
| `amount_user_max_ratio` | float | ≥0 | 最大单笔 / 用户均值 |

#### 时间循环特征（2 个，v1.2）

| 特征键 | 类型 | 值域 | 含义 |
|--------|------|------|------|
| `night_tx_ratio` | float | 0~1 | 凌晨 1am-5am 交易占比 |
| `odd_hour_tx_ratio` | float | 0~1 | 非工作时间交易占比 |

#### 交互特征（6 个，v1.3）⚠️ 最重要

| 特征键 | 类型 | 值域 | 含义 |
|--------|------|------|------|
| `device_ip_risk` | float | 0~1 | 设备+IP 同时切换的复合风险 |
| `night_automation` | float | 0~1 | 凌晨时段 + 快速操作的自动化脚本风险 |
| `velocity_amount` | float | ≥0 | 高频交易 × 金额异常的交互（RF 重要性 #1） |
| `burst_max_ratio` | float | 0~1 | 交易突发 × 大额比例 |
| `velocity_ratio` | float | ≥0 | 短窗口 / 长窗口交易比例 |
| `amount_exposure` | float | ≥0 | 总交易金额暴露度 |

**异常：** CSV 读取失败/字段缺失时抛出 `ValueError`

### Per-User 模式

```python
def extract_features_per_user(csv_path: str) -> dict[str, dict[str, float]]:
# → {"ACC_00001": {特征...}, "ACC_00002": {特征...}, ...}
```

每个账户独立提取 23 维特征。

### 风险打分 `score_risk()`

```python
def score_risk(
    features: dict[str, float],
    weights: dict[str, float] | None = None,
) -> dict[str, float | str]:
```

**返回结构：**
```json
{"score": 36.4, "level": "MEDIUM", "level_range": "30~70"}
```

**风险等级（v1.4 Optuna 优化）：**

| 等级 | 分数 | level 值 | 建议动作 |
|------|------|----------|---------|
| 低风险 | 0~29 | `"LOW"` | 放行 |
| 中风险 | 30~70 | `"MEDIUM"` | 人工复核 |
| 高风险 | 71~100 | `"HIGH"` | 拦截/告警 |

### ML 混合评分 `score_risk_hybrid()`

```python
def score_risk_hybrid(
    features: dict[str, float],
    alpha: float | str = 0.2,
) -> dict[str, float | str]:
# → {"score": 31.5, "level": "MEDIUM", "rule_score": 36.4, "ml_score": 30.3}
```

默认使用 **Calibrated RF**（Platt 校准的 RandomForest），模型文件预置在 `models/` 下，开箱即用。

| alpha 值 | 含义 |
|----------|------|
| `0.2`（默认） | 20% 规则 + 80% ML |
| `0.0` | 纯 ML |
| `1.0` | 纯规则 |
| `"auto"` | 动态调整：多规则命中→信规则，ML 高置信→信 ML |

---

## 集成示例

### 后端：批量打分

```python
# 一键打分所有账户
per_user = extract_features_per_user("input.csv")
results = []
for uid, features in per_user.items():
    r = score_risk(features)
    results.append({
        "account_id": uid,
        "score": r["score"],
        "level": r["level"],
    })

# 高风险告警
high_risk = [r for r in results if r["level"] == "HIGH"]
print(f"高风险账户: {len(high_risk)}/{len(results)}")
```

### 前端：展示风险信息

```javascript
// API 返回示例
{
  "account_id": "ACC_00016",
  "score": 84.0,
  "level": "HIGH",
  "top_features": [
    {"name": "金额异常", "value": 1.0},
    {"name": "行为耗时异常", "value": 1.0},
    {"name": "交易频率", "value": 91.0}
  ]
}
```

---

## 准确率（v1.4.0 统一评估）

测试集 15 账户，验证集 16 账户，按账户 69/16/15 分层划分。

### 模型原始预测（predict argmax）

| 模型 | Val Acc | Test Acc | Test Spearman | Cost | 大小 |
|------|:-------:|:--------:|:-------------:|:----:|:----:|
| **Calibrated RF（默认）** | 75.0% | **80.0%** | **0.825** | **4** | 593KB |
| Ensemble (16特征) | 68.8% | 80.0% | 0.854 | 7 | 760KB |
| Ensemble (23特征) | 68.8% | 80.0% | 0.854 | 7 | 1.5MB |

### 端到端评分（分数→等级映射，阈值 29/70）

| 评分方式 | Val Acc | Val Kappa | Test Acc | Test Kappa | Cost |
|----------|:-------:|:---------:|:--------:|:----------:|:----:|
| 纯规则 score_risk() | **87.5%** | 0.771 | 60.0% | 0.237 | 12 |
| 混合 α=0.2（默认） | 87.5% | 0.771 | **80.0%** | **0.605** | 8 |
| 混合 α=auto | 87.5% | 0.771 | 73.3% | 0.492 | 9 |

> 代价矩阵: 漏报 HIGH=10, MEDIUM=2, 误报 LOW=1, MEDIUM=3  
> Val 上规则和混合持平（87.5%）因为阈值在 Val 上优化；Test 上混合碾压规则（80% vs 60%）

---

## 速度

测试环境：Windows 10, Python 3.12

| 操作 | 耗时 | 说明 |
|------|------|------|
| 全量特征提取 (100 账户) | ~840 ms | 含 23 特征计算 |
| 单账户特征提取 | ~8 ms | 增量更新场景 |
| 规则打分 | **5 µs** | 纯 Python，几乎零延迟 |
| Calibrated RF 推理 | **26 ms** | 含标准化+预测 |
| 单账户端到端 (规则) | **8 ms** | 提取+打分 |
| 单账户端到端 (ML) | **34 ms** | 提取+ML打分 |

---

## 项目结构

```
feature_extraction.py   # 特征提取（23 维，组员 import 入口）
scoring_model.py         # 风险打分（规则引擎 + ML 混合，组员 import 入口）
train_ml_model.py        # ML 模型训练 + 增强评估
evaluate.py              # 评估 CLI（python evaluate.py full|splits|hybrid|calibrate）
run_eval.py              # Per-account 对比 规则 vs 混合
benchmark.py             # 性能基准测试
tests/                   # pytest (52 cases)
models/                  # 训练好的模型文件（开箱即用）
  ├── calibrated_rf.pkl        ← 默认模型 (593KB)
  ├── ensemble_selected.pkl    ← 16特征 Ensemble
  ├── ensemble.pkl             ← 23特征 Ensemble
  ├── scaler.pkl / scaler_selected.pkl
  └── rfecv_selector.pkl       ← 特征选择器
data/splits/             # 数据划分（69/16/15）
```

---

## 已知局限

| 局限 | 原因 | 缓解措施 |
|------|------|---------|
| 训练集仅 69 账户 | 模拟数据共 100 账户 | Platt 校准 + 特征选择 23→16 缓解过拟合 |
| 7 个特征被 RFECV 剔除 | 模拟数据特征分布简单 | 保留在代码中，真实数据会激活 |
| HIGH 类仅 7 样本 | 数据生成的角色故事特性 | Cost-sensitive 评估补偿 |
| 无实时打分能力 | 当前为批处理模式 | 单次 <35ms，API 封装即可 |
