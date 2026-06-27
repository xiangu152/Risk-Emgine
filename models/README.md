# 预训练模型

本目录包含风控引擎的预训练机器学习模型，可直接使用，无需重新训练。

## 模型列表

| 模型文件 | 大小 | 说明 | 使用场景 |
|----------|------|------|----------|
| `calibrated_rf.pkl` | 593KB | Platt 校准的 Random Forest（23 特征） | **默认模型**，推荐使用 |
| `ensemble.pkl` | 1.5MB | 集成模型（23 特征） | 备选模型 |
| `ensemble_selected.pkl` | 760KB | 集成模型（16 特征，RFECV 选择） | 特征选择后的模型 |
| `random_forest.pkl` | 125KB | 基础 Random Forest | 基线模型 |
| `smote_xgboost.pkl` | 289KB | SMOTE + XGBoost | 处理不平衡数据 |

## 辅助文件

| 文件 | 说明 |
|------|------|
| `scaler.pkl` | StandardScaler，用于 23 特征标准化 |
| `scaler_selected.pkl` | StandardScaler，用于 16 特征标准化 |
| `rfecv_selector.pkl` | RFECV 特征选择器（23→16） |
| `ensemble_selected_features.json` | 选中的 16 个特征名称列表 |

## 使用方法

### Python API

```python
import joblib
from scoring_model import score_risk_hybrid

# 加载模型（score_risk_hybrid 内部会自动加载）
result = score_risk_hybrid(features, alpha=0.2)
# → {"score": 31.5, "level": "MEDIUM", "rule_score": 36.4, "ml_score": 30.3}
```

### 直接加载模型

```python
import joblib

# 加载校准后的 Random Forest
model = joblib.load("models/calibrated_rf.pkl")
scaler = joblib.load("models/scaler.pkl")

# 预测
X_scaled = scaler.transform(X)
proba = model.predict_proba(X_scaled)
```

## 模型性能

| 模型 | Val Acc | Test Acc | Test Spearman | Cost |
|------|:-------:|:--------:|:-------------:|:----:|
| **Calibrated RF** | 75.0% | **80.0%** | **0.825** | **4** |
| Ensemble (16 特征) | 68.8% | 80.0% | 0.854 | 7 |
| Ensemble (23 特征) | 68.8% | 80.0% | 0.854 | 7 |

> 代价矩阵: 漏报 HIGH=10, MEDIUM=2; 误报 LOW=1, MEDIUM=3

## 重新训练模型

如需重新训练模型，请运行：

```bash
# 确保 data/splits/ 目录下有 train.csv, val.csv, test.csv
uv run python train_ml_model.py
```

训练完成后，新模型将保存到本目录。

## 特征说明

### 23 维完整特征

基础特征（7 个）+ 速度特征（4 个）+ 间隔金额特征（4 个）+ 时间循环特征（2 个）+ 交互特征（6 个）

详见项目根目录 `README.md` 的「特征体系」章节。

### 16 维选择特征

通过 RFECV 特征选择，从 23 维中选出最重要的 16 维特征，具体列表见 `ensemble_selected_features.json`。
