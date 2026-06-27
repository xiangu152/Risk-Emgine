# Scripts 开发工具脚本

本目录包含风控引擎的开发、分析和调试工具脚本。这些脚本不是核心生产代码，而是用于模型开发、性能评估和特征分析。

## 脚本列表

| 脚本 | 用途 | 说明 |
|------|------|------|
| `benchmark.py` | 性能基准测试 | 测量特征提取和打分的延迟与吞吐量 |
| `calibrate_thresholds.py` | 阈值校准 | 基于验证集 PR 曲线优化 LOW/MEDIUM/HIGH 边界 |
| `optimize_thresholds.py` | 阈值优化 | 使用代价矩阵搜索最优分界点 |
| `shap_analysis.py` | SHAP 分析 | 分析特征贡献度，生成可视化报告 |
| `run_eval.py` | 评估脚本 | 在训练/验证/测试集上评估模型表现 |
| `generate_behavior_data.py` | 测试数据生成 | 生成用于开发测试的行为日志数据 |

## 使用方法

所有脚本都应从项目根目录运行：

```bash
# 性能基准测试
uv run python scripts/benchmark.py

# 阈值校准
uv run python scripts/calibrate_thresholds.py

# 阈值优化
uv run python scripts/optimize_thresholds.py

# SHAP 特征分析
uv run python scripts/shap_analysis.py

# 模型评估
uv run python scripts/run_eval.py

# 生成测试数据
uv run python scripts/generate_behavior_data.py
```

## 输出说明

### benchmark.py
输出到控制台，显示：
- 特征提取延迟（全量/单账户）
- 规则打分延迟（µs 级）
- ML 推理延迟（ms 级）
- 全量吞吐量

### calibrate_thresholds.py / optimize_thresholds.py
输出到控制台，显示：
- 不同阈值组合的准确率、Kappa 系数
- 成本敏感分析结果
- 建议的最优阈值

### shap_analysis.py
输出到：
- 控制台：特征重要性排序、方向性分析
- `data/model/` 目录：SVG 可视化图表
  - `shap_importance_bar.svg` - 特征重要性柱状图
  - `shap_beeswarm.svg` - SHAP 蜂群图
  - `shap_heatmap.svg` - SHAP 热力图
  - `shap_dependence.svg` - 特征依赖图

### run_eval.py
输出到控制台，显示：
- 各数据集（train/val/test）的准确率、Kappa、Spearman 相关系数
- 每个账户的预测详情
- 混合模型修复/误伤的案例

### generate_behavior_data.py
输出到 `data/behavior_logs.csv`，包含约 170 行测试数据。

## 注意事项

1. **路径引用**：所有脚本已配置为从 `scripts/` 目录正确引用项目根目录的资源
2. **依赖**：确保已安装所有依赖（`uv sync`）
3. **数据**：部分脚本需要 `data/splits/` 目录下的训练/验证/测试集数据
4. **模型**：SHAP 分析和评估脚本需要 `models/` 目录下的预训练模型
