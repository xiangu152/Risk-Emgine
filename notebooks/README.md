# Jupyter 分析笔记本

本目录包含风控引擎的 Jupyter Notebook，用于数据分析、模型探索和结果可视化。

## 笔记本列表

| 文件 | 说明 |
|------|------|
| `risk_analysis.ipynb` | 完整的风险分析流程（特征工程、模型训练、评估） |
| `risk_analysis_naive.ipynb` | 简化版分析（基础特征、快速验证） |
| `risk_analysis_executed.ipynb` | 已执行的完整分析（包含输出结果） |
| `risk_analysis_naive_executed.ipynb` | 已执行的简化分析（包含输出结果） |

## 使用方法

### 启动 Jupyter

```bash
# 启动 Jupyter Lab
uv run jupyter lab

# 或启动经典 Jupyter Notebook
uv run jupyter notebook
```

### 查看已执行的笔记本

`*_executed.ipynb` 文件包含完整的执行结果，可直接打开查看，无需重新运行。

## 笔记本内容

### risk_analysis.ipynb

1. **数据加载与探索**
   - 加载行为日志 CSV
   - 数据分布可视化
   - 缺失值分析

2. **特征工程**
   - 23 维特征提取
   - 特征相关性分析
   - 特征分布可视化

3. **模型训练**
   - Random Forest 训练
   - XGBoost 训练
   - 集成模型训练

4. **模型评估**
   - 准确率、Kappa、Spearman 相关系数
   - 混淆矩阵
   - ROC 曲线

5. **特征重要性**
   - SHAP 值分析
   - 特征贡献度可视化

### risk_analysis_naive.ipynb

简化版分析流程，适合快速验证和调试。

## 注意事项

- 部分笔记本需要 `data/splits/` 目录下的训练/验证/测试集数据
- 已执行的笔记本可能包含较大的输出，建议定期清理
- 建议使用 Jupyter Lab 获得更好的开发体验
