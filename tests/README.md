# 测试套件

本目录包含风控引擎的自动化测试用例，使用 pytest 框架。

## 测试文件

| 文件 | 测试数 | 覆盖范围 |
|------|--------|----------|
| `test_feature_extraction.py` | 13 | 特征提取模块的所有特征计算逻辑 |
| `test_scoring_model.py` | 25 | 规则引擎的 16 条规则和评分逻辑 |
| `test_api_integration.py` | 9 | 端到端 API 集成测试 |

## 运行测试

```bash
# 运行所有测试
uv run pytest tests/ -v

# 运行特定测试文件
uv run pytest tests/test_feature_extraction.py -v

# 运行特定测试用例
uv run pytest tests/test_scoring_model.py::test_rule1_device_tx -v

# 显示测试覆盖率（需安装 pytest-cov）
uv run pytest tests/ --cov=. --cov-report=html
```

## 测试数据

`data/` 目录包含测试用的 CSV 文件：

| 文件 | 说明 |
|------|------|
| `test_normal.csv` | 正常场景，4 个用户 |
| `test_empty.csv` | 空 CSV（仅表头） |
| `test_no_login_fail.csv` | 无登录失败事件 |
| `test_no_transactions.csv` | 无交易事件 |
| `test_missing_amount.csv` | 缺失金额字段 |
| `test_extreme.csv` | 极端场景（高设备复用、高 IP 变更） |

## 测试覆盖范围

### feature_extraction 测试

- ✅ 正常数据特征提取
- ✅ 设备复用率计算（device_reuse_ratio）
- ✅ IP 变更频率计算（ip_change_freq）
- ✅ 交易频率计算（tx_freq）
- ✅ 登录失败率计算（login_fail_ratio）
- ✅ 金额异常分数计算（amount_anomaly_score）
- ✅ 空数据异常处理
- ✅ 缺失字段处理
- ✅ 极端场景处理
- ✅ Per-user 模式测试

### scoring_model 测试

- ✅ 全零特征评分
- ✅ 全一特征评分
- ✅ 中间值评分
- ✅ 16 条规则单独测试
- ✅ 规则组合测试
- ✅ 边界值测试
- ✅ 分数截断测试（0~100）
- ✅ 风险等级映射测试

### API 集成测试

- ✅ 特征提取快速入门
- ✅ 特征值范围验证
- ✅ Per-user 模式测试
- ✅ 风险评分等级映射
- ✅ 批量评分测试
- ✅ 混合评分测试
- ✅ JSON 输出格式测试
- ✅ 错误处理测试

## 添加新测试

1. 在 `data/` 目录添加测试数据文件（如需要）
2. 在对应的 `test_*.py` 文件中添加测试函数
3. 使用 pytest 的 `assert` 进行断言
4. 运行 `uv run pytest tests/ -v` 验证

## 注意事项

- 测试使用 `tests/data/` 目录下的固定数据集，确保测试可重复
- 测试不依赖外部服务或网络连接
- 所有测试应在 10 秒内完成
