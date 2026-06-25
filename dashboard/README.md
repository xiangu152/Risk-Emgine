# 风控引擎可视化面板

Web 可视化界面，用于查看风控引擎的风险评分结果。

## 启动

```bash
# 安装依赖（首次）
uv sync

# 启动服务
cd dashboard
uv run uvicorn server:app --reload --port 8000
```

浏览器打开 http://localhost:8000

## 功能

| 页面 | 说明 |
|------|------|
| 风险概览 | 统计卡片、风险分布饼图、评分散点图、Top10 高风险排行 |
| 账户列表 | 全部账户评分表，支持搜索和风险等级筛选，点击查看详情 |
| 用户详情 | 23维特征雷达图、评分对比、16条规则命中状态、行为时间线 |
| 特征分析 | 特征权重柱状图、各等级雷达对比、特征分布直方图 |
| 数据上传 | 拖拽或点击上传 CSV 行为日志文件 |

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `POST /api/upload` | POST | 上传 CSV 并处理 |
| `GET /api/overview` | GET | 全局统计数据 |
| `GET /api/users` | GET | 所有用户评分列表 |
| `GET /api/users/{id}` | GET | 单用户详情 |
| `GET /api/features/weights` | GET | 23维特征权重 |
| `GET /api/features/distribution` | GET | 特征分布统计 |
