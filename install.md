# 安装指南

> 供智能体和开发者参考的环境搭建步骤

---

## 🐳 Docker 一键部署（推荐）

最快上手方式，无需手动安装 Python 和依赖。

### 前置条件

| 依赖 | 版本 | 说明 |
|------|------|------|
| Docker | ≥ 20.10 | [安装 Docker](https://docs.docker.com/get-docker/) |
| Docker Compose | ≥ 2.0 | Docker Desktop 自带 |

### 一键启动

```bash
git clone <repo-url>
cd Risk-Emgine
docker compose up -d
```

浏览器打开 **http://localhost:8000** 即可使用 Web 仪表盘。

### 常用命令

```bash
# 后台启动
docker compose up -d

# 查看日志
docker compose logs -f

# 停止服务
docker compose down

# 重新构建（代码更新后）
docker compose up -d --build
```

### 挂载数据

`data/` 目录已挂载到容器内，放入 CSV 文件即可在 Web 界面上传：

```bash
# 将数据放入 data/ 目录
cp your_data.csv data/
# 然后在 Web 界面上传即可
```

### Docker 运行 CLI

```bash
# 进入容器执行命令
docker compose exec risk-engine uv run python main.py --per-user data/your_data.csv
```

---

## 手动安装

如需本地开发或无法使用 Docker，按以下步骤操作。

### 前置条件

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | ≥ 3.12 | 项目要求，推荐 3.12.x |
| uv | 任意 | Python 包管理器，替代 pip |

### 安装 uv（如未安装）

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Homebrew
brew install uv
```

---

## 安装步骤

### 1. 克隆仓库

```bash
git clone <repo-url>
cd Risk-Emgine
```

### 2. 安装依赖

```bash
uv sync
```

这会自动：
- 创建 `.venv` 虚拟环境（使用系统 Python 3.12+）
- 安装 `pyproject.toml` 中声明的所有运行时和开发依赖（约 170 个包）
- 生成 `uv.lock` 锁文件

### 3. 准备数据文件

集成测试和 CLI 工具需要数据文件位于 `data/` 目录下。项目自带一份生成好的数据集：

```bash
# 从 data_creater 子模块复制数据到项目根目录
cp module/data_creater/data/data_final/20260601_20260630_risk.csv data/
```

### 4. 验证安装

```bash
# 运行全部测试（52 个用例）
uv run pytest tests/ -v

# 验证核心模块可导入
uv run python -c "from feature_extraction import extract_features; from scoring_model import score_risk; print('OK')"

# 验证 dashboard 可加载
uv run python -c "from dashboard.server import app; print(app.title)"
```

预期结果：52 tests passed，无报错。

---

## 运行方式

### CLI 风险评分

```bash
# 批量模式
uv run python main.py data/20260601_20260630_risk.csv

# Per-user 模式
uv run python main.py --per-user data/20260601_20260630_risk.csv

# 单用户
uv run python main.py --per-user -u ACC_00001 data/20260601_20260630_risk.csv
```

### Web 仪表盘

```bash
cd dashboard
uv run uvicorn server:app --reload --port 8000
# 浏览器打开 http://localhost:8000
```

### 评估

```bash
uv run python evaluate.py full          # 全量评估
uv run python evaluate.py splits        # 分集评估
uv run python evaluate.py hybrid        # α 参数扫描
uv run python evaluate.py calibrate     # 阈值网格搜索
uv run python evaluate.py all           # 全部评估
```

### ML 训练

```bash
# 需要 data/splits/{train,val,test}.csv
uv run python train_ml_model.py
```

---

## 子模块：合成数据生成

`module/data_creater/` 是独立的子模块，用于生成带标签的行为日志数据。

### 额外依赖

该子模块的依赖（`httpx`, `python-dotenv`）已包含在主项目的 `pyproject.toml` 中，无需单独安装。

### 配置

在 `module/data_creater/` 下创建 `.env` 文件：

```env
ANTHROPIC_BASE_URL=https://your-llm-api-endpoint
ANTHROPIC_AUTH_TOKEN=your-api-key
MODEL=model-name
```

### 运行

```bash
python module/data_creater/main.py 20260601 20260630 ./data train 100
```

参数说明见 `module/data_creater/README.md`。

---

## 项目结构速查

```
Risk-Emgine/
├── main.py                    # CLI 入口
├── feature_extraction.py      # 23 维特征提取
├── scoring_model.py           # 规则引擎 + ML 混合评分
├── train_ml_model.py          # ML 模型训练
├── evaluate.py                # 评估 CLI
├── models/                    # 预训练模型（开箱即用）
├── dashboard/                 # FastAPI + ECharts 仪表盘
│   ├── server.py
│   └── static/index.html
├── module/data_creater/       # 合成数据生成子模块
├── tests/                     # pytest 测试套件
├── data/                      # 运行时数据（gitignored）
├── pyproject.toml             # 依赖声明
└── API_README.md              # API 详细文档
```

---

## 常见问题

### `python` 命令找不到

macOS 上 `python` 可能不在 PATH 中。所有命令统一用 `uv run python` 执行。

### 集成测试报 FileNotFoundError

`test_api_integration.py` 需要 `data/20260601_20260630_risk.csv`。确保已执行步骤 3（复制数据文件）。

### data_creater 的 `.env` 缺失

合成数据生成需要 LLM API 凭据。如不需要生成数据，可忽略此步骤，核心评分功能不依赖它。

---

## 依赖清单（主要）

| 包 | 用途 |
|----|------|
| numpy, pandas, scipy | 数值计算 |
| scikit-learn | ML 模型、预处理 |
| xgboost | 梯度提升 |
| imbalanced-learn | SMOTE 过采样 |
| joblib | 模型序列化 |
| shap | 特征重要性分析 |
| optuna | 超参优化 |
| tabpfn | 表格数据基础模型（可选） |
| fastapi, uvicorn | Web 仪表盘 |
| httpx | LLM API 客户端（data_creater） |
| pytest | 测试框架 |
