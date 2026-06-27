FROM python:3.12-slim AS base

WORKDIR /app

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 先复制依赖声明，利用 Docker 缓存
COPY pyproject.toml ./
RUN uv sync --no-dev --frozen 2>/dev/null || uv sync --no-dev

# 复制项目代码和模型
COPY . .

# 确保 data 目录存在
RUN mkdir -p data

EXPOSE 8000

# 默认启动 Web 仪表盘
CMD ["uv", "run", "uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
