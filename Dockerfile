FROM python:3.12-slim AS base

WORKDIR /app

# 使用国内镜像源加速apt下载
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 使用国内PyPI镜像
ENV UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
ENV UV_EXTRA_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple/

# 先复制依赖声明，利用 Docker 缓存
COPY pyproject.toml uv.lock ./

# 只安装生产依赖（server需要的包）
RUN uv sync --no-dev --frozen

# 复制项目代码和模型
COPY . .

# 确保 data 目录存在
RUN mkdir -p data

EXPOSE 8000

# 默认启动 Web 仪表盘
CMD ["uv", "run", "--no-sync", "uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
