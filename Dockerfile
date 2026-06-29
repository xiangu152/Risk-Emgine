# ========== Stage 1: Builder ==========
FROM python:3.12-slim AS builder

WORKDIR /app

# 使用国内镜像源加速apt下载
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 安装构建依赖（gcc/g++ 只在编译阶段需要）
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

# ========== Stage 2: Runtime ==========
FROM python:3.12-slim AS runtime

WORKDIR /app

# 使用国内镜像源加速apt下载
RUN sed -i 's/deb.debian.org/mirrors.aliyun.com/g' /etc/apt/sources.list.d/debian.sources

# 运行时仅需 minimal 依赖（libstdc++ 供 C 扩展使用）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 && \
    rm -rf /var/lib/apt/lists/*

# 从 builder 阶段复制已安装的 Python 包
COPY --from=builder /app/.venv /app/.venv

# 复制项目代码和模型
COPY . .

# 确保 data 目录存在
RUN mkdir -p data

EXPOSE 8000

# 默认启动 Web 仪表盘
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "dashboard.server:app", "--host", "0.0.0.0", "--port", "8000"]
