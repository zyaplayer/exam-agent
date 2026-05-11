# ============================================
# 考研伴学 Agent - Docker 镜像
# ============================================
# 构建命令:
#   docker build -t exam-agent:latest .
#
# 基于 Python 3.12 slim 镜像（比完整版小约 200MB）
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# ---- 系统依赖 ----
# 安装编译工具链（sentence-transformers 等包构建时需要 gcc/g++）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# ---- Python 依赖（分层缓存优化） ----
# 先复制 requirements.txt 单独一层，这样改代码不需要重新 pip install
COPY requirements.txt .

# 安装 Python 依赖
# [注] sentence-transformers 首次运行时会自动下载模型文件
RUN pip install --no-cache-dir -r requirements.txt

# ---- 应用代码 ----
# 只复制运行必需的代码目录
COPY app/ ./app/
COPY main.py .
COPY static/ ./static/

# ---- HuggingFace 模型缓存目录 ----
# 设为 Volume 挂载点，避免每次重建容器都重新下载模型（~100MB）
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME

# ---- 暴露端口 ----
EXPOSE 8000

# ---- 启动命令 ----
# 不使用 --reload（生产模式），由 docker-compose 管理重启策略
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
# [缺陷] 单进程 uvicorn，并发能力有限。
# [后续扩展] 生产环境改为:
#   CMD ["gunicorn", "main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]
