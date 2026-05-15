"""
考研伴学 Agent - FastAPI 主入口
=================================
启动 API 服务，挂载路由、CORS 中间件、静态文件服务。
"""
import os
os.environ["HF_ENDPOINT"] = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")


from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.api.routes import router as api_router


# ============================================
# 应用生命周期管理（替代已弃用的 @app.on_event）
# ============================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI 应用生命周期管理。
    - 启动时: 创建数据目录、打印启动信息。
    - 关闭时: 打印关闭信息。
    """
    # ---- 启动逻辑 ----
    dirs_to_create = [
        settings.BASE_DIR / settings.CHROMA_PERSIST_DIR.lstrip("./"),
        settings.BASE_DIR / settings.RAW_DOCS_DIR.lstrip("./"),
        settings.BASE_DIR / settings.SUMMARY_DB_DIR.lstrip("./"),
    ]
    for d in dirs_to_create:
        d.mkdir(parents=True, exist_ok=True)

    # 启动预检：API Key 是否已配置
    if not settings.DEEPSEEK_API_KEY or settings.DEEPSEEK_API_KEY.startswith("sk-your-"):
        print("[警告] DEEPSEEK_API_KEY 未配置或仍为占位值，LLM 调用将失败")
    else:
        print(f"[启动] DeepSeek API Key 已配置 ({settings.DEEPSEEK_API_KEY[:8]}...)")

    print(f"[启动] {settings.PROJECT_NAME} v{settings.PROJECT_VERSION}")
    print(f"[启动] 文档上传目录: {settings.RAW_DOCS_DIR}")
    print(f"[启动] 向量库目录:   {settings.CHROMA_PERSIST_DIR}")

    yield  # ← 应用运行期间停在此处

    # ---- 关闭逻辑 ----
    print(f"[关闭] {settings.PROJECT_NAME} 已停止")
    # 清理 ChromaDB 客户端
    try:
        from app.services.vector_store import _chroma_clients
        for path_key, client in list(_chroma_clients.items()):
            try:
                del _chroma_clients[path_key]
            except Exception:
                pass
    except Exception:
        pass


# ============================================
# FastAPI 应用实例化
# ============================================

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.PROJECT_VERSION,
    description="以考研为导向的智能体后端，支持 RAG 知识问答、做题练习与学习规划。",
    lifespan=lifespan,
    # [缺陷] docs_url 和 redoc_url 使用默认值，生产环境建议关闭或加鉴权。
    # [后续扩展] 生产环境设置 docs_url=None 关闭 Swagger UI。
)

# ============================================
# CORS 跨域中间件
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发阶段允许所有来源
    # [后续扩展] 生产环境改为白名单: allow_origins=["https://yourdomain.com"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 挂载 API 路由
# ============================================

app.include_router(api_router)

# ============================================
# 挂载静态文件（前端测试页）
# ============================================

static_dir = settings.BASE_DIR / "static"
static_dir.mkdir(parents=True, exist_ok=True)

# 将 static/ 目录挂载到 /static 子路径（避免与 API 路由冲突）
app.mount("/static", StaticFiles(directory=str(static_dir), html=True), name="static")


# 根路径 → 返回前端测试页
@app.get("/")
async def root():
    """根路径 → 返回前端测试页"""
    from fastapi.responses import FileResponse
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "考研伴学 Agent API 已启动", "docs": "/docs"}


# ============================================
# 主程序入口
# ============================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        # reload=True 表示代码变更时自动重启（仅开发环境）
        # [后续扩展] 生产环境用 gunicorn + uvicorn workers，而非单进程 uvicorn。
    )
