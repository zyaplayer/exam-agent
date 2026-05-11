"""
考研伴学 Agent - API 路由接口层
=================================
定义所有对外 HTTP 接口。
MVP 范围：对话（SSE 流式）、文档上传入库、Collection 列表、健康检查。
"""

import os
import shutil
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.agent.router import classify_intent, Intent
from app.agent.qa_chain import ask_question_stream
from app.services.vector_store import ingest_document, list_collections


# ============================================
# 路由实例
# ============================================

router = APIRouter()

# [缺陷] 所有路由的前缀硬编码。如果后续要加版本号（如 /api/v1/chat），
#   需要逐个修改。应在 router 初始化时统一设置 prefix="/api/v1"。


# ============================================
# 请求/响应模型
# ============================================

class ChatRequest(BaseModel):
    """对话请求体"""
    message: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="学生的提问内容",
        examples=["什么是拉格朗日中值定理？"],
    )
    collection_name: str = Field(
        default="default",
        description="要检索的知识库 Collection 名称",
        examples=["default", "线性代数", "考研英语"],
    )
    # [缺陷] 缺少 conversation_id 字段，无法关联多轮对话历史。
    # [后续扩展] 新增 conversation_id: Optional[str] 字段。


class ChatErrorResponse(BaseModel):
    """对话错误响应"""
    error: str = Field(..., description="错误描述")
    detail: Optional[str] = Field(default=None, description="详细错误信息")


class IngestResponse(BaseModel):
    """文档入库成功响应"""
    filename: str = Field(..., description="原始文件名")
    collection_name: str = Field(..., description="目标 Collection")
    chunk_count: int = Field(..., description="切分并入库的文档块数量")
    message: str = Field(default="文档入库成功", description="状态描述")


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(default="ok", description="服务状态")
    project: str = Field(default=settings.PROJECT_NAME, description="项目名称")
    version: str = Field(default=settings.PROJECT_VERSION, description="版本号")
    # [缺陷] 健康检查未包含模型可用性、ChromaDB 连接状态等深度检查。
    # [后续扩展] 增加 deep_check 参数，返回各依赖组件的连通性。


class CollectionListResponse(BaseModel):
    """Collection 列表响应"""
    collections: list[str] = Field(..., description="Collection 名称列表")
    count: int = Field(..., description="Collection 总数")


# ============================================
# 占位消息（非 QA 意图的兜底回复）
# ============================================

_PLACEHOLDER_MESSAGES: dict[Intent, str] = {
    Intent.QUIZ: (
        "做题练习功能正在全力开发中，预计下个版本上线！"
        "当前你可以：\n"
        "1. 先上传历年真题 PDF 到知识库\n"
        "2. 通过问答模式让我逐题讲解\n"
        "3. 把不会的题目直接发给我，我来分析"
    ),
    Intent.PLANNER: (
        "学习规划功能正在开发中，敬请期待！"
        "当前你可以：\n"
        "1. 向我提问任何考研知识点\n"
        "2. 上传教材和笔记，我会基于你的资料作答\n"
        "3. 让我帮你分析某个概念或定理"
    ),
}

# [缺陷] 占位消息写死在代码中，无法热更新。
# [后续扩展] 可移到 prompts.py 或配置文件中统一管理。


# ============================================
# 辅助函数
# ============================================

async def _stream_placeholder(intent: Intent):
    """
    将占位消息模拟为流式输出。
    保持与正常问答一致的 SSE 格式，前端无需区分。
    """
    message = _PLACEHOLDER_MESSAGES.get(intent, "该功能暂未开放，请尝试其他操作。")
    # 按字符逐字输出，模拟打字效果
    # [缺陷] 中文逐字输出与正常 LLM token 流式粒度不一致（token != 字符），
    #   前端打字机动画可能节奏不同。
    for char in message:
        yield char
    # [缺陷] 简单的 sleep 模拟打字延迟。asyncio.sleep 需要 import。


async def _sse_wrapper(generator):
    """
    将异步生成器包装为标准 SSE（Server-Sent Events）格式。

    每输出格式：
        data: {token}\n\n

    以 [DONE] 标记流结束。
    """
    async for token in generator:
        # SSE 格式：data: <内容>\n\n
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


# ============================================
# 一、对话接口（核心）
# ============================================

@router.post(
    "/api/chat",
    summary="考研伴学对话接口（SSE 流式）",
    description="接收学生问题，自动识别意图并返回流式回答。",
    response_description="SSE 文本流，每个 token 作为 data 事件推送",
)
async def chat(req: ChatRequest):
    """
    核心对话接口。

    流程:
        1. 接收用户消息
        2. 意图识别（router.classify_intent）
        3. 根据意图分发:
           - QA      → qa_chain.ask_question_stream（RAG + LLM 流式生成）
           - QUIZ    → 占位消息（做题功能开发中）
           - PLANNER → 占位消息（规划功能开发中）
        4. 以 SSE 格式流式返回
    """
    # 步骤1: 意图识别
    route_result = classify_intent(req.message)

    # 步骤2: 根据意图选择处理链路
    try:
        if route_result.intent == Intent.QA:
            # 核心链路：RAG 流式问答
            token_generator = ask_question_stream(
                question=req.message,
                collection_name=req.collection_name,
            )
        else:
            # 占位链路：功能开发中
            token_generator = _stream_placeholder(route_result.intent)
    except ValueError as e:
        # API Key 未配置等预期内错误
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        # [缺陷] 宽泛的 Exception 捕获可能掩盖真实问题。
        # [后续扩展] 细化异常类型，区分"可重试"和"不可重试"错误。
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时发生内部错误: {str(e)}",
        )

    # 步骤3: 以 SSE 格式流式返回
    return StreamingResponse(
        _sse_wrapper(token_generator),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # 禁用 Nginx 缓冲（部署时有用）
            # [缺陷] 未设置 CORS 头，虽然 main.py 中有全局 CORS 中间件，
            #   但如果单独部署此路由到其他网关，可能会跨域失败。
        },
    )


# ============================================
# 二、文档上传接口
# ============================================

@router.post(
    "/api/documents/upload",
    summary="上传文档并入库",
    description="接收文件上传，保存后自动解析、切分、向量化并存入 ChromaDB。",
    response_model=IngestResponse,
)
async def upload_document(
    file: UploadFile = File(
        ...,
        title="文档文件",
        description="支持 PDF / Markdown / TXT 格式",
    ),
    collection_name: str = "default",
):
    """
    文档上传与自动入库接口。

    流程:
        1. 验证文件格式
        2. 保存到 data/raw_docs/
        3. 调用 ingest_document（解析 → 切分 → 向量化 → 入库）
        4. 返回入库结果
    """
    # 步骤1: 验证文件后缀
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    allowed = {".pdf", ".markdown", ".txt"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 '{suffix}'。支持的格式: {', '.join(allowed)}",
        )

    # 步骤2: 确保上传目录存在
    raw_docs_dir = settings.BASE_DIR / settings.RAW_DOCS_DIR.lstrip("./")
    raw_docs_dir.mkdir(parents=True, exist_ok=True)

    # 步骤3: 保存文件到本地
    save_path = raw_docs_dir / filename
    try:
        with open(save_path, "wb") as f:
            # 分块读取，避免大文件撑爆内存
            while chunk := await file.read(1024 * 1024):  # 1MB 每块
                f.write(chunk)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文件保存失败: {str(e)}",
        )
    # [缺陷] 没有文件大小限制，恶意上传大文件可能撑爆磁盘。
    # [后续扩展] 在读取循环中加入 size 累计，超出上限（如 100MB）返回 413 错误。
    # [缺陷] 同名文件直接覆盖，没有做版本管理或冲突检测。

    # 步骤4: 入库
    try:
        chunk_count = ingest_document(
            file_path=str(save_path),
            collection_name=collection_name,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"文档入库失败: {str(e)}",
        )

    return IngestResponse(
        filename=filename,
        collection_name=collection_name,
        chunk_count=chunk_count,
    )


# ============================================
# 三、Collection 列表接口
# ============================================

@router.get(
    "/api/documents",
    summary="列出所有知识库 Collection",
    response_model=CollectionListResponse,
)
async def get_collections():
    """
    返回 ChromaDB 中所有的 Collection 名称与总数。
    用于前端下拉框展示或管理界面。
    """
    try:
        cols = list_collections()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"获取 Collection 列表失败: {str(e)}",
        )

    return CollectionListResponse(
        collections=cols,
        count=len(cols),
    )


# ============================================
# 四、健康检查接口
# ============================================

@router.get(
    "/api/health",
    summary="服务健康检查",
    response_model=HealthResponse,
)
async def health_check():
    """
    简单的健康检查，返回项目名称和版本。
    可用于 Docker 容器健康探测或负载均衡健康检查。
    """
    return HealthResponse()


# ============================================
# 已知缺陷汇总
# ============================================
#
# 1. 【无认证/鉴权】
#    - 所有接口裸奔，任何人可调。
#    - 后续至少加入 API Key 鉴权（如 Bearer Token 或简单的 X-API-Key 头）。
#
# 2. 【无请求频率限制】
#    - 无限调用可能消耗大量 API 额度。
#    - 后续加入 slowapi 或 Redis 令牌桶做速率限制。
#
# 3. 【上传文件无去重/版本管理】
#    - 同名文件直接覆盖，重复上传产生冗余向量。
#
# 4. 【无文件大小限制】
#    - 大文件可能撑爆磁盘或导致 ChromaDB 性能问题。
#
# 5. 【无异步检索】
#    - ask_question_stream 中检索是同步的，阻塞事件循环。
#
# 6. 【无请求日志/监控】
#    - 缺少请求耗时、成功率等基础指标。
#
# 7. 【缺少 planner 和 quiz 链路的真实实现】
#    - 当前仅返回占位文本，核心功能缺失。
