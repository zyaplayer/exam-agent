"""
考研伴学 Agent - API 路由接口层
=================================
定义所有对外 HTTP 接口。
/chat 接口接入三条真实链路: QA / Quiz / Planner。
"""

import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.core.config import settings
from app.agent.router import classify_intent, Intent
from app.agent.qa_chain import ask_question_stream
from app.agent.quiz_generator import (
    generate_simple_quiz_stream,
    generate_exam_quiz_stream,
    get_all_bindings,
    create_binding,
    delete_binding,
)
from app.agent.planner_chain import generate_plan_stream
from app.services.vector_store import ingest_document, list_collections
from app.services.conversation_service import (
    new_conversation_id,
    append_message,
    get_history,
)


# ============================================
# 路由实例
# ============================================

router = APIRouter()


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
    )
    collection_name: str = Field(
        default="default",
        description="要检索的知识库 Collection 名称",
    )
    conversation_id: str = Field(
        default="",
        description="对话ID（不传则自动创建新对话）",
    )


class ChatErrorResponse(BaseModel):
    """对话错误响应"""
    error: str = Field(..., description="错误描述")
    detail: Optional[str] = Field(default=None, description="详细错误信息")


class IngestResponse(BaseModel):
    """文档入库成功响应"""
    filename: str
    collection_name: str
    chunk_count: int
    message: str = "文档入库成功"


class CollectionListResponse(BaseModel):
    """Collection 列表响应"""
    collections: list[str]
    count: int


class BindingRequest(BaseModel):
    """创建绑定请求"""
    textbook_collections: list[str] = Field(..., min_length=1)
    exam_collections: list[str] = Field(..., min_length=1)
    label: str = ""


class BindingResponse(BaseModel):
    """绑定记录"""
    id: str
    textbook_collections: list[str]
    exam_collections: list[str]
    label: str


# ============================================
# 辅助：SSE 包装器
# ============================================

async def _sse_wrapper(generator):
    """
    将异步生成器包装为标准 SSE（Server-Sent Events）格式。

    每输出格式:
        data: {token}\n\n

    以 [DONE] 标记流结束。
    """
    async for token in generator:
        yield f"data: {token}\n\n"
    yield "data: [DONE]\n\n"


# ============================================
# 一、对话接口（核心 — 三条链路）
# ============================================

@router.post(
    "/api/chat",
    summary="考研伴学对话接口（SSE 流式）",
    description="接收学生问题，自动识别意图并分发给 QA / Quiz / Planner 链路。",
)
async def chat(req: ChatRequest):
    """
    核心对话接口。

    意图分发:
      - QA       → qa_chain.ask_question_stream（RAG 知识问答）
      - QUIZ     → quiz_generator.generate_simple_quiz_stream（做题练习）
      - PLANNER  → planner_chain.generate_plan_stream（学习规划）
    """
    # 步骤1: 意图识别（混合方案：规则 + LLM）
    route_result = classify_intent(req.message)

    # 步骤1.5: 管理对话ID
    conv_id = req.conversation_id or new_conversation_id()

    # 步骤2: 根据意图选择处理链路
    try:
        if route_result.intent == Intent.QA:
            token_generator = ask_question_stream(
                question=req.message,
                collection_name=req.collection_name,
                conversation_id=conv_id,
            )
        elif route_result.intent == Intent.QUIZ:
            token_generator = generate_simple_quiz_stream(
                message=req.message,
                collection_name=req.collection_name,
                conversation_id=conv_id,
            )
        elif route_result.intent == Intent.PLANNER:
            token_generator = generate_plan_stream(
                message=req.message,
                conversation_id=conv_id,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"未知的意图类型: {route_result.intent}",
            )
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理请求时发生内部错误: {str(e)}",
        )

    # 步骤3: 包装生成器（收集完整回复 → 保存历史）
    async def history_wrapper():
        full_answer = []
        async for token in token_generator:
            full_answer.append(token)
            yield token
        append_message(conv_id, "user", req.message)
        append_message(conv_id, "assistant", "".join(full_answer))

    return StreamingResponse(
        _sse_wrapper(history_wrapper()),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================
# 二、试卷模式出题接口（独立接口）
# ============================================

class ExamQuizRequest(BaseModel):
    """试卷模式出题请求"""
    message: str = Field(..., min_length=1, description="出卷要求")
    textbook_collections: list[str] = Field(
        default=["default"],
        description="教材 Collection 名称列表",
    )
    exam_collections: Optional[list[str]] = Field(
        default=None,
        description="试卷 Collection 名称列表（不填则从绑定中自动查找）",
    )


@router.post(
    "/api/quiz/exam",
    summary="试卷模式出题（SSE 流式）",
    description="以参考真题卷为模板，从教材知识点范围仿写完整试卷。需先上传教材和真题卷。",
)
async def quiz_exam(req: ExamQuizRequest):
    """试卷模式出题"""
    try:
        token_generator = generate_exam_quiz_stream(
            message=req.message,
            textbook_collections=req.textbook_collections,
            exam_collections=req.exam_collections,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return StreamingResponse(
        _sse_wrapper(token_generator),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ============================================
# 三、文档上传接口
# ============================================

@router.post(
    "/api/documents/upload",
    summary="上传文档并入库",
    response_model=IngestResponse,
)
async def upload_document(
    file: UploadFile = File(...),
    collection_name: str = Form("default"),
):
    """上传文档 → 解析 → 切分 → 向量化 → 入库"""
    filename = file.filename or "unknown"
    suffix = Path(filename).suffix.lower()
    allowed = {".pdf", ".markdown", ".txt"}
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式 '{suffix}'。支持的格式: {', '.join(allowed)}",
        )

    raw_docs_dir = settings.BASE_DIR / settings.RAW_DOCS_DIR.lstrip("./")
    raw_docs_dir.mkdir(parents=True, exist_ok=True)

    # 同名文件加时间戳防覆盖
    stem = Path(filename).stem
    save_path = raw_docs_dir / f"{stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{suffix}"

    # 文件大小限制: 100MB
    max_size = 100 * 1024 * 1024
    total_size = 0
    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > max_size:
                    f.close()
                    save_path.unlink()
                    raise HTTPException(status_code=413, detail="文件过大，单次上传不得超过 100MB")
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文件保存失败: {str(e)}")

    try:
        chunk_count = ingest_document(
            file_path=str(save_path),
            collection_name=collection_name,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"文档入库失败: {str(e)}")

    return IngestResponse(
        filename=filename,
        collection_name=collection_name,
        chunk_count=chunk_count,
    )


# ============================================
# 四、Collection 列表接口
# ============================================

@router.get(
    "/api/documents",
    summary="列出所有知识库 Collection",
    response_model=CollectionListResponse,
)
async def get_collections():
    """返回 ChromaDB 中所有 Collection 名称"""
    try:
        cols = list_collections()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取失败: {str(e)}")

    return CollectionListResponse(collections=cols, count=len(cols))


# ============================================
# 五、Collection 绑定管理接口
# ============================================

@router.post(
    "/api/collections/bind",
    summary="创建教材-试卷 Collection 绑定",
    response_model=BindingResponse,
)
async def bind_collections(req: BindingRequest):
    """
    创建教材 Collection 与试卷 Collection 的绑定关系。
    绑定后，试卷模式出题会自动找到对应的参考真题卷。
    """
    try:
        binding = create_binding(
            textbook_collections=req.textbook_collections,
            exam_collections=req.exam_collections,
            label=req.label,
        )
        return BindingResponse(**binding)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"绑定失败: {str(e)}")


@router.get(
    "/api/collections/bindings",
    summary="列出所有绑定关系",
    response_model=list[BindingResponse],
)
async def list_bindings():
    """返回所有教材-试卷 Collection 绑定关系"""
    try:
        bindings = get_all_bindings()
        return [BindingResponse(**b) for b in bindings]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {str(e)}")


@router.delete(
    "/api/collections/bindings/{binding_id}",
    summary="删除绑定关系",
)
async def remove_binding(binding_id: str):
    """删除指定 ID 的绑定关系"""
    try:
        deleted = delete_binding(binding_id)
        if not deleted:
            raise HTTPException(status_code=404, detail=f"绑定 {binding_id} 不存在")
        return {"message": "绑定已删除", "id": binding_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {str(e)}")


# ============================================
# 六、对话历史管理接口
# ============================================

from app.services.conversation_service import (
    get_history as get_conv_history,
    delete_conversation,
    list_conversations,
)


@router.get(
    "/api/conversations",
    summary="列出所有对话",
)
async def list_convs():
    """返回所有对话列表（用于前端对话历史面板）"""
    try:
        return {"conversations": list_conversations()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get(
    "/api/conversations/{conversation_id}",
    summary="获取指定对话的历史",
)
async def get_conv(conversation_id: str):
    """返回指定对话的完整历史消息"""
    history = get_conv_history(conversation_id)
    return {"conversation_id": conversation_id, "messages": history}


@router.delete(
    "/api/conversations/{conversation_id}",
    summary="删除对话历史",
)
async def delete_conv(conversation_id: str):
    """删除指定对话的历史"""
    if delete_conversation(conversation_id):
        return {"message": "对话已删除", "conversation_id": conversation_id}
    raise HTTPException(status_code=404, detail="对话不存在")


# ============================================
# 五.2 Collection 名称映射查询
# ============================================

@router.get(
    "/api/collections/aliases",
    summary="获取 Collection 名称映射（中文→内部名）",
)
async def get_aliases():
    """返回 collection_aliases.json 的内容，供前端展示原始中文名称"""
    import json
    aliases_file = settings.BASE_DIR / "data" / "collection_aliases.json"
    if not aliases_file.exists():
        return {"aliases": {}}
    return {"aliases": json.loads(aliases_file.read_text(encoding="utf-8"))}


# ============================================
# 六、健康检查接口
# ============================================

@router.get(
    "/api/health",
    summary="服务健康检查",
)
async def health_check(deep: bool = Query(False, description="是否深度检查依赖组件")):
    """
    健康检查。
    - 浅度: 只验证进程存活
    - 深度: 验证 DeepSeek API / ChromaDB / 嵌入模型
    """
    if not deep:
        return {
            "status": "ok",
            "project": settings.PROJECT_NAME,
            "version": settings.PROJECT_VERSION,
        }

    checks = {
        "process": "ok",
        "deepseek_api": "ok",
        "chromadb": "ok",
        "embedding_model": "ok",
    }

    # 检查 DeepSeek API
    try:
        from app.core.llm_manager import get_llm
        llm = get_llm(streaming=False)
        test_response = llm.invoke("ping")
        if not test_response.content:
            raise RuntimeError("API 返回为空")
    except Exception as e:
        checks["deepseek_api"] = f"error: {str(e)[:100]}"

    # 检查 ChromaDB
    try:
        list_collections()
    except Exception as e:
        checks["chromadb"] = f"error: {str(e)[:100]}"

    # 检查嵌入模型
    try:
        from app.core.llm_manager import get_embeddings
        emb = get_embeddings()
        _ = emb.embed_query("测试文本")
    except Exception as e:
        checks["embedding_model"] = f"error: {str(e)[:100]}"

    all_ok = all(v == "ok" for v in checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "project": settings.PROJECT_NAME,
        "version": settings.PROJECT_VERSION,
        "checks": checks,
    }


# ============================================
# 已知缺陷汇总
# ============================================
#
# 1. 【无认证/鉴权】所有接口裸奔，任何人可调。
# 2. 【无请求频率限制】无限调用可能消耗大量 API 额度。
# 3. 【试卷模式独立接口 /api/quiz/exam 与 /api/chat 分离】
#    - 用户在聊天中说"出一张卷子"会走简单出题模式。
#    - 试卷模式需要前端显式调用 /api/quiz/exam 接口。
#    - [后续扩展] 在意图路由中识别"出一张完整试卷"→自动路由到试卷模式。
# 4. 【无请求日志/监控】缺少请求耗时、成功率等基础指标。
# 5. 【上传文件无大小限制】恶意上传可能撑爆磁盘。
