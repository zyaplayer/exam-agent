"""
考研伴学 Agent - RAG 知识问答链路
===================================
核心链路：向量检索 → Prompt 组装 → LLM 生成 → 流式返回。
支持普通模式和流式模式（SSE）两种调用方式。
"""

from typing import AsyncIterator, List

from langchain_core.documents import Document

from app.core.config import settings
from app.core.llm_manager import get_llm
from app.services.vector_store import hybrid_search as search_documents
from app.agent.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT_TEMPLATE
from app.agent.query_rewriter import rewrite_query


# ============================================
# 第一部分：上下文构建
# ============================================

def _format_retrieved_context(docs: List[Document]) -> str:
    """
    将检索到的文档片段格式化为 Prompt 中可用的上下文字符串。

    格式示例:
        【片段 1】（来源: 线性代数.pdf, 第3页）
        矩阵的秩定义为...

        【片段 2】（来源: 高数笔记.markdown）
        拉格朗日中值定理...

    参数:
        docs: 从向量库检索的相关 Document 列表

    返回:
        格式化后的上下文字符串。如果 docs 为空，返回占位提示文本。
    """
    if not docs:
        return ""

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        # 提取 Markdown 标题层级信息（Header_1=章, Header_2=节, Header_3=小节）
        header_parts = []
        for level in range(1, settings.MARKDOWN_HEADER_MAX_LEVEL + 1):
            header_text = doc.metadata.get(f"Header_{level}", "")
            if header_text:
                header_parts.append(header_text)
        hierarchy = " → ".join(header_parts) if header_parts else ""

        location = f"来源: {source}"
        if page:
            location += f", 第{page}页"
        if hierarchy:
            location += f", 章节: {hierarchy}"

        content = doc.page_content
        if len(content) > 400:
            content = content[:300] + "\n...(已截断)...\n" + content[-100:]
        parts.append(f"【片段 [编号{i}]】（{location}）\n{content}")




    return "\n\n".join(parts)


# ============================================
# 第二部分：核心问答
# ============================================

def ask_question(
    question: str,
    collection_name: str = "default",
    k: int | None = None,
    temperature: float = 0.7,
) -> str:
    """
    RAG 知识问答（非流式），返回完整回答字符串。

    适用于: 命令行测试、调试、批量问答场景。

    参数:
        question:        学生的问题
        collection_name: 要检索的 ChromaDB Collection 名称
        k:               检索片段数（默认取配置值）
        temperature:    LLM 温度

    返回:
        LLM 生成的中文回答（完整字符串）

    示例:
        >>> answer = ask_question("什么是矩阵的秩？")
        >>> print(answer)
    """
    if k is None:
        k = settings.RETRIEVAL_K

        # 步骤1: 向量检索
    from app.services.vector_store import hierarchical_search
    retrieved_docs = hierarchical_search(
        query=question,
        collection_name=collection_name,
        k=k,
    )

    # 步骤2: 知识库为空时直接拒绝回答，不调用 LLM
    if not retrieved_docs:
        return (
            "当前知识库中暂无相关内容，无法回答你的问题。\n\n"
            "建议：\n"
            "1. 先上传相关教材或笔记到知识库\n"
            "2. 或者换一个知识库中已有资料的问题试试"
        )
    
    # 步骤2.5: 最低相关性门槛（最高分片段不够相关 → 拒绝回答）
    
    if retrieved_docs:
        top_score = max(
            doc.metadata.get("rerank_score", doc.metadata.get("combined_score", 0))
            for doc in retrieved_docs
        )
        if top_score < 0.3:
            return (
                "抱歉，我在教材中没有找到与你的问题高度相关的内容。\n\n"
                "建议：\n"
                "1. 尝试换一种表述方式提问\n"
                "2. 检查是否已上传了相关科目的教材\n"
                "3. 直接查阅教材中对应的章节"
            )


    # 步骤3: 构建上下文
    context = _format_retrieved_context(retrieved_docs)


    # 步骤4: 组装 Prompt
    user_prompt = QA_USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
        conversation_history="",
    )


    # 步骤5: 调用 LLM
    try:
        llm = get_llm(provider="deepseek", temperature=temperature, streaming=False)

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            return "⚠️ API Key 未配置或已失效，请在 .env 中设置正确的 DEEPSEEK_API_KEY。"
        if "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            return "⚠️ DeepSeek API 响应超时，请稍后重试。"
        return f"⚠️ AI 服务暂时不可用，请稍后重试。（错误: {error_msg[:80]}）"



# ============================================
# 第三部分：流式问答（SSE）
# ============================================

async def ask_question_stream(
    question: str,
    collection_name: str = "default",
    k: int | None = None,
    temperature: float = 0.7,
    conversation_id: str = "",
) -> AsyncIterator[str]:
    """
    RAG 知识问答（流式），逐 token 返回生成内容。

    适用于: FastAPI SSE 接口，前端实时展示打字效果。

    参数:
        question:        学生的问题
        collection_name: 要检索的 ChromaDB Collection 名称
        k:               检索片段数（默认取配置值）
        temperature:    LLM 温度

    Yields:
        每次 yield 一个 token 字符串

    使用示例（FastAPI SSE）:
        @app.get("/api/chat/stream")
        async def chat_stream(q: str):
            return StreamingResponse(
                ask_question_stream(q),
                media_type="text/event-stream",
            )
    """
    if k is None:
        k = settings.RETRIEVAL_K

    # 步骤1: 查询重写（指代消解）
    rewritten_question = question
    if conversation_id:
        import asyncio
        rewritten_question = await rewrite_query(question, conversation_id)

    # 步骤1.5: 检索（使用改写后的问题）
    from app.services.vector_store import hierarchical_search
    retrieved_docs = hierarchical_search(
        query=rewritten_question,   # ← 改这里
        collection_name=collection_name,
        k=k,
    )


    # 步骤2: 知识库为空时直接拒绝回答，不调用 LLM
    if not retrieved_docs:
        message = (
            "当前知识库中暂无相关内容，无法回答你的问题。\n\n"
            "建议：\n"
            "1. 先上传相关教材或笔记到知识库\n"
            "2. 或者换一个知识库中已有资料的问题试试"
        )
        for char in message:
            yield char
        return
    


    # 步骤3: 构建上下文
    context = _format_retrieved_context(retrieved_docs)


        # 步骤4: 组装 Prompt（注入对话历史 + Token 截断保护）
    from app.services.conversation_service import format_history_for_prompt
    from app.utils.token_counter import check_and_truncate
    conversation_history = format_history_for_prompt(conversation_id, max_turns=5) if conversation_id else ""


    user_prompt = check_and_truncate(
        system_prompt=QA_SYSTEM_PROMPT,
        user_prompt_template=QA_USER_PROMPT_TEMPLATE,
        docs=retrieved_docs,
        conversation_history=conversation_history,
        question=question,
    )


    # 步骤5: 调用流式 LLM
    try:
        llm = get_llm(provider="deepseek", temperature=temperature, streaming=True)

        from langchain_core.messages import SystemMessage, HumanMessage

        messages = [
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        async for chunk in llm.astream(messages):
            if chunk.content:
                yield chunk.content
    except Exception as e:
        error_msg = str(e)
        if "api_key" in error_msg.lower() or "unauthorized" in error_msg.lower():
            yield "⚠️ API Key 未配置或已失效，请在 .env 中设置正确的 DEEPSEEK_API_KEY。"
        elif "timeout" in error_msg.lower() or "timed out" in error_msg.lower():
            yield "⚠️ DeepSeek API 响应超时，请稍后重试。"
        else:
            yield f"⚠️ AI 服务暂时不可用，请稍后重试。（{error_msg[:60]}）"

