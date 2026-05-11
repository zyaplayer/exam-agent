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
from app.services.vector_store import search_documents
from app.agent.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT_TEMPLATE


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
        return "（暂无相关参考知识片段，请根据您的通识知识回答）"
        # [缺陷] 当知识库为空时，LLM 会依赖自身训练数据生成答案，
        #   可能产生幻觉或偏离教材表述。应在前端提示用户先上传资料。

    parts: List[str] = []
    for i, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", "")
        # [缺陷] metadata 中可能有更多有用字段（如 chapter, subject, year），
        #   当前只用了 source 和 page，信息利用率低。
        location = f"来源: {source}"
        if page:
            location += f", 第{page}页"

        parts.append(f"【片段 {i}】（{location}）\n{doc.page_content}")

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
    retrieved_docs = search_documents(
        query=question,
        collection_name=collection_name,
        k=k,
    )

    # 步骤2: 构建上下文
    context = _format_retrieved_context(retrieved_docs)

    # 步骤3: 组装 Prompt
    user_prompt = QA_USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    # 步骤4: 调用 LLM
    llm = get_llm(provider="deepseek", temperature=temperature, streaming=False)

    # LangChain ChatOpenAI 的消息格式
    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    response = llm.invoke(messages)

    # [缺陷] llm.invoke 返回 AIMessage，其 .content 就是生成的文本。
    #   但如果 future 切换为非 LangChain 模型（如直接调 openai SDK），
    #   此处代码需全部重写。
    # [后续扩展] 在 llm_manager 中封装统一的 invoke 接口，解耦 LangChain 依赖。

    return response.content


# ============================================
# 第三部分：流式问答（SSE）
# ============================================

async def ask_question_stream(
    question: str,
    collection_name: str = "default",
    k: int | None = None,
    temperature: float = 0.7,
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

    # 步骤1: 向量检索（同步操作，但耗时较短）
    retrieved_docs = search_documents(
        query=question,
        collection_name=collection_name,
        k=k,
    )

    # 步骤2: 构建上下文
    context = _format_retrieved_context(retrieved_docs)

    # 步骤3: 组装 Prompt
    user_prompt = QA_USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    # 步骤4: 调用流式 LLM
    llm = get_llm(provider="deepseek", temperature=temperature, streaming=True)

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # [缺陷] 检索是同步的，阻塞了整个协程。
    #   如果 ChromaDB 检索耗时较长（如 Collection 很大），会造成明显延迟。
    # [后续扩展] 将 search_documents 改为 async，或在线程池中执行。
    # [后续扩展] 先 yield 一个"正在检索知识库..."的状态消息给前端。

    async for chunk in llm.astream(messages):
        # chunk.content 是当前增量 token
        if chunk.content:
            yield chunk.content

    # [缺陷] 流式结束后没有额外处理（如保存对话记录到日志/数据库）。
    # [后续扩展] 在 finally 块中记录完整问答到对话历史。


# ============================================
# 已知缺陷汇总
# ============================================
#
# 1. 【检索与生成耦合】
#    - 检索逻辑（search_documents）和生成逻辑写在同一函数中。
#    - 无法独立测试检索质量或替换检索策略（如混合检索 BM25+向量）。
#    - 后续拆分方向：抽取 RetrievalPipeline 类，将检索策略可配置化。
#
# 2. 【无对话历史】
#    - 每一轮问答都是独立的，LLM 看不到之前对话的上下文。
#    - 用户追问/反问时无法理解上下文（如"那这个定理的逆命题成立吗？"）。
#    - 后续需注入 conversation_history 到 Prompt 中。
#
# 3. 【无 Token 计数与截断】
#    - 检索片段 + System Prompt + 历史对话的总 token 数可能超出模型上下文窗口。
#    - 当前无任何防护，超长时会直接报错或静默截断。
#    - 后续应加入 tiktoken 计数 + 按优先级截断检索片段。
#
# 4. 【无检索质量评估】
#    - 不做任何检查就直接用检索结果生成答案。
#    - 如果检索出的片段完全不相关，LLM 也会强行"缝合"出答案。
#    - 后续应加入相似度分数过滤（< 阈值则告知用户知识库无相关内容）。
#
# 5. 【Collection 管理粗糙】
#    - 只接受 collection_name 字符串参数，无法跨 Collection 混合检索。
#    - 后续可支持多 Collection 联合检索（如同时查"高数"和"线性代数"库）。
#
# 6. 【错误处理缺失】
#    - LLM API 调用失败、ChromaDB 连接断开等异常没有 try/except。
#    - 后续应捕获异常并返回友好的错误提示给前端。
#
# 7. 【缺少引用标注】
#    - 当前通过 Prompt 告知 LLM 基于片段回答，但生成结果中没有
#      标注每一句话的来源（哪个文档、第几页）。
#    - 后续应强制 LLM 在回答中标注引用编号，前端联动展示。
