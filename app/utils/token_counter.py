"""
考研伴学 Agent - Token 计数器与截断工具
==========================================
提供 token 估算和上下文截断功能，防止超出模型窗口限制。
"""

from typing import List
from langchain_core.documents import Document
from app.core.config import settings


# DeepSeek V4 Flash 上下文窗口为 1M tokens，但为安全保留余量
# 实际使用中，Prompt 过长会增加延迟和成本，建议控制在合理范围
MAX_CONTEXT_TOKENS = getattr(settings, "MAX_CONTEXT_TOKENS", 8000)


def estimate_tokens(text: str) -> int:
    """
    估算文本的 token 数量。

    中文场景估算规则:
      - 中文字符 ~1.5 chars/token
      - 英文字符 ~4 chars/token
      - 混合时取保守值 ~2 chars/token

    用于截断判断，不要求精确。
    """
    if not text:
        return 0
    # 保守估计：每 2 个字符 = 1 token
    return len(text) // 2


def estimate_messages_tokens(
    system_prompt: str,
    user_prompt: str,
    conversation_history: str = "",
) -> int:
    """
    估算整个 Prompt 的总 token 数。

    参数:
        system_prompt:        System 提示词
        user_prompt:          User 提示词（含检索上下文）
        conversation_history: 对话历史文本

    返回:
        估算的 token 总数
    """
    total = 0
    total += estimate_tokens(system_prompt)
    total += estimate_tokens(user_prompt)
    if conversation_history:
        total += estimate_tokens(conversation_history)
    # 预留 100 tokens 余量给 ChatOpenAI 的内部消息格式包装
    return total + 100


def truncate_documents(
    docs: List[Document],
    max_doc_tokens: int,
) -> List[Document]:
    """
    按相关性分数截断文档列表，保留最重要的文档。

    策略:
      1. 优先保留 relevance_score 更高的文档
      2. 从后往前逐文档截断，直到总 token 数低于上限
      3. 如果单个文档超过上限，截断其内容并标注

    参数:
        docs:           检索出的文档列表（已按分数排序）
        max_doc_tokens: 文档部分允许的最大 token 数

    返回:
        截断后的文档列表
    """
    if not docs:
        return docs

    kept: List[Document] = []
    current_tokens = 0

    for doc in docs:
        doc_tokens = estimate_tokens(doc.page_content)
        if current_tokens + doc_tokens <= max_doc_tokens:
            kept.append(doc)
            current_tokens += doc_tokens
        elif current_tokens < max_doc_tokens:
            # 最后一个文档：截断填充剩余空间
            remaining = max_doc_tokens - current_tokens
            if remaining > 100:  # 至少留 100 tokens 才有意义
                truncated_text = doc.page_content[: remaining * 2] + "\n...[已截断]"
                truncated_doc = Document(
                    page_content=truncated_text,
                    metadata={**doc.metadata, "truncated": True},
                )
                kept.append(truncated_doc)
            break  # 空间用尽
        else:
            break  # 一个文档也放不下，直接停止

    return kept


def check_and_truncate(
    system_prompt: str,
    user_prompt_template: str,
    docs: List[Document],
    conversation_history: str = "",
    question: str = "",
    context_placeholder: str = "{context}",
    history_placeholder: str = "{conversation_history}",
) -> str:
    """
    完整的 Token 检查与截断流程。

    步骤:
      1. 估算总 token
      2. 超出上限 → 截断检索文档
      3. 仍超出 → 截断对话历史
      4. 重新组装 user_prompt

    参数:
        system_prompt:        系统提示词
        user_prompt_template: 用户提示词模板（含占位符）
        docs:                 检索文档列表
        conversation_history: 对话历史文本
        question:             当前问题
        context_placeholder:  上下文占位符
        history_placeholder:  历史占位符

    返回:
        最终的 user_prompt 字符串（已截断）
    """
    # 先格式化上下文看看总 token
    context = "\n\n".join([d.page_content for d in docs])
    
    user_prompt = user_prompt_template.format(
        question=question,
        context=context,
        conversation_history=conversation_history,
    )

    total = estimate_messages_tokens(system_prompt, user_prompt, conversation_history)

    if total <= MAX_CONTEXT_TOKENS:
        return user_prompt  # 未超限，直接返回

    # ---- 超出上限，开始截断 ----
    # 计算可用于文档和历史的 token 预算
    overhead = (
        estimate_tokens(system_prompt)
        + estimate_tokens(question)
        + 200  # 模板固定文本
    )
    available = MAX_CONTEXT_TOKENS - overhead

    # 对话历史占 30%，检索文档占 70%
    history_budget = int(available * 0.3) if conversation_history else 0
    doc_budget = available - history_budget

    # 截断文档
    truncated_docs = truncate_documents(docs, doc_budget)

    # 重新组装
    context = "\n\n".join([d.page_content for d in truncated_docs])
    
    # 截断历史（简单的取后段）
    truncated_history = conversation_history
    if conversation_history and estimate_tokens(conversation_history) > history_budget:
        # 按字符粗略截断历史的后面部分
        max_history_chars = history_budget * 2
        truncated_history = "...[历史已截断]\n" + conversation_history[-max_history_chars:]

    return user_prompt_template.format(
        question=question,
        context=context,
        conversation_history=truncated_history,
    )
