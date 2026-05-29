"""
考研伴学 Agent - 查询重写服务
===============================
指代消解：将模糊追问（"它的逆命题呢？"）转化为完整问题。
"""

from app.core.llm_manager import get_llm
from app.agent.prompts import QUERY_REWRITE_PROMPT


async def rewrite_query(
    question: str,
    conversation_id: str = "",
) -> str:
    """
    对用户提问做指代消解，使问题自包含全部语境。

    参数:
        question:        用户原始提问
        conversation_id: 对话ID（用于获取历史上下文）

    返回:
        改写后的问题。无历史/无指代词时原样返回。

    示例:
        >>> rewrite_query("它的逆命题成立吗？", convo_id)
        "拉格朗日中值定理的逆命题成立吗？"
        >>> rewrite_query("什么是极限？", "")
        "什么是极限？"  # 无需改写
    """
    # 无对话历史 → 无需改写
    if not conversation_id:
        return question

    from app.services.conversation_service import get_history

    history = get_history(conversation_id)
    # 历史为空或只有 1 轮 → 无需改写
    if len(history) < 2:
        return question

    # 取最近 3 轮历史作为上下文
    recent = history[-6:]  # 3 轮 = 6 条消息
    history_text = "\n".join(
        f"{'学生' if m['role'] == 'user' else '助教'}: {m['content']}"
        for m in recent
    )

    # 快速判断是否需要 LLM 改写（避免无必要的 API 调用）
    # 如果提问不含指代词，直接跳过
    has_pronoun = any(
        w in question for w in ["它", "这", "那", "这个", "那个", "这种", "那种", "其", "该"]
    )
    if not has_pronoun:
        return question

    # 调用轻量 LLM 做指代消解
    try:
        llm = get_llm(provider="deepseek", temperature=0.0, streaming=False)

        from langchain_core.messages import HumanMessage

        prompt = QUERY_REWRITE_PROMPT.format(
            history=history_text,
            question=question,
        )
        messages = [HumanMessage(content=prompt)]
        response = llm.invoke(messages)
        rewritten = response.content.strip()

        # 如果 LLM 返回了异常长的内容或空内容，保护性回退
        if not rewritten or len(rewritten) > 200:
            return question

        return rewritten
    except Exception:
        # LLM 调用失败 → 静默回退到原文
        return question
