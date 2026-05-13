"""
考研伴学 Agent - 做题练习链路
===============================
两种出题模式:
  1. 简单出题模式: 用户指定数量/方向/形式/难度，基于向量库知识点出题
  2. 试卷模式:     以参考真题卷为模板，从教材知识点范围仿写完整试卷

所有出题均附带: 标准答案 + 详细解析 + 考察知识点说明
"""

import json
from pathlib import Path
from typing import AsyncIterator, Optional

from app.core.config import settings
from app.core.llm_manager import get_llm
from app.services.vector_store import search_documents
from app.agent.prompts import (
    QUIZ_SIMPLE_SYSTEM_PROMPT,
    QUIZ_SIMPLE_USER_TEMPLATE,
    QUIZ_EXAM_SYSTEM_PROMPT,
    QUIZ_EXAM_USER_TEMPLATE,
)


# ============================================
# 第一部分：Collection 绑定管理
# ============================================

# 绑定关系存储文件
BINDINGS_FILE = settings.BASE_DIR / "data" / "bindings.json"


def _load_bindings() -> list[dict]:
    """
    从 bindings.json 加载所有 Collection 绑定关系。

    返回:
        [
            {
                "id": "b001",
                "textbook_collections": ["高数教材", "线代教材"],
                "exam_collections": ["高数真题2023", "线代真题2023"],
                "label": "数学科目全套"
            },
            ...
        ]
    """
    if not BINDINGS_FILE.exists():
        return []
    try:
        return json.loads(BINDINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def _save_bindings(bindings: list[dict]) -> None:
    """保存绑定关系到文件"""
    BINDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BINDINGS_FILE.write_text(
        json.dumps(bindings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_bindings_for_textbook(textbook_collection: str) -> list[dict]:
    """
    查找与指定教材 Collection 关联的所有试卷 Collection。

    参数:
        textbook_collection: 教材 Collection 名称

    返回:
        匹配的绑定记录列表
    """
    bindings = _load_bindings()
    matched = []
    for b in bindings:
        if textbook_collection in b.get("textbook_collections", []):
            matched.append(b)
    return matched


def get_all_bindings() -> list[dict]:
    """获取所有绑定关系"""
    return _load_bindings()


def create_binding(
    textbook_collections: list[str],
    exam_collections: list[str],
    label: str = "",
) -> dict:
    """
    创建新的 Collection 绑定。

    参数:
        textbook_collections: 教材 Collection 名称列表
        exam_collections:    试卷 Collection 名称列表
        label:               绑定标签（如"数学科目全套"）

    返回:
        新创建的绑定记录
    """
    import uuid

    bindings = _load_bindings()
    new_binding = {
        "id": str(uuid.uuid4())[:8],
        "textbook_collections": textbook_collections,
        "exam_collections": exam_collections,
        "label": label or f"绑定_{len(bindings) + 1}",
    }
    bindings.append(new_binding)
    _save_bindings(bindings)
    return new_binding


def delete_binding(binding_id: str) -> bool:
    """
    删除指定 ID 的绑定。

    返回:
        是否成功删除
    """
    bindings = _load_bindings()
    original_len = len(bindings)
    bindings = [b for b in bindings if b["id"] != binding_id]
    if len(bindings) < original_len:
        _save_bindings(bindings)
        return True
    return False


# ============================================
# 第二部分：检索试卷模板格式
# ============================================

def _collect_exam_format(exam_collections: list[str]) -> str:
    """
    从指定的试卷 Collection 中检索真题，提取题型结构作为格式模板。

    参数:
        exam_collections: 试卷 Collection 名称列表

    返回:
        拼接后的参考试卷文本，供 LLM 模仿格式。
        如果找不到试卷，返回占位提示文本。
    """
    all_samples: list[str] = []
    for col_name in exam_collections:
        # 检索试卷 Collection 中的代表性内容
        # 用通用检索词搜出整张试卷
        docs = search_documents(
            query="试题 答案 解析 选择 填空 计算 证明",
            collection_name=col_name,
            k=10,
        )
        if docs:
            parts = []
            for doc in docs:
                parts.append(doc.page_content)
            sample_text = "\n\n".join(parts)
            all_samples.append(
                f"## 参考试卷: {col_name}\n{sample_text}"
            )
        else:
            all_samples.append(
                f"## 参考试卷: {col_name}\n（Collection 存在但无检索结果）"
            )

    if all_samples:
        return "\n\n---\n\n".join(all_samples)
    else:
        return "（暂无参考试卷，请先上传历年真题到试卷 Collection）"


# ============================================
# 第三部分：简单出题模式
# ============================================

def _parse_simple_quiz_params(message: str) -> dict:
    """
    从用户消息中解析简单出题的参数（基于关键词启发式，不调 LLM）。

    返回:
        {
            "question_count": 题目数量（默认 3）,
            "question_topic": 题目方向（默认"综合"）,
            "question_type":  题目形式（默认"选择题"）,
            "difficulty":     难度（默认"基础"）
        }
    """
    text = message

    # 解析数量
    question_count = 3  # 默认
    import re
    count_match = re.search(r"(\d+)\s*道", text)
    if count_match:
        question_count = int(count_match.group(1))

    # 解析题目形式
    question_type = "选择题"
    if "填空题" in text:
        question_type = "填空题"
    elif "计算题" in text:
        question_type = "计算题"
    elif "证明题" in text:
        question_type = "证明题"
    elif "综合" in text or "混合" in text:
        question_type = "综合（选择题+填空题+计算题）"

    # 解析难度
    difficulty = "基础"
    if "强化" in text:
        difficulty = "强化"
    elif "冲刺" in text or "难题" in text:
        difficulty = "冲刺"

    # 解析题目方向
    # 方式1: 显式关键词 "关于XX" / "方向：XX" / "知识点：XX"
    topic_match = re.search(r"(?:关于|方向[：:]|知识点[：:])\s*(.+?)(?:[，。,.\n]|$)", text)
    if topic_match:
        question_topic = topic_match.group(1).strip()
    else:
        # 方式2: 从 "出N道XX题" 中提取 XX
        #   "出5道极限计算题" → "极限计算"
        #   "来3道线性代数填空题" → "线性代数"
        topic_match = re.search(r"\d+\s*道\s*(.+?)(?:题|$)", text)
        if topic_match:
            raw_topic = topic_match.group(1).strip()
            # 去掉末尾的题型关键词（计算/选择/填空/证明/综合）
            question_topic = re.sub(
                r"(计算|选择|填空|证明|综合|混合)$", "", raw_topic
            ).strip()
        else:
            question_topic = "综合"

    if not question_topic:
        question_topic = "综合"


    return {
        "question_count": min(question_count, 10),  # 单次最多 10 题
        "question_topic": question_topic,
        "question_type": question_type,
        "difficulty": difficulty,
    }


async def generate_simple_quiz_stream(
    message: str,
    collection_name: str = "default",
) -> AsyncIterator[str]:
    """
    简单出题模式（流式）。

    用户: "出5道极限计算题，选择题，强化难度"
    Agent: 基于向量库检索相关知识点 → LLM 生成题目 + 答案 + 解析

    参数:
        message:         用户消息
        collection_name: 教材 Collection 名称

    Yields:
        逐 token 流式返回题目
    """
    # 步骤1: 解析参数
    params = _parse_simple_quiz_params(message)

    # 步骤2: 从向量库检索相关知识点
    import asyncio
    retrieved_docs = await asyncio.to_thread(
        search_documents,
        query=params["question_topic"],
        collection_name=collection_name,
        k=5,
    )

    # 步骤3: 构建上下文
    if retrieved_docs:
        parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "未知")
            parts.append(f"【参考 {i}】来源: {source}\n{doc.page_content}")
        context = "\n\n".join(parts)
    else:
        context = (
            "（知识库中暂无此知识点的参考资料，"
            "请基于通识知识出题，但难度和风格要贴近考研）"
        )

    # 步骤4: LLM 生成题目（流式）
    llm = get_llm(provider="deepseek", temperature=0.6, streaming=True)

    from langchain_core.messages import SystemMessage, HumanMessage

    # 先输出参数摘要
    yield f"## 出题参数\n"
    yield f"- 题目数量: {params['question_count']} 道\n"
    yield f"- 题目方向: {params['question_topic']}\n"
    yield f"- 题目形式: {params['question_type']}\n"
    yield f"- 难度: {params['difficulty']}\n"
    yield f"- 知识来源: {collection_name}\n\n"
    yield "---\n\n"

    user_prompt = QUIZ_SIMPLE_USER_TEMPLATE.format(
        question_count=params["question_count"],
        question_topic=params["question_topic"],
        question_type=params["question_type"],
        difficulty=params["difficulty"],
        context=context,
    )

    messages = [
        SystemMessage(content=QUIZ_SIMPLE_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


# ============================================
# 第四部分：试卷模式
# ============================================

async def generate_exam_quiz_stream(
    message: str,
    textbook_collections: list[str] | None = None,
    exam_collections: list[str] | None = None,
) -> AsyncIterator[str]:
    """
    试卷模式出题（流式）。

    前置条件:
      1. 已上传教材到指定 Collection（有知识点可供出题）
      2. 已上传参考真题卷到试卷 Collection
      3. 教材与试卷已通过前端绑定关联

    如果未指定 exam_collections，会自动从 bindings.json 查找关联。

    参数:
        message:               用户消息（包含出卷要求）
        textbook_collections:  教材 Collection 名称列表
        exam_collections:      试卷 Collection 名称列表（可选，自动查找关联）

    Yields:
        逐 token 流式返回完整试卷
    """
    import asyncio

    textbook_collections = textbook_collections or ["default"]
    exam_collections = exam_collections or []

    # 步骤1: 如果未指定试卷，自动从绑定中查找
    if not exam_collections:
        for tb in textbook_collections:
            bindings = get_bindings_for_textbook(tb)
            for b in bindings:
                exam_collections.extend(b.get("exam_collections", []))
        # 去重
        exam_collections = list(set(exam_collections))

    # 步骤2: 收集试卷模板格式
    exam_format_text = _collect_exam_format(exam_collections)
    if "暂无参考试卷" in exam_format_text:
        yield (
            "## ⚠️ 未找到参考试卷\n\n"
            "试卷模式需要参考真题卷的格式来出题。请：\n"
            "1. 上传历年真题 PDF 到试卷 Collection\n"
            "2. 在前端「知识库管理」中绑定教材和试卷 Collection\n\n"
            "暂时无法生成试卷，请先上传参考试卷资料。"
        )
        return

    # 步骤3: 从教材 Collection 收集知识点范围
    knowledge_parts: list[str] = []
    for col_name in textbook_collections:
        docs = await asyncio.to_thread(
            search_documents,
            query="知识点 章节 定义 定理",
            collection_name=col_name,
            k=8,
        )
        if docs:
            col_parts = []
            for doc in docs:
                header_info = ""
                for level in range(1, 4):
                    h = doc.metadata.get(f"Header_{level}", "")
                    if h:
                        header_info += f"[{h}] "
                col_parts.append(f"{header_info}{doc.page_content}")
            knowledge_parts.append(
                f"### {col_name}\n" + "\n\n".join(col_parts)
            )

    knowledge_scope = (
        "\n\n---\n\n".join(knowledge_parts)
        if knowledge_parts
        else "（知识库中暂无教材知识点，请先上传教材）"
    )

    # 步骤4: 解析用户的对试卷的具体要求
    params = _parse_exam_params(message)

    # 步骤5: LLM 生成试卷（流式）
    llm = get_llm(provider="deepseek", temperature=0.4, streaming=True)

    from langchain_core.messages import SystemMessage, HumanMessage

    # 先输出摘要
    yield f"## 试卷生成参数\n"
    yield f"- 科目: {params['subject']}\n"
    yield f"- 难度: {params['difficulty']}\n"
    yield f"- 参考模板: {', '.join(exam_collections) if exam_collections else '无'}\n"
    yield f"- 教材来源: {', '.join(textbook_collections)}\n\n"
    yield "---\n\n"

    user_prompt = QUIZ_EXAM_USER_TEMPLATE.format(
        subject=params["subject"],
        difficulty=params["difficulty"],
        include_choice=params["include_choice"],
        include_fill=params["include_fill"],
        include_calc=params["include_calc"],
        include_proof=params["include_proof"],
        exam_format=exam_format_text,
        knowledge_scope=knowledge_scope,
    )

    messages = [
        SystemMessage(content=QUIZ_EXAM_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


def _parse_exam_params(message: str) -> dict:
    """
    解析试卷模式的用户参数。

    返回:
        {
            "subject": 科目名称,
            "difficulty": 难度,
            "include_choice": "是/否",
            "include_fill": "是/否",
            "include_calc": "是/否",
            "include_proof": "是/否"
        }
    """
    text = message

    # 科目
    subject = "未指定"
    import re
    subj_match = re.search(r"(?:科目[：:]|考试[：:])\s*(.+?)(?:[，。,.\n]|$)", text)
    if subj_match:
        subject = subj_match.group(1).strip()

    # 难度
    difficulty = "基础"
    if "强化" in text:
        difficulty = "强化"
    elif "冲刺" in text:
        difficulty = "冲刺"

    # 题型包含
    include_choice = "是"  # 选择题默认包含
    if "不要选择" in text or "不含选择" in text:
        include_choice = "否"
    include_fill = "是" if ("填空" in text or "综合" in text) else "否"
    include_calc = "是" if ("计算" in text or "综合" in text) else "否"
    include_proof = "是" if ("证明" in text or "综合" in text) else "否"

    return {
        "subject": subject,
        "difficulty": difficulty,
        "include_choice": include_choice,
        "include_fill": include_fill,
        "include_calc": include_calc,
        "include_proof": include_proof,
    }


# ============================================
# 第五部分：快速咨询（从做题中随时解答疑问）
# ============================================

async def quiz_consult_stream(
    question: str,
    use_rag: bool = True,
    collection_name: str = "default",
) -> AsyncIterator[str]:
    """
    用户在做题过程中遇到不懂的概念，可以快速咨询。

    与 planner_chain.quick_consult_stream 功能相同，此处提供独立入口。
    use_rag=False 时使用 LLM 自身训练知识回答，不强制查向量库。

    参数:
        question:        用户的疑问
        use_rag:         是否启用 RAG
        collection_name: RAG 检索的目标 Collection

    Yields:
        逐 token 流式返回解答
    """
    import asyncio
    from app.agent.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT_TEMPLATE

    if use_rag:
        docs = await asyncio.to_thread(
            search_documents,
            query=question,
            collection_name=collection_name,
            k=4,
        )
        if docs:
            parts = []
            for i, doc in enumerate(docs, start=1):
                source = doc.metadata.get("source", "未知")
                parts.append(f"【参考 {i}】来源: {source}\n{doc.page_content}")
            context = "\n\n".join(parts)
        else:
            context = "（知识库中暂无相关参考资料）"
    else:
        context = "（本题使用通识知识回答，不强制参考教材库）"

    llm = get_llm(provider="deepseek", temperature=0.7, streaming=True)

    from langchain_core.messages import SystemMessage, HumanMessage

    user_prompt = QA_USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context,
    )

    messages = [
        SystemMessage(content=QA_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


# ============================================
# 已知缺陷汇总
# ============================================
#
# 1. 【简单出题的参数解析基于正则，非 LLM】
#    - 参数提取依赖正则表达式和关键词匹配，对复杂表述可能不准确。
#    - 后续可改为 LLM 提取参数（类似 planner_chain 的 _extract_student_info）。
#
# 2. 【试卷模式的知识点检索粗糙】
#    - 使用固定查询词 "知识点 章节 定义 定理"，不一定能覆盖所有科目。
#    - 后续应根据试卷模板中出现的知识点做针对性检索。
#
# 3. 【bindings.json 无并发保护】
#    - 多用户同时操作绑定可能导致写入冲突。
#    - 后续引入文件锁（fcntl/portalocker）或迁移到 SQLite。
#
# 4. 【阅卷模式未实现】
#    - 当前无批改/评分/错因分析功能。
#    - 后续通过 LangGraph 多节点协作实现。
#
# 5. 【试卷模式下教材与试卷的知识对齐未做】
#    - 如果试卷模板涉及的知识点超出了教材范围，LLM 出题可能跑偏。
#    - 后续加入知识点交集分析，自动筛选共同覆盖范围出题。
#
# 6. 【无题目持久化】
#    - 生成的题目不保存，刷新即丢失。
#    - 后续可存入"我的题库" Collection，支持回顾和重做。
