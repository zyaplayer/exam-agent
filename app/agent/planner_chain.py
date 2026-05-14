"""
考研伴学 Agent - 学习规划链路
===============================
基于考试时间、科目盘点、教材章节目录，生成备考计划。
支持两种模式:
  - 时间线模式: 用户提到考试日期时，生成三阶段倒排计划
  - 灵活模式:   用户未提日期时，按章节内容量编排学习路线
支持生成知识点"学习卡片"，以及与 QA 链路联动解答疑问。
"""

import json
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import AsyncIterator, Optional

from app.core.config import settings
from app.core.llm_manager import get_llm
from app.services.vector_store import search_documents, list_collections
from app.agent.prompts import (
    PLANNER_EXTRACT_SYSTEM_PROMPT,
    PLANNER_EXTRACT_USER_TEMPLATE,
    PLANNER_GENERATE_SYSTEM_PROMPT,
    PLANNER_GENERATE_USER_TEMPLATE,
    PLANNER_FLEXIBLE_SYSTEM_PROMPT,
    PLANNER_FLEXIBLE_USER_TEMPLATE,
    PLANNER_CARD_SYSTEM_PROMPT,
    PLANNER_CARD_USER_TEMPLATE,
)


# ============================================
# 第一部分：时间系统
# ============================================

def calculate_remaining_days(exam_date_str: str) -> int:
    """
    根据考试日期计算剩余备考天数。

    参数:
        exam_date_str: YYYY-MM-DD 格式的考试日期字符串

    返回:
        剩余天数。解析失败或日期已过返回 -1。
    """
    try:
        exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
        today = date.today()
        remaining = (exam_date - today).days
        return max(0, remaining)
    except (ValueError, TypeError):
        return -1


def get_phase_dates(remaining_days: int) -> dict:
    """
    根据剩余天数计算三阶段的起止日期。

    三阶段时间分配:
        - 基础阶段: 50%
        - 强化阶段: 30%
        - 冲刺阶段: 20%

    返回:
        {
            "base_start": "...",  "base_end": "...",
            "strengthen_start": "...", "strengthen_end": "...",
            "sprint_start": "...", "sprint_end": "...",
            "base_days": N, "strengthen_days": N, "sprint_days": N
        }
    """
    today = date.today()
    phase1_days = max(1, int(remaining_days * 0.5))
    phase2_days = max(1, int(remaining_days * 0.3))
    phase3_days = max(1, remaining_days - phase1_days - phase2_days)

    base_start = today
    base_end = base_start + timedelta(days=phase1_days - 1)
    strengthen_start = base_end + timedelta(days=1)
    strengthen_end = strengthen_start + timedelta(days=phase2_days - 1)
    sprint_start = strengthen_end + timedelta(days=1)
    sprint_end = sprint_start + timedelta(days=phase3_days - 1)

    return {
        "base_start": base_start.strftime("%Y-%m-%d"),
        "base_end": base_end.strftime("%Y-%m-%d"),
        "strengthen_start": strengthen_start.strftime("%Y-%m-%d"),
        "strengthen_end": strengthen_end.strftime("%Y-%m-%d"),
        "sprint_start": sprint_start.strftime("%Y-%m-%d"),
        "sprint_end": sprint_end.strftime("%Y-%m-%d"),
        "base_days": phase1_days,
        "strengthen_days": phase2_days,
        "sprint_days": phase3_days,
    }


# ============================================
# 第二部分：科目盘点 / 教材大纲加载
# ============================================

def scan_available_subjects() -> list[dict]:
    """
    扫描 ChromaDB 中已有的 Collection 列表，
    以及 data/summary_db/ 中的教材目录大纲文件。

    返回:
        [
            {
                "collection_name": "高数教材",
                "has_outline": True,
                "outline_file": "E:/.../data/summary_db/高数教材.markdown"
            },
            ...
        ]
    """
    collections = list_collections()
    summary_dir = settings.BASE_DIR / settings.SUMMARY_DB_DIR.lstrip("./")
    summary_dir.mkdir(parents=True, exist_ok=True)

    # 建立"文件名(不含后缀) → 完整路径"的映射
    outline_files: dict[str, str] = {}
    if summary_dir.exists():
        for f in summary_dir.glob("*.markdown"):
            outline_files[f.stem] = str(f)
        for f in summary_dir.glob("*.md"):
            outline_files[f.stem] = str(f)
        for f in summary_dir.glob("*.txt"):
            outline_files[f.stem] = str(f)

    subjects = []
    for col_name in collections:
        subject_info: dict = {
            "collection_name": col_name,
            "has_outline": col_name in outline_files,
            "outline_file": outline_files.get(col_name),
        }
        subjects.append(subject_info)

    # [缺陷] 没有统计每个 Collection 的具体 chunk 数量。
    # [后续扩展] 调用 ChromaDB .count() 获取每个 Collection 的知识点密度。

    return subjects


def load_outline_content(outline_file: str) -> str:
    """
    加载教材章节目录大纲文件的内容。

    参数:
        outline_file: 大纲文件的路径

    返回:
        大纲文本内容。文件不存在时返回空字符串。
    """
    path = Path(outline_file)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return ""


# ============================================
# 第三部分：信息提取（从用户消息中提取关键参数）
# ============================================

async def _extract_student_info(message: str) -> dict:
    """
    使用 LLM 从用户消息中提取备考规划所需的关键信息。

    返回的 dict 结构:
        {
            "target_school": 目标院校或 null,
            "target_major": 目标专业或 null,
            "exam_date": "YYYY-MM-DD" 或 null,
            "weak_subjects": [弱项科目列表],
            "strong_subjects": [强项科目列表],
            "available_hours_per_day": 每天可用小时或 null,
            "additional_notes": 其他备注或 null
        }
    """
    llm = get_llm(provider="deepseek", temperature=0.1, streaming=False)

    from langchain_core.messages import SystemMessage, HumanMessage

    messages = [
        SystemMessage(content=PLANNER_EXTRACT_SYSTEM_PROMPT),
        HumanMessage(
            content=PLANNER_EXTRACT_USER_TEMPLATE.format(message=message)
        ),
    ]

    response = llm.invoke(messages)
    raw = response.content.strip()

    # 尝试从响应中提取 JSON（可能被包裹在 markdown 代码块中）
    json_match = re.search(r"\{[\s\S]*\}", raw)
    if json_match:
        raw = json_match.group(0)

    try:
        result = json.loads(raw)
        # 确保所有字段都存在（LLM 可能漏字段）
        defaults = {
            "target_school": None,
            "target_major": None,
            "exam_date": None,
            "weak_subjects": [],
            "strong_subjects": [],
            "available_hours_per_day": None,
            "additional_notes": None,
        }
        for key, default_val in defaults.items():
            if key not in result:
                result[key] = default_val
        return result
    except json.JSONDecodeError:
        return defaults


# ============================================
# 第四部分：规划生成（核心）
# ============================================

async def generate_plan_stream(
    message: str,
    exam_date_override: Optional[str] = None,
    conversation_id: str = "",
) -> AsyncIterator[str]:
    """
    根据用户消息生成备考计划（流式返回）。

    两种模式:
      - 时间线模式: 用户提到考试日期 → 三阶段倒排计划
      - 灵活模式:   用户未提日期 → 按章节内容量编排路线

    参数:
        message:            用户的输入消息
        exam_date_override: 手动指定考试日期（可选，YYYY-MM-DD）

    Yields:
        逐 token 流式返回生成的计划文本
    """
    # ---- 步骤1: 提取用户信息 ----
    student_info = await _extract_student_info(message)

    # ---- 步骤2: 盘点可用科目 ----
    subjects = scan_available_subjects()

    subject_descriptions: list[str] = []
    outline_texts: list[str] = []
    weak_subjects = student_info.get("weak_subjects") or []
    strong_subjects = student_info.get("strong_subjects") or []

    for subj in subjects:
        desc = f"- {subj['collection_name']}"
        if subj.get("has_outline"):
            desc += "（有章节目录大纲）"
            outline_content = load_outline_content(subj["outline_file"])
            if outline_content:
                outline_texts.append(
                    f"\n### {subj['collection_name']} 章节大纲\n{outline_content}"
                )
        subject_descriptions.append(desc)

    # ---- 步骤3: 判断是否有考试日期 → 选择模式 ----
    exam_date = exam_date_override or student_info.get("exam_date")
    use_timeline = False
    phase_dates: dict = {}
    remaining_days = 0

    if exam_date:
        remaining_days = calculate_remaining_days(exam_date)
        if remaining_days > 0:
            use_timeline = True
            phase_dates = get_phase_dates(remaining_days)

    # ---- 步骤4: 根据模式调用 LLM ----
    llm = get_llm(provider="deepseek", temperature=0.5, streaming=True)

    from langchain_core.messages import SystemMessage, HumanMessage

    # 注入对话历史
    from app.services.conversation_service import format_history_for_prompt
    conversation_history = format_history_for_prompt(conversation_id) if conversation_id else ""

    if use_timeline:
        # ---- 时间线模式 ----
        system_prompt = PLANNER_GENERATE_SYSTEM_PROMPT
        user_prompt = PLANNER_GENERATE_USER_TEMPLATE.format(
            target_school=student_info.get("target_school") or "未指定",
            target_major=student_info.get("target_major") or "未指定",
            exam_date=exam_date,
            remaining_days=remaining_days,
            weak_subjects=", ".join(weak_subjects) if weak_subjects else "未指定",
            strong_subjects=", ".join(strong_subjects) if strong_subjects else "未指定",
            available_hours_per_day=student_info.get("available_hours_per_day") or "未指定",
            available_collections="\n".join(subject_descriptions)
                if subject_descriptions
                else "（知识库中暂无教材）",
            subject_outlines="\n".join(outline_texts)
                if outline_texts
                else "（暂无章节目录大纲，请在 data/summary_db/ 中放入教材目录文件）",
            conversation_history=conversation_history,
        )
        # 先输出时间线信息
        yield f"## 备考时间规划\n"
        yield f"- 考试日期: {exam_date}\n"
        yield f"- 剩余天数: {remaining_days} 天\n"
        yield f"- 基础阶段: {phase_dates['base_start']} ~ {phase_dates['base_end']} "
        yield f"({phase_dates['base_days']} 天)\n"
        yield f"- 强化阶段: {phase_dates['strengthen_start']} ~ {phase_dates['strengthen_end']} "
        yield f"({phase_dates['strengthen_days']} 天)\n"
        yield f"- 冲刺阶段: {phase_dates['sprint_start']} ~ {phase_dates['sprint_end']} "
        yield f"({phase_dates['sprint_days']} 天)\n\n"
    else:
        # ---- 灵活模式（无考试日期） ----
        system_prompt = PLANNER_FLEXIBLE_SYSTEM_PROMPT
        user_prompt = PLANNER_FLEXIBLE_USER_TEMPLATE.format(
            target_school=student_info.get("target_school") or "未指定",
            target_major=student_info.get("target_major") or "未指定",
            weak_subjects=", ".join(weak_subjects) if weak_subjects else "未指定",
            strong_subjects=", ".join(strong_subjects) if strong_subjects else "未指定",
            available_hours_per_day=student_info.get("available_hours_per_day") or "未指定",
            available_collections="\n".join(subject_descriptions)
                if subject_descriptions
                else "（知识库中暂无教材）",
            subject_outlines="\n".join(outline_texts)
                if outline_texts
                else "（暂无章节目录大纲，请在 data/summary_db/ 中放入教材目录文件）",
            conversation_history=conversation_history,
        )
        yield "## 灵活学习路线（无固定日期）\n\n"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


# ============================================
# 第五部分：学习卡片生成
# ============================================

async def generate_learning_card_stream(
    knowledge_point: str,
    collection_name: str = "default",
) -> AsyncIterator[str]:
    """
    为指定知识点生成"学习卡片"（流式）。

    学习卡片包含：核心定义 + 关键公式 + 典型例题 + 易错点 + 关联知识。

    参数:
        knowledge_point: 知识点名称（如"拉格朗日中值定理"）
        collection_name: 要检索的 Collection 名称

    Yields:
        逐 token 流式返回学习卡片内容
    """
    import asyncio

    # 步骤1: 从向量库检索相关片段
    retrieved_docs = await asyncio.to_thread(
        search_documents,
        query=knowledge_point,
        collection_name=collection_name,
        k=3,
    )

    # 步骤2: 构建上下文
    if retrieved_docs:
        parts = []
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "未知")
            parts.append(f"【参考 {i}】来源: {source}\n{doc.page_content}")
        context = "\n\n".join(parts)
    else:
        context = "（知识库中暂无此知识点的参考资料，请基于通识知识生成学习卡片）"

    # 步骤3: LLM 生成卡片（流式）
    llm = get_llm(provider="deepseek", temperature=0.3, streaming=True)

    from langchain_core.messages import SystemMessage, HumanMessage

    user_prompt = PLANNER_CARD_USER_TEMPLATE.format(
        knowledge_point=knowledge_point,
        context=context,
    )

    messages = [
        SystemMessage(content=PLANNER_CARD_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


# ============================================
# 第六部分：快速咨询接口
# ============================================

async def quick_consult_stream(
    question: str,
    use_rag: bool = True,
    collection_name: str = "default",
) -> AsyncIterator[str]:
    """
    在学习规划或做题过程中，用户可以快速咨询不懂的知识点。

    与 qa_chain 的区别:
        - use_rag=True  : 先从向量库检索，再基于检索结果回答
        - use_rag=False : 直接使用 LLM 自身训练数据回答（不检索）

    参数:
        question:        用户的问题
        use_rag:         是否启用 RAG 检索
        collection_name: RAG 检索的目标 Collection

    Yields:
        逐 token 流式返回答案
    """
    import asyncio
    from app.agent.prompts import QA_SYSTEM_PROMPT, QA_USER_PROMPT_TEMPLATE

    if use_rag:
        retrieved_docs = await asyncio.to_thread(
            search_documents,
            query=question,
            collection_name=collection_name,
            k=4,
        )
        if retrieved_docs:
            parts = []
            for i, doc in enumerate(retrieved_docs, start=1):
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
# 1. 【书目盘点缺少 chunk 统计】
#    - scan_available_subjects 不返回每个 Collection 的具体文档量。
#    - 导致计划无法区分"有大量资料"和"只有空壳"的科目。
#
# 2. 【大纲文件匹配靠文件名精确匹配】
#    - 当前按 Collection 名匹配同名大纲文件，没有模糊匹配或别名。
#    - 后续应支持别名映射或通过 bindings.json 关联。
#
# 3. 【信息提取 LLM JSON 遵从度不稳定】
#    - DeepSeek 模型可能输出非纯 JSON（包裹在 ```json 代码块中）。
#    - 当前有正则提取 + 字段补全兜底，但不能保证所有字段正确。
#
# 4. 【学习卡片不持久化】
#    - 生成后不保存，刷新即丢失。
#    - 后续可存入 ChromaDB 专用 Collection，支持"我的学习卡片"管理。
#
# 5. 【快速咨询与 qa_chain 存在功能重叠】
#    - quick_consult_stream 和 qa_chain.ask_question_stream 逻辑高度相似。
#    - 后续应抽取公共基类，两个函数作为参数化的变体。
#
# 6. 【无多科目联合规划】
#    - 当前一次规划一个科目，无法同时为"高数+英语+政治"生成交叉计划。
#    - 后续应支持多科目时间片分配策略。
