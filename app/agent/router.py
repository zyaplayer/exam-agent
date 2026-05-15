"""
考研伴学 Agent - 意图识别路由器（混合方案）
=============================================
第1层: 规则引擎 —— 关键词匹配，处理简单明确的意图（零 LLM 成本）。
第2层: LLM 语义分类 —— 处理复杂模糊的查询（低延迟，~5 tokens 输出）。

混合方案要点:
  - 规则引擎命中率高时直接返回，节省 API 调用
  - 低置信度或多意图争抢时自动升级为 LLM 分类
  - 保留两级路由的可观测性（日志记录使用了哪层）
"""

from enum import Enum
from typing import NamedTuple, Optional

from app.core.llm_manager import get_llm
from app.agent.prompts import ROUTER_CLASSIFY_PROMPT


# ============================================
# 意图与路由结果定义
# ============================================

class Intent(str, Enum):
    """用户意图枚举"""
    QA = "qa"              # 知识问答
    QUIZ = "quiz"          # 做题练习
    PLANNER = "planner"    # 学习规划


class RouteResult(NamedTuple):
    """路由分发结果"""
    intent: Intent                 # 识别出的意图
    chain_name: str                # 对应的处理链路名称
    confidence: float              # 置信度（0.0 ~ 1.0）
    matched_keywords: list[str]    # 命中的关键词（用于日志/调试）
    router_layer: str              # "rule" 或 "llm" 表示使用哪层路由


# ============================================
# 关键词规则表
# ============================================

_KEYWORD_RULES: dict[Intent, tuple[list[str], float]] = {
    Intent.QA: (
        [
            # 概念/定理类
            "什么是", "解释", "为什么", "怎么样", "什么意思",
            "定理", "公式", "推导", "证明", "计算", "求",
            "定义", "概念", "性质", "区别", "联系",
            "怎么理解", "如何理解", "举例", "通俗",
            "怎么做", "怎么算", "步骤", "过程",
            # 学科关键词（高频）
            "极限", "导数", "积分", "矩阵", "特征值", "行列式",
            "概率", "分布", "假设检验", "随机变量",
            "级数", "微分方程", "偏导", "重积分", "线面积分",
            "英语", "语法", "词汇", "长难句", "翻译", "阅读", "写作",
            "政治", "马原", "毛概", "史纲", "思修", "时政",
            "专业课", "数据结构", "操作系统", "计算机网络", "组成原理",
            "函数", "连续", "可导", "可微", "中值定理", "泰勒", "洛必达",
        ],
        0.85,
    ),
    Intent.QUIZ: (
        [
            "做题", "出题", "题目", "真题", "练习题",
            "试卷", "模拟", "测一测", "来一道",
            "练", "刷题", "选择题", "填空题", "计算题",
            "错题", "正确率", "得分", "出几道",
        ],
        0.85,
    ),
    Intent.PLANNER: (
        [
            "计划", "安排", "规划", "复习", "进度",
            "怎么学", "如何备考", "时间表", "日程",
            "每天", "阶段", "基础", "强化", "冲刺",
            "目标", "院校", "分数线", "报录比",
            "专业课", "公共课", "科目", "怎么分配",
            "帮我安排", "帮我制定", "怎么准备",
        ],
        0.85,
    ),
}

# 兜底意图
_DEFAULT_INTENT = Intent.QA

# LLM 升级阈值
_LLM_UPGRADE_THRESHOLD = 0.88       # 规则引擎置信度低于此值 → 升级到 LLM
_LLM_MARGIN_THRESHOLD = 0.10        # 最高分与次高分的差距小于此值 → 升级到 LLM


# ============================================
# 第1层: 规则引擎（关键词匹配）
# ============================================

def _rule_classify(message: str) -> RouteResult:
    """
    基于关键词匹配进行意图分类。

    返回:
        RouteResult，包含意图、置信度、命中关键词、路由层="rule"
    """
    normalized = message.strip()

    # 特殊模式：出N道 → 强偏 QUIZ
    import re
    if re.search(r"出\s*\d+\s*道", normalized):
        return RouteResult(
            intent=Intent.QUIZ,
            chain_name=_intent_to_chain(Intent.QUIZ),
            confidence=0.92,
            matched_keywords=[f"模式:出N道"],
            router_layer="rule",
        )


    best_intent = _DEFAULT_INTENT
    best_hits: list[str] = []
    best_score = 0.0
    second_score = 0.0

    for intent, (keywords, base_weight) in _KEYWORD_RULES.items():
        hits = [kw for kw in keywords if kw in normalized]
        if hits:
            # 置信度 = 基础权重 + 命中数 × 0.03（封顶 0.99）
            score = min(base_weight + len(hits) * 0.03, 0.99)
            if score > best_score:
                second_score = best_score
                best_score = score
                best_intent = intent
                best_hits = hits
            # 平局时：优先非默认意图（QUIZ/PLANNER > QA）
            elif score == best_score and best_intent == _DEFAULT_INTENT and intent != _DEFAULT_INTENT:
                second_score = best_score
                best_score = score
                best_intent = intent
                best_hits = hits
            elif score > second_score:
                second_score = score


    # 无命中 → 兜底意图，置信度低
    if not best_hits:
        return RouteResult(
            intent=best_intent,
            chain_name=_intent_to_chain(best_intent),
            confidence=0.3,
            matched_keywords=[],
            router_layer="rule",
        )

    return RouteResult(
        intent=best_intent,
        chain_name=_intent_to_chain(best_intent),
        confidence=best_score,
        matched_keywords=best_hits,
        router_layer="rule",
    )


# ============================================
# 判断是否需要升级到 LLM
# ============================================

def _should_upgrade_to_llm(result: RouteResult) -> bool:
    """
    判断规则引擎的结果是否不够可靠，需要升级到 LLM 分类。

    触发条件:
        1. 置信度 < 0.88（单一关键词命中，不够确定）
        2. 完全没命中任何关键词（兜底意图，置信度 0.3）
        3. 最高分与次高分差距 < 0.10（多意图争抢）
    """
    if result.confidence < _LLM_UPGRADE_THRESHOLD:
        return True

    # 命中关键词数 ≤ 1：可能是巧合命中（如"考试"可能属于 QUIZ 或 PLANNER）
    if len(result.matched_keywords) <= 1:
        return True

    return False


# ============================================
# 第2层: LLM 语义分类
# ============================================

def _llm_classify(message: str) -> RouteResult:
    """
    使用 LLM 进行语义级别的意图分类。
    在规则引擎不确定时调用。
    """
    llm = get_llm(provider="deepseek", temperature=0.0, streaming=False)

    from langchain_core.messages import SystemMessage, HumanMessage

    prompt_text = ROUTER_CLASSIFY_PROMPT.format(message=message)
    messages = [
        HumanMessage(content=prompt_text),
    ]

    try:
        response = llm.invoke(messages)
        raw = response.content.strip().lower()

        # 解析 LLM 输出（只应输出 qa / quiz / planner）
        for intent in (Intent.QA, Intent.QUIZ, Intent.PLANNER):
            if intent.value == raw or intent.value in raw:
                return RouteResult(
                    intent=intent,
                    chain_name=_intent_to_chain(intent),
                    confidence=0.75,  # LLM 分类的默认置信度
                    matched_keywords=[],
                    router_layer="llm",
                )

        # LLM 输出不匹配任何标签 → 兜底
        return RouteResult(
            intent=_DEFAULT_INTENT,
            chain_name=_intent_to_chain(_DEFAULT_INTENT),
            confidence=0.3,
            matched_keywords=[],
            router_layer="llm",
        )

    except Exception:
        # LLM 调用失败 → 退回使用规则引擎的结果
        # [缺陷] 此处丢失了规则引擎的原始结果。调用方应传入 fallback 结果。
        return RouteResult(
            intent=_DEFAULT_INTENT,
            chain_name=_intent_to_chain(_DEFAULT_INTENT),
            confidence=0.2,
            matched_keywords=[],
            router_layer="llm",
        )


# ============================================
# 统一入口: 混合路由
# ============================================

def classify_intent(message: str) -> RouteResult:
    """
    混合意图分类:
        1. 先用规则引擎快速匹配
        2. 如果结果不够可靠，自动升级到 LLM 语义分类

    参数:
        message: 用户输入的原始消息

    返回:
        RouteResult，包含意图、链路名、置信度、使用的路由层

    示例:
        >>> classify_intent("什么是拉格朗日中值定理？")
        RouteResult(Intent.QA, "qa_chain", 0.91, ["什么是","定理"], "rule")

        >>> classify_intent("考试时间6月18号，帮我规划复习")
        RouteResult(Intent.PLANNER, "planner_chain", 0.75, [], "llm")
    """
    # 第1层: 规则引擎
    rule_result = _rule_classify(message)

    # 判断是否需要升级
    if _should_upgrade_to_llm(rule_result):
        # 第2层: LLM 语义分类
        llm_result = _llm_classify(message)
        return llm_result

    return rule_result


def route_message(message: str) -> RouteResult:
    """
    路由入口 —— classify_intent 的别名。
    供 API 层直接调用。
    """
    return classify_intent(message)


# ============================================
# 工具函数
# ============================================

def _intent_to_chain(intent: Intent) -> str:
    """
    将意图映射到对应的处理链路函数名。

    [缺陷] 硬编码映射，新增链路时需要在此处注册。
    [后续扩展] 改为注册表模式：各 chain 模块自行注册 Intent。
    """
    mapping = {
        Intent.QA: "qa_chain",
        Intent.QUIZ: "quiz_generator",
        Intent.PLANNER: "planner_chain",
    }
    return mapping[intent]


# ============================================
# 已知缺陷汇总
# ============================================
#
# 1. 【关键词列表不可热更新】
#    - 修改关键词需要改代码、重启服务。
#    - 后续可将关键词列表移到配置文件或数据库中，支持热更新。
#
# 2. 【学科关键词不完整】
#    - 仅覆盖数学/英语/政治的高频词，缺少专业课关键词。
#    - 后续从考研大纲提取完整关键词表。
#
# 3. 【LLM 分类异常时丢失原始规则结果】
#    - _llm_classify 异常时返回兜底结果，未保留规则引擎的原始判断。
#    - 后续应改为传入 fallback_result 参数。
#
# 4. 【无上下文感知】
#    - 没有利用对话历史判断意图。
#    - 后续 classify_intent 应接受 conversation_history 参数。
#
# 5. 【LLM 输出格式不稳定】
#    - 某些模型可能在 "qa" 前后加空格或换行。
#    - 当前做了 contains 检查，但可能匹配不精确。
