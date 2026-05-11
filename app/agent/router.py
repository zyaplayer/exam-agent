"""
考研伴学 Agent - 意图识别路由器
=================================
根据用户输入消息，判断意图并分发到对应的处理链路。
MVP 阶段使用关键词 + 规则匹配（零 LLM 调用成本）。
后续可升级为 LLM 语义分类以处理边界情况。
"""

from enum import Enum
from typing import NamedTuple


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


# ============================================
# 关键词规则表
# ============================================

# 格式：{ "意图": ([关键词列表], 权重) }
# 权重用于计算置信度：命中关键词数越多，置信度越高
# [缺陷] 当前权重固定为 1.0，未考虑关键词优先级或位置权重。
# [后续扩展] 可引入 TF-IDF 或简单词频统计，让权重更合理。

_KEYWORD_RULES: dict[Intent, tuple[list[str], float]] = {
    Intent.QA: (
        [
            # 概念/定理类
            "什么是", "解释", "为什么", "怎么样", "什么意思",
            "定理", "公式", "推导", "证明", "计算", "求",
            "定义", "概念", "性质", "区别", "联系",
            "怎么理解", "如何理解", "举例", "通俗",
            # 学科关键词（容易触发知识问答的）
            "极限", "导数", "积分", "矩阵", "特征值",
            "概率", "分布", "假设检验",
            "英语", "语法", "词汇", "长难句", "翻译",
            "政治", "马原", "毛概", "史纲", "思修",
            # [缺陷] 学科关键词列表不完整，仅覆盖了数学/英语/政治的高频词。
            # [后续扩展] 从考研大纲提取完整关键词表，或改用向量相似度匹配。
        ],
        0.85,  # 基础权重
    ),
    Intent.QUIZ: (
        [
            "做题", "出题", "题目", "真题", "练习题",
            "考试", "模拟", "试卷", "测一测", "来一道",
            "练", "刷题", "选择题", "填空题", "计算题",
            "错题", "正确率", "得分",
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
        ],
        0.85,
    ),
}

# 兜底意图 —— 所有关键词都不命中时走 QA
_DEFAULT_INTENT = Intent.QA


# ============================================
# 路由核心逻辑
# ============================================

def classify_intent(message: str) -> RouteResult:
    """
    根据用户消息进行意图分类。

    策略（MVP）:
        1. 对消息做简单预处理（小写/去多余空格）。
        2. 遍历所有意图的关键词表，统计命中数。
        3. 命中数最多的意图胜出，平局时按 Intent 优先级（QA > QUIZ > PLANNER）。
        4. 零命中时走兜底意图（QA）。

    参数:
        message: 用户输入的原始消息

    返回:
        RouteResult，包含意图、链路名、置信度和命中关键词

    示例:
        >>> classify_intent("什么是拉格朗日中值定理？")
        RouteResult(intent=Intent.QA, chain_name="qa_chain", confidence=0.85, ...)

        >>> classify_intent("给我出一道极限计算的题")
        RouteResult(intent=Intent.QUIZ, chain_name="quiz_generator", confidence=0.85, ...)
    """
    # 预处理
    normalized = message.strip()
    # [缺陷] 当前仅做 strip，没有做真正的大小写无关匹配。
    #   中文没有大小写问题所以影响不大，但后续如处理英文数学术语需改进。
    # [后续扩展] 增加繁简体转换、标点符号统一等预处理。

    # 计算每个意图的命中情况
    best_intent = _DEFAULT_INTENT
    best_hits: list[str] = []
    best_score = 0

    for intent, (keywords, base_weight) in _KEYWORD_RULES.items():
        hits = [kw for kw in keywords if kw in normalized]
        if hits:
            # 置信度 = 基础权重 + 命中数量加成（封顶 1.0）
            score = min(base_weight + len(hits) * 0.03, 1.0)
            if score > best_score:
                best_score = score
                best_intent = intent
                best_hits = hits
            elif score == best_score and best_intent == _DEFAULT_INTENT:
                # 平局时，优先非默认意图（QA 是默认意图，应让位给 QUIZ/PLANNER）
                best_intent = intent
                best_hits = hits

    # 组装结果
    route_result = RouteResult(
        intent=best_intent,
        chain_name=_intent_to_chain(best_intent),
        confidence=best_score if best_hits else 0.3,
        # [缺陷] 兜底意图的置信度硬编码为 0.3，没有理论依据。
        matched_keywords=best_hits,
    )

    return route_result


def _intent_to_chain(intent: Intent) -> str:
    """
    将意图映射到对应的处理链路函数名。
    这是路由表的核心映射逻辑。

    [缺陷] 当前是硬编码映射。后续新增链路时需要在两处修改：
       1) Intent 枚举中添加新值
       2) 此函数中新增映射
    [后续扩展] 可改为注册表模式：各 chain 模块自行注册自己的 Intent，
       此处自动发现，实现开闭原则。
    """
    mapping = {
        Intent.QA: "qa_chain",
        Intent.QUIZ: "quiz_generator",
        Intent.PLANNER: "planner_chain",
    }
    return mapping[intent]


def route_message(message: str) -> RouteResult:
    """
    路由入口 —— classify_intent 的别名。
    供 API 层直接调用。

    [缺陷] 当前同步函数。如果未来某个链路需要异步初始化
      （如预热模型或加载向量库），需改为 async。
    """
    return classify_intent(message)


# ============================================
# 已知缺陷汇总
# ============================================
#
# 1. 【关键词匹配粗粒度】
#    - 短消息或含义模糊的消息容易误分类。
#      例如"极限怎么求"和"极限挑战的题来一道"后者应该属于 QUIZ 但可能命中 QA。
#    - 后续加入否定词检测（如"不是要出题"），减少误触。
#
# 2. 【无多意图识别】
#    - 用户一句话可能同时包含问答+规划需求（如"概率论怎么复习，先看哪章"）。
#    - 当前只返回单一意图，后半段需求被忽略。
#    - 后续可改为多标签分类，返回 [QA, PLANNER] 并引导用户澄清。
#
# 3. 【无上下文感知】
#    - 没有利用对话历史来判断意图。
#      用户上一轮在问"极限的定义"，这一轮说"再解释一下"，
#      当前无法推断出仍在 QA 链路。
#    - 后续 router 应接受 conversation_history 参数。
#
# 4. 【关键词列表不可热更新】
#    - 修改关键词需要改代码、重启服务。
#    - 后续可将关键词列表移到配置文件或数据库中，支持热更新。
#
# 5. 【无 LLM 兜底机制】
#    - 当关键词匹配置信度极低（如 < 0.5）时，没有调用 LLM 做二次判断。
#    - 后续实现混合策略：关键词优先 → 低置信度时调 LLM 分类。
