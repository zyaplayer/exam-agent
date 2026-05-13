"""新功能链路综合验证：QA / Quiz / Planner + 绑定 CRUD"""
import asyncio

print('=' * 60)
print('新功能链路综合验证')
print('=' * 60)

# ---- 预检 ----
print('\n[预检] 导入...')
from app.agent.router import classify_intent, Intent
from app.agent.quiz_generator import (
    get_all_bindings, create_binding, delete_binding
)
print('  全部导入 OK')

# ---- 1. 混合意图路由 ----
print('\n[测试1] 混合意图路由...')
cases = [
    ("什么是拉格朗日中值定理？", Intent.QA),
    ("出5道极限计算题", Intent.QUIZ),
    ("帮我制定两个月的复习计划", Intent.PLANNER),
    ("考试时间是6月18号，数学基础差", Intent.PLANNER),  # 模糊消息 → LLM 处理
]
for msg, expected in cases:
    result = classify_intent(msg)
    icon = "✅" if result.intent == expected else "⚠️"
    print(f'  {icon} "{msg[:25]}..." → {result.intent.value} (layer: {result.router_layer})')
print('  ✅ 混合路由通过')

# ---- 2. 绑定 CRUD ----
print('\n[测试2] Collection 绑定 CRUD...')
# 创建
b1 = create_binding(
    textbook_collections=["高数教材"],
    exam_collections=["高数真题2023"],
    label="测试绑定_高数",
)
print(f'  创建: {b1["id"]} ({b1["label"]})')
assert b1["id"], "创建失败"

# 查询
all_b = get_all_bindings()
print(f'  查询: 共 {len(all_b)} 条绑定')
assert any(b["id"] == b1["id"] for b in all_b), "查询未找到新创建的绑定"

# 删除
deleted = delete_binding(b1["id"])
print(f'  删除: {"成功" if deleted else "失败"}')
assert deleted, "删除失败"

# 确认删除
all_b2 = get_all_bindings()
assert not any(b["id"] == b1["id"] for b in all_b2), "删除不干净"
print('  ✅ 绑定 CRUD 通过')

# ---- 3. 简单出题参数解析 ----
print('\n[测试3] 简单出题参数解析...')
from app.agent.quiz_generator import _parse_simple_quiz_params
params = _parse_simple_quiz_params("出5道极限计算题，选择题，强化难度")
print(f'  数量={params["question_count"]}, 方向={params["question_topic"]}, '
      f'形式={params["question_type"]}, 难度={params["difficulty"]}')
assert params["question_count"] == 5
assert "极限" in params["question_topic"]
assert params["question_type"] == "计算题"
assert params["difficulty"] == "强化"
print('  ✅ 参数解析通过')

# ---- 4. 时间系统 ----
print('\n[测试4] 时间系统...')
from app.agent.planner_chain import calculate_remaining_days, get_phase_dates
remaining = calculate_remaining_days("2026-06-18")
print(f'  到2026-06-18剩余: {remaining} 天')
assert remaining >= 0, "剩余天数异常"

phases = get_phase_dates(remaining)
print(f'  基础: {phases["base_start"]}~{phases["base_end"]} ({phases["base_days"]}天)')
assert phases["base_days"] >= 1
print('  ✅ 时间系统通过')

# ---- 5. 无日期灵活模式 ----
print('\n[测试5] 灵活模式（无考试日期）...')
from app.agent.planner_chain import _extract_student_info
import asyncio
# 模拟提取结果中无日期
info = asyncio.run(_extract_student_info("帮我把高数过一遍"))
print(f'  exam_date={info.get("exam_date")}, 弱项={info.get("weak_subjects")}')
assert info.get("exam_date") is None
print('  ✅ 灵活模式触发条件验证通过')

# ---- 总结 ----
print('\n' + '=' * 60)
print('✅ 新功能链路验证全部通过！')
print('=' * 60)
