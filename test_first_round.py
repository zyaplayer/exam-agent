"""第一轮 6 项修复综合验证"""
import asyncio

print('=' * 60)
print('第一轮 6 项修复综合验证')
print('=' * 60)

# ---- 预检 ----
print('\n[预检] 模块导入...')
from app.core.config import settings
from app.core.llm_manager import get_embeddings, get_llm
from app.services.vector_store import search_documents, list_collections
from app.agent.qa_chain import ask_question, ask_question_stream, _format_retrieved_context
from app.api.routes import router
print('  全部模块导入 OK')

# ---- 修复④: H3 切分粒度 ----
print(f'\n[修复4] H3 切分粒度: MARKDOWN_SPLIT_HEADER_LEVEL = {settings.MARKDOWN_SPLIT_HEADER_LEVEL}')
assert settings.MARKDOWN_SPLIT_HEADER_LEVEL >= 3, '切分层级应 >= 3'
print('  ✅ 通过')

# ---- 修复③: 相似度分数过滤 ----
print(f'\n[修复3] 相似度阈值: RETRIEVAL_THRESHOLD = {settings.RETRIEVAL_THRESHOLD}')
assert settings.RETRIEVAL_THRESHOLD > 0, '阈值应 > 0（已启用过滤）'
print('  ✅ 通过')

# ---- 修复⑥: 健康检查 ----
print('\n[修复6] 健康检查深度探测...')
from fastapi.testclient import TestClient
from main import app
client = TestClient(app)

resp = client.get("/api/health")
data = resp.json()
print(f'  浅度检查: status={data.get("status")}')
assert data["status"] == "ok", "浅度检查应返回 ok"

resp = client.get("/api/health?deep=true")
data = resp.json()
print(f'  深度检查: status={data.get("status")}')
checks = data.get("checks", {})
for component, result in checks.items():
    icon = "[OK]" if result == "ok" else "[FAIL]"
    print(f'    {icon} {component}: {result}')
assert "checks" in data, "deep=true 应返回 checks 字段"
print('  ✅ 通过')

# ---- 修复②: 层级元数据注入 ----
print('\n[修复2] 层级元数据注入...')
from langchain_core.documents import Document

test_doc = Document(
    page_content="极限是微积分中最基本的概念之一。",
    metadata={
        "source": "高数笔记.markdown",
        "page": "3",
        "Header_1": "高等数学",
        "Header_2": "极限与连续",
        "Header_3": "极限的定义",
    }
)
formatted = _format_retrieved_context([test_doc])
has_hierarchy = "章节: 高等数学" in formatted
print(f'  上下文包含章节层级: {has_hierarchy}')
if has_hierarchy:
    print('    ✅ 章节信息已注入上下文')
else:
    print('    ❌ 章节信息未出现在上下文中')
assert has_hierarchy, "层级元数据未被注入"
print('  ✅ 通过')

# ---- 修复①: 空知识库拒绝回答 ----
print('\n[修复1] 空知识库拒绝回答...')
empty_result = ask_question("测试问题", collection_name="default")
is_reject = "暂无相关内容" in empty_result or "无法回答" in empty_result
print(f'  拒绝提示: {empty_result[:80]}...')
assert is_reject, f"空知识库应拒绝回答，实际返回: {empty_result[:100]}"
print('  ✅ 通过')

# ---- 修复⑤: 异步检索 ----
print('\n[修复5] 异步检索...')

async def test_async():
    try:
        chunks = []
        async for chunk in ask_question_stream("测试",collection_name="default"):
            chunks.append(chunk)
        full_text = "".join(chunks)
        return "暂无相关内容" in full_text
    except Exception as e:
        print(f'  ❌ 异步检索异常: {e}')
        return False

result = asyncio.run(test_async())
assert result, "异步流式检索未正常工作"
print('  ✅ 通过')

# ---- 总结 ----
print('\n' + '=' * 60)
print('✅ 全部 6 项修复验证通过！')
print('=' * 60)
