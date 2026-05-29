"""
考研伴学 Agent - Rerank 重排序服务
====================================
使用 CrossEncoder 对候选文档精排，提升检索精度。
"""

from typing import List
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document

_RERANKER_INSTANCE: CrossEncoder | None = None
_MODEL_NAME = "BAAI/bge-reranker-base"  # ~300MB，更轻更快，中文效果也很好


def _get_reranker() -> CrossEncoder:
    global _RERANKER_INSTANCE
    if _RERANKER_INSTANCE is None:
        print(f"[Reranker] 正在加载 {_MODEL_NAME} ...")
        _RERANKER_INSTANCE = CrossEncoder(_MODEL_NAME)
        print("[Reranker] 加载完成")
    return _RERANKER_INSTANCE


def rerank(
    query: str,
    docs: List[Document],
    top_k: int = 4,
    min_score: float = 0.0,
) -> List[Document]:
    """
    使用 CrossEncoder 对候选文档列表精细重排序。

    参数:
        query:     用户查询
        docs:      候选文档列表（通常为粗筛后的 Top-20）
        top_k:     返回的最相关文档数
        min_score: Rerank 分数阈值（低于此值丢弃，0.0 表示不过滤）

    返回:
        按 Rerank 分数降序排列的 Top-K 文档（分数存入 metadata["rerank_score"]）

    注意:
        - Reranker 是逐对比较的，时间复杂度 O(n)。
        - 候选文档数建议控制在 20-50 个以保证延迟可控。
    """
    if not docs:
        return docs

    try:
        model = _get_reranker()
    except Exception as e:
        print(f"[Reranker] 模型加载失败，跳过精排: {e}")
        return docs[:top_k]

    pairs = [(query, doc.page_content) for doc in docs]
    scores = model.predict(pairs)

    # 将分数附加到 metadata
    for doc, score in zip(docs, scores):
        doc.metadata["rerank_score"] = round(float(score), 4)

    # 排序（分数越高越相关）
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    # 过滤 + 截断
    result = []
    for doc, score in scored:
        if min_score > 0 and float(score) < min_score:
            continue
        doc.metadata["rerank_score"] = round(float(score), 4)
        result.append(doc)
        if len(result) >= top_k:
            break

    return result
