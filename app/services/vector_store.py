"""
考研伴学 Agent - ChromaDB 向量库服务
=====================================
提供向量库的初始化、文档入库、相似度检索等操作。
封装了 ChromaDB 的底层细节，上层 Agent 只需调用这里的函数。
"""

import chromadb
from pathlib import Path
from typing import List, Optional

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.llm_manager import get_embeddings



# ---- ChromaDB 全局单例客户端（避免重复初始化冲突） ----
import threading
_chroma_clients: dict = {}
_chroma_lock = threading.Lock()
# [注] chromadb 1.5+ 不允许同一目录创建多个不同设置的 PersistentClient，
#   因此必须全局共享同一个客户端实例。

# ============================================
# 第一部分：向量库连接


def _get_embedding_function():
    """
    获取 Embedding 函数（全局单例，懒加载）。
    使用本地 HuggingFace 模型，无需 API Key。
    """
    return get_embeddings()

def _get_persistent_client() -> chromadb.PersistentClient:
    """
    获取线程安全的 ChromaDB 持久化客户端单例。
    所有需要直接操作 ChromaDB 的函数都应通过此函数获取客户端。
    """
    persist_dir = str(settings.BASE_DIR / settings.CHROMA_PERSIST_DIR.lstrip("./"))
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    with _chroma_lock:
        if persist_dir not in _chroma_clients:
            _chroma_clients[persist_dir] = chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    is_persistent=True,
                ),
            )
        return _chroma_clients[persist_dir]
    
# ============================================
# Collection 名称转码
# ============================================

# 中文名称 → 有效 ChromaDB 名称的映射文件
_ALIASES_FILE = settings.BASE_DIR / "data" / "collection_aliases.json"


def _sanitize_collection_name(name: str) -> str:
    """
    将用户输入的名称转为 ChromaDB 允许的格式（[a-zA-Z0-9._-]）。

    规则:
      1. 纯英文/数字 → 直接使用
      2. 含有中文/特殊字符 → 保留原文字符为前缀取前8字 + 短哈希
         例如 "高数教材" → "gsc_7f3a"

    同时将映射关系存入 aliases.json，供前端展示原始名称。
    """
    import re
    import hashlib
    import json

    # 检查名称是否已合法
    if re.match(r"^[a-zA-Z0-9][a-zA-Z0-9._-]*[a-zA-Z0-9]$", name) and len(name) >= 3:
        return name

    # 不合法的名称 → 生成别名
    aliases = {}
    if _ALIASES_FILE.exists():
        try:
            aliases = json.loads(_ALIASES_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass

    # 如果之前已为此中文名生成过映射，直接返回
    if name in aliases:
        return aliases[name]

    # 提取中文拼音首字母作为前缀
    import unicodedata
    safe_prefix = re.sub(r"[^a-zA-Z0-9]", "", name)[:8] or "col"
    
    # 后缀用哈希前6位确保唯一
    name_hash = hashlib.md5(name.encode()).hexdigest()[:6]
    
    safe_name = f"{safe_prefix}_{name_hash}"

    # 保存映射
    aliases[name] = safe_name
    _ALIASES_FILE.parent.mkdir(parents=True, exist_ok=True)
    _ALIASES_FILE.write_text(
        json.dumps(aliases, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return safe_name



def get_vector_store(
    collection_name: str = "default",
) -> Chroma:
    """
    获取或创建一个 ChromaDB Collection 对应的 LangChain Chroma 包装对象。

    参数:
        collection_name: Collection 名称。
                        不同的教材/科目可以使用不同的 Collection 隔离数据。

    返回:
        langchain_chroma.Chroma 实例，可直接调用 add_documents / similarity_search
    """
    collection_name = _sanitize_collection_name(collection_name)
    shared_client = _get_persistent_client()

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embedding_function(),
        client=shared_client,
    )


    return vector_store


# ============================================
# 第二部分：文档入库
# ============================================

def add_documents(
    docs: List[Document],
    collection_name: str = "default",
) -> int:
    """
    将一批 Document 向量化后存入指定的 ChromaDB Collection。

    参数:
        docs:            待入库的 Document 列表
        collection_name: 目标 Collection 名称

    返回:
        实际入库的文档数量

    注意:
        如果同名文档已存在，当前版本不做去重，会产生冗余向量。
        [后续扩展] 入库前先查重（按 source + page 或文本哈希判断）。
    """
    if not docs:
        return 0

    collection_name = _sanitize_collection_name(collection_name)

    # ---- 文本哈希去重 ----
    import hashlib

    vector_store = get_vector_store(collection_name)

    # 获取已有文档的内容哈希集合
    existing_count = vector_store._collection.count()
    if existing_count > 0:
        existing_data = vector_store._collection.get(
            include=["metadatas"]
        )
        existing_hashes = {
            m.get("content_hash", "")
            for m in (existing_data.get("metadatas") or [])
            if m
        }
    else:
        existing_hashes = set()

    # 过滤掉重复的文档块
    new_docs = []
    for doc in docs:
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        if content_hash in existing_hashes:
            continue  # 重复块，跳过
        doc.metadata["content_hash"] = content_hash
        new_docs.append(doc)

    if not new_docs:
        return 0  # 全部重复，无需入库

    # 入库新文档块
    ids = [f"doc_{existing_count + i}" for i in range(len(new_docs))]
    vector_store.add_documents(documents=new_docs, ids=ids)

    return len(new_docs)



# ============================================
# 第三部分：相似度检索
# ============================================

def search_documents(
    query: str,
    collection_name: str = "default",
    k: Optional[int] = None,
    metadata_filter: Optional[dict] = None,
) -> List[Document]:
    """
    在指定 Collection 中进行相似度检索，返回最相关的文档块。

    参数:
        query:           用户查询文本
        collection_name: 目标 Collection 名称
        k:               返回的文档数量（默认取配置中的 RETRIEVAL_K）
        metadata_filter: 元数据过滤条件，如 {"subject": "高数", "year": "2023"}
                         为 None 时不过滤。

    返回:
        相关度降序排列的 Document 列表（每个包含 page_content 和 metadata）
    """
    if k is None:
        k = settings.RETRIEVAL_K

    collection_name = _sanitize_collection_name(collection_name)

    vector_store = get_vector_store(collection_name)

    if vector_store._collection.count() == 0:
        return []

    # 构建 ChromaDB 支持的 where 过滤条件
    search_kwargs = {"k": k}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter

    results_with_scores = vector_store.similarity_search_with_score(
        query=query,
        **search_kwargs,
    )

    threshold = settings.RETRIEVAL_THRESHOLD
    filtered_docs: List[Document] = []
    for doc, score in results_with_scores:
        if threshold <= 0.0 or score <= threshold:
            doc.metadata["relevance_score"] = round(score, 4)
            filtered_docs.append(doc)

    return filtered_docs



def hybrid_search(
    query: str,
    collection_name: str = "default",
    k: Optional[int] = None,
    vector_weight: float = 0.6,
    metadata_filter: Optional[dict] = None,
) -> List[Document]:
    """
    混合检索：稠密向量 + BM25 关键词，加权融合排序。

    参数:
        query:           用户查询文本
        collection_name: 目标 Collection 名称
        k:               返回的文档数量
        vector_weight:   向量分数权重（0~1），剩余为 BM25 权重。
                         0.6 = 偏语义，0.3 = 偏关键词精确匹配。
        metadata_filter: 元数据过滤条件

    返回:
        加权融合后排序的 Document 列表
    """
    from rank_bm25 import BM25Okapi
    import jieba  # type: ignore  # 无类型存根，运行时正常

    if k is None:
        k = settings.RETRIEVAL_K

    vector_store = get_vector_store(collection_name)

    collection_count = vector_store._collection.count()
    if collection_count == 0:
        return []

    # ---- 步骤1: 获取全量文档用于 BM25 ----
    all_data = vector_store._collection.get(include=["documents", "metadatas"])
    all_docs = all_data.get("documents") or []
    all_metadatas = all_data.get("metadatas") or []
    if not all_docs:
        return []

    # 中文分词
    tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
    tokenized_query = list(jieba.cut(query))

    # ---- 步骤2: BM25 关键词检索 ----
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_max = max(bm25_scores) if bm25_scores else 1
    bm25_normalized = [s / bm25_max if bm25_max else 0 for s in bm25_scores]

    # ---- 步骤3: 向量检索 ----
    search_kwargs = {"k": collection_count}
    if metadata_filter:
        search_kwargs["filter"] = metadata_filter
    results_with_scores = vector_store.similarity_search_with_score(query=query, **search_kwargs)
    vectors_by_id = {}
    for doc, score in results_with_scores:
        doc_id = doc.metadata.get("content_hash", doc.page_content[:50])
        vectors_by_id[doc_id] = score

    # ---- 步骤4: 加权融合 ----
    final_scores: list[tuple[Document, float]] = []
    for i, doc_text in enumerate(all_docs):
        doc = Document(
            page_content=doc_text,
            metadata=all_metadatas[i] if i < len(all_metadatas) else {},
        )
        content_hash = doc.metadata.get("content_hash", doc.page_content[:50])
        vec_score_raw = vectors_by_id.get(content_hash, 2.0)
        vec_score = max(0, 1 - vec_score_raw / 2.0)
        keyword_score = bm25_normalized[i]
        combined = vector_weight * vec_score + (1 - vector_weight) * keyword_score
        doc.metadata["keyword_score"] = round(keyword_score, 4)
        doc.metadata["vector_score"] = round(vec_score, 4)
        doc.metadata["combined_score"] = round(combined, 4)
        final_scores.append((doc, combined))

    # 去重 + 按融合分数排序
    seen = set()
    deduped: list[Document] = []
    final_scores.sort(key=lambda x: x[1], reverse=True)
    for doc, score in final_scores:
        content_hash = doc.metadata.get("content_hash", "")
        if content_hash in seen:
            continue
        seen.add(content_hash)
        if settings.RETRIEVAL_THRESHOLD <= 0.0 or score >= (1 - settings.RETRIEVAL_THRESHOLD / 2.0):
            doc.metadata["relevance_score"] = round(score, 4)
            deduped.append(doc)
            if len(deduped) >= k:
                break

    return deduped



# ============================================
# 第四部分：Collection 管理
# ============================================

def list_collections() -> List[str]:
    """
    列出当前 ChromaDB 中所有的 Collection 名称（返回原始中文名）。

    返回:
        Collection 名称列表（已从内部安全名反向映射为原始名称）
    """
    import json
    client = _get_persistent_client()
    collections = client.list_collections()
    safe_names = [col.name for col in collections]

    # 尝试从 aliases.json 反向映射为原始中文名称
    reverse_aliases: dict[str, str] = {}
    if _ALIASES_FILE.exists():
        try:
            aliases = json.loads(_ALIASES_FILE.read_text(encoding="utf-8"))
            reverse_aliases = {v: k for k, v in aliases.items()}
        except Exception:
            pass

    display_names = [reverse_aliases.get(name, name) for name in safe_names]
    return display_names



def delete_collection(collection_name: str) -> bool:
    """
    删除指定的 Collection。

    参数:
        collection_name: 要删除的 Collection 名称

    返回:
        是否成功删除
    """

    collection_name = _sanitize_collection_name(collection_name)
    client = _get_persistent_client()


    try:
        client.delete_collection(collection_name)
        # 同时清空缓存中的 embedding 引用（如果有的话）
        return True
    except Exception:
        return False


# ============================================
# 第五部分：一键入库便捷函数
# ============================================

def ingest_document(
    file_path: str,
    collection_name: str = "default",
) -> int:
    """
    协调“文档解析 → 切分 → 向量化 → 入库”的完整流程。
    这是外部调用的主要入口，一行代码完成文档入库。

    参数:
        file_path:       待入库的文件路径
        collection_name: 目标 Collection 名称

    返回:
        入库的文档块数量

    示例:
        >>> n = ingest_document("./data/raw_docs/线性代数.pdf")
        >>> print(f"已入库 {n} 个知识块")
    """
    from app.utils.document import process_document

    collection_name = _sanitize_collection_name(collection_name)

    # 步骤1+2: 加载并切分文档
    chunks = process_document(file_path)
    if not chunks:
        raise RuntimeError(f"文档处理结果为空: {file_path}")

    # 步骤3: 向量化并存入 ChromaDB
    count = add_documents(chunks, collection_name)

    return count


# ============================================
# 当前代码缺陷总结 (2025-05-09)
# ============================================
#
# 1. 【无连接池 / 连接复用】
#    - 每次调用 get_vector_store 都重新初始化 Chroma 客户端连接。
#    - 高并发场景下性能差，且可能触发文件锁冲突。
#    - 后续拆分方向：单例 Chroma 客户端 + 线程安全的 Collection 缓存。
#
# 2. 【入库无去重】
#    - 同一份文档重复上传会产生冗余向量块，浪费磁盘并降低检索精度。
#    - 后续应在 add_documents 前按 (source, page, chunk_hash) 过滤已有记录。
#
# 3. 【检索无分数过滤】
#    - 当前使用 similarity_search，不返回相似度分数。
#    - 无法滤除低相关度的噪声结果（如相似度 < 0.5 的块）。
#    - 后续改用 similarity_search_with_score + RETRIEVAL_THRESHOLD。
#
# 4. 【Embedding Provider 硬编码】
#    - _get_embedding_function 写死了 "deepseek"，无法在 Collection 级别切换。
#    - 后续应支持每个 Collection 绑定不同 embedding provider。
#
# 5. 【无批量入库优化】
#    - 大批量文档依次入库，没有并发/批处理。
#    - ChromaDB 支持 batch add，后续可加入分批次 + 进度回调。
#
# 6. 【不支持文档删除 / 更新】
#    - 只能整库删除，无法按 source 删除单份文档的向量。
#    - 后续应提供 delete_by_source() 和 update_document() 方法。
#
# 7. 【缺少数据校验】
#    - ingest_document 对 PDF 页数、文件大小没有上限检查。
#    - 后续应加入文件大小限制 + 空文件检测。
