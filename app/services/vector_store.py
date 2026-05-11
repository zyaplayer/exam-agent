"""
考研伴学 Agent - ChromaDB 向量库服务
=====================================
提供向量库的初始化、文档入库、相似度检索等操作。
封装了 ChromaDB 的底层细节，上层 Agent 只需调用这里的函数。
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings

from app.core.config import settings
from app.core.llm_manager import get_embeddings


# ============================================
# 第一部分：向量库连接


def _get_embedding_function():
    """
    获取 Embedding 函数（全局单例，懒加载）。
    使用本地 HuggingFace 模型，无需 API Key。
    """
    return get_embeddings()


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
    # 确保持久化目录存在
    persist_dir = str(settings.BASE_DIR / settings.CHROMA_PERSIST_DIR.lstrip("./"))
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=_get_embedding_function(),
        persist_directory=persist_dir,
        # ChromaDB 客户端设置
        client_settings=ChromaSettings(
            anonymized_telemetry=False,  # 关闭遥测
            is_persistent=True,
        ),
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

    vector_store = get_vector_store(collection_name)

    # Chroma.add_documents 会自动调用 embedding_function 做向量化
    # ids 使用自增，保证每条记录可唯一定位
    existing_count = vector_store._collection.count()
    ids = [f"doc_{existing_count + i}" for i in range(len(docs))]

    vector_store.add_documents(documents=docs, ids=ids)

    # [缺陷] 每次 add_documents 都会重新初始化 Chroma 连接，
    #   大批量入库时效率低。后续应改为连接池或上下文管理器。

    return len(docs)


# ============================================
# 第三部分：相似度检索
# ============================================

def search_documents(
    query: str,
    collection_name: str = "default",
    k: Optional[int] = None,
) -> List[Document]:
    """
    在指定 Collection 中进行相似度检索，返回最相关的文档块。

    参数:
        query:           用户查询文本
        collection_name: 目标 Collection 名称
        k:               返回的文档数量（默认取配置中的 RETRIEVAL_K）

    返回:
        相关度降序排列的 Document 列表（每个包含 page_content 和 metadata）
    """
    if k is None:
        k = settings.RETRIEVAL_K

    vector_store = get_vector_store(collection_name)

    # 检查 Collection 是否为空
    if vector_store._collection.count() == 0:
        return []

    results = vector_store.similarity_search(
        query=query,
        k=k,
    )

    # [缺陷] similarity_search 不支持 score_threshold 过滤（如仅返回 >0.7 的结果），
    #   可能返回完全不相关的向量。可通过 similarity_search_with_score 获取分数后自行过滤。
    # [后续扩展] 改用 similarity_search_with_score 并加入阈值过滤。

    return results


# ============================================
# 第四部分：Collection 管理
# ============================================

def list_collections() -> List[str]:
    """
    列出当前 ChromaDB 中所有的 Collection 名称。
    可用于管理界面展示或 API 调用。

    返回:
        Collection 名称列表
    """
    import chromadb

    persist_dir = str(settings.BASE_DIR / settings.CHROMA_PERSIST_DIR.lstrip("./"))
    client = chromadb.PersistentClient(path=persist_dir)
    collections = client.list_collections()
    return [col.name for col in collections]


def delete_collection(collection_name: str) -> bool:
    """
    删除指定的 Collection。

    参数:
        collection_name: 要删除的 Collection 名称

    返回:
        是否成功删除
    """
    import chromadb

    persist_dir = str(settings.BASE_DIR / settings.CHROMA_PERSIST_DIR.lstrip("./"))
    client = chromadb.PersistentClient(path=persist_dir)

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
