"""
考研伴学 Agent - 文档解析与切分工具
====================================
支持的格式：PDF / Markdown / TXT
切分策略：优先按 Markdown 标题层级切分（保留知识结构），
        文本层级不够时回退为字符递归切分。
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from app.core.config import settings


# ============================================
# 第一部分：文档加载
# ============================================

def load_document(file_path: str) -> List[Document]:
    """
    根据文件后缀自动选择合适的加载器，将文件加载为 LangChain Document 列表。

    参数:
        file_path: 文件路径（支持 .pdf / .markdown / .txt）

    返回:
        List[Document] —— 每个 Document 的 page_content 为提取后的文本

    异常:
        ValueError: 不支持的文件格式
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".pdf":
        loader = PyPDFLoader(str(path))
        # [缺陷] PyPDFLoader 只能提取文本层 PDF，无法处理扫描版（需要 OCR）。
        # [缺陷] 不提取表格、公式、图片，考研数学/理工类资料会丢失大量信息。
        # [后续扩展] 可替换为 Marker / Nougat / Mathpix 等专业工具。
        docs = loader.load()
        # 为每个 page 补上来源元数据
        for i, doc in enumerate(docs):
            doc.metadata["source"] = str(path)
            doc.metadata["page"] = i + 1
        return docs

    elif suffix == ".markdown":
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(path)
        return docs

    elif suffix == ".txt":
        loader = TextLoader(str(path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(path)
        return docs

    else:
        raise ValueError(
            f"不支持的文件格式: '{suffix}'。"
            f"当前支持: .pdf, .markdown, .txt"
            # [TODO: 后续可扩展 .docx（python-docx）、.html 等格式]
        )


# ============================================
# 第二部分：Markdown 标题层级切分
# ============================================

# 根据配置动态生成 LangChain 支持的标题分隔符
# MARKDOWN_HEADER_MAX_LEVEL=4 → ["#", "##", "###", "####"]
_HEADERS_TO_SPLIT_ON = [
    ("#" * i, f"Header_{i}")
    for i in range(1, settings.MARKDOWN_HEADER_MAX_LEVEL + 1)
]


def _build_header_splitter() -> MarkdownHeaderTextSplitter:
    """
    构建 Markdown 标题切分器。
    每次调用都从 settings 读取最新配置，保证灵活性。
    """
    # 只保留切分层级以下的标题（如 SPLIT_LEVEL=2，则只按 H1、H2 切分）
    effective_headers = _HEADERS_TO_SPLIT_ON[:settings.MARKDOWN_SPLIT_HEADER_LEVEL]

    return MarkdownHeaderTextSplitter(
        headers_to_split_on=effective_headers,
        # strip_headers=False 表示保留标题行原文（不删除）
        strip_headers=False,
    )


def split_by_markdown_headers(docs: List[Document]) -> List[Document]:
    """
    按 Markdown 标题层级切分文档。
    适用于已经有标题结构的教材笔记（.markdown 或从 PDF 转 Markdown 后的内容）。

    切分策略:
        1. 先用 MarkdownHeaderTextSplitter 按标题边界切分（保证不跨章节）。
        2. 如果某块仍超过 CHUNK_SIZE，再用 RecursiveCharacterTextSplitter 做二级切分。

    参数:
        docs: 待切分的 Document 列表

    返回:
        切分并打上元数据标签后的 Document 列表
    """
    header_splitter = _build_header_splitter()
    # 二级切分器 —— 当某个块仍然过长时使用
    chunk_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", ".", " "],
        # [缺陷] 当前分隔符仅适配中英文混合，未针对数学公式、代码块优化。
    )

    final_docs: List[Document] = []

    for doc in docs:
        # 步骤1: Markdown 标题切分
        try:
            header_chunks = header_splitter.split_text(doc.page_content)
        except Exception:
            # [缺陷] 如果文档没有标准 Markdown 标题结构，
            #   split_text 可能返回空列表。此时回退到纯字符切分。
            header_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata)]

        # 如果没有标题结构，直接把原文送去字符切分
        if not header_chunks:
            header_chunks = [Document(page_content=doc.page_content, metadata=doc.metadata)]

        # 步骤2: 对每个块做长度检查，超长再切
        for chunk_doc in header_chunks:
            chunk_text = chunk_doc.page_content.strip()
            if not chunk_text:
                continue

            # 合并元数据：原始文档元数据 + 标题切分产生的层级元数据
            merged_metadata = {**doc.metadata, **chunk_doc.metadata, "split_method": "header"}

            if len(chunk_text) <= settings.CHUNK_SIZE:
                final_docs.append(
                    Document(
                        page_content=chunk_text,
                        metadata=merged_metadata,
                    )
                )
            else:
                # 超长块：递归字符切分
                sub_chunks = chunk_splitter.split_text(chunk_text)
                for sub in sub_chunks:
                    sub = sub.strip()
                    if not sub:
                        continue
                    final_docs.append(
                        Document(
                            page_content=sub,
                            metadata={
                                **merged_metadata,
                                "split_method": "recursive",
                            },
                        )
                    )

    # [缺陷] 没有做去重，同一段落出现在不同位置时会产生冗余块。
    # [后续扩展] 可加入基于文本哈希的去重逻辑。
    return final_docs


def process_document(file_path: str) -> List[Document]:
    """
    一键流程: 加载 → 切分。
    这是文档处理的主要入口，外部只需调用此函数。

    参数:
        file_path: 文件路径

    返回:
        已切分的 Document 列表，可直接送入向量库
    """
    docs = load_document(file_path)
    return split_by_markdown_headers(docs)


# ============================================
# 当前代码缺陷总结 (2025-05-09)
# ============================================
#
# 1. 【PDF 解析能力弱】
#    - 仅支持文本层 PDF，无法处理扫描版（需要 OCR）。
#    - 无法提取表格、公式、图片，考研数学/理工科资料会丢失核心内容。
#    - 后续拆分方向：新增 document_parser.py 专门做 PDF -> Markdown，
#      集成 Marker / Nougat / Mathpix OCR 等高质量转换工具。
#
# 2. 【Markdown 标题切分依赖文档自身结构】
#    - 纯 TXT 或结构混乱的 PDF 没有标题层级，会直接退化为字符切分，
#      丢失语义边界。
#    - 后续拆分方向：text_splitter.py 独立出来，增加更多切分策略
#      （按语义段落、按题型、按知识点标签等）。
#
# 3. 【无去重机制】
#    - 同一份文档多次处理会产生重复向量块，浪费存储和检索精度。
#    - 后续可加入基于 SHA256 哈希的去重。
#
# 4. 【元数据不完整】
#    - 当前仅记录 source 和 page，缺少科目、年份、章节等标签，
#      导致后续无法按维度过滤检索。
#    - 后续 ingest_document 应接受 metadata 参数注入。
#
# 5. 【未支持 .docx / .html 等常见格式】
#    - 后续可在 load_document 中新增对应加载器。
