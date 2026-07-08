"""
考研伴学 Agent - 教材大纲提取器
================================
从上传的教材中自动识别目录页，提取结构化章节目录，
并保存为 Markdown 大纲文件到 data/summary_db/。
"""

import re
import json
from pathlib import Path

from langchain_core.documents import Document

from app.core.config import settings


def _has_toc_pattern(text: str) -> bool:
    """
    判断文本是否包含目录页特征。

    特征:
      1. 包含 "目录" 或 "CONTENTS" 标题
      2. 包含大量点线+页码模式（如 "极限的定义 ........... 42"）
      3. 连续多行都是短文本+页码结尾
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return False

    # 特征1: "目录"标题
    if "目录" in "".join(lines[:3]) or "CONTENTS" in "".join(lines[:3]).upper():
        return True

    # 特征2: 点线+页码模式
    dot_page_pattern = re.compile(r"\.{3,}\s*\d+$|…+\s*\d+$")
    dot_lines = [l for l in lines if dot_page_pattern.search(l)]
    if len(dot_lines) >= 3:
        return True

    return False


def extract_outline_from_docs(docs: list[Document]) -> list[dict]:
    """
    从文档列表中识别目录页并提取章节目录结构。

    返回格式:
        [
            {"title": "第一章 函数与极限", "page": 1, "level": 1},
            {"title": "1.1 函数的概念", "page": 3, "level": 2},
            {"title": "1.2 数列的极限", "page": 8, "level": 2},
            ...
        ]
    """
    # 第一步：找到目录页
    toc_texts = []
    for doc in docs:
        if _has_toc_pattern(doc.page_content):
            toc_texts.append(doc.page_content)

    if not toc_texts:
        return []

    full_toc = "\n".join(toc_texts)

    # 第二步：提取目录条目
    # 模式: 标题文本 ... 页码
    #   "第一章 极限与连续............ 1"
    #   "1.1 极限的定义............... 3"
    toc_line_pattern = re.compile(
        r"^(.+?)\s*(?:\.{2,}|…+)\s*(\d+)\s*$",
        re.MULTILINE,
    )

    outline = []
    for match in toc_line_pattern.finditer(full_toc):
        title = match.group(1).strip()
        page = int(match.group(2))

        # 判断层级: 第一章/第X章 → level 1, X.X → level 2, X.X.X → level 3
        if re.match(r"^第[一二三四五六七八九十\d]+章", title):
            level = 1
        elif re.match(r"^\d+\.\d+\.\d+", title):
            level = 3
        elif re.match(r"^\d+\.\d+", title):
            level = 2
        elif not any(c.isdigit() for c in title[:3]):
            level = 1  # 附录、前言等非编号章节
        else:
            level = 1

        outline.append({"title": title, "page": page, "level": level})

    return outline


def save_outline_to_summary(
    outline: list[dict],
    collection_name: str,
) -> str | None:
    """
    将提取的章节目录保存为 Markdown 大纲文件到 data/summary_db/。

    参数:
        outline:          extract_outline_from_docs 的返回结果
        collection_name:  教材 Collection 名称（用作文件名）

    返回:
        保存的文件路径，无目录内容时返回 None
    """
    if not outline:
        return None

    summary_dir = settings.BASE_DIR / settings.SUMMARY_DB_DIR.lstrip("./")
    summary_dir.mkdir(parents=True, exist_ok=True)

    lines = [f"# {collection_name} 章节目录\n"]
    for item in outline:
        prefix = "#" * min(item["level"] + 1, 4)
        lines.append(f"{prefix} {item['title']}（第{item['page']}页）")

    file_path = summary_dir / f"{collection_name}.markdown"
    file_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(file_path)


def process_and_save_outline(
    docs: list[Document],
    collection_name: str,
) -> tuple[list[dict], str | None]:
    """
    一键流程：识别目录 → 提取大纲 → 保存到 summary_db/。

    返回: (outline列表, 保存路径或None)
    """
    outline = extract_outline_from_docs(docs)
    saved_path = save_outline_to_summary(outline, collection_name)
    return outline, saved_path
