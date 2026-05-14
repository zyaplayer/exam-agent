"""
考研伴学 Agent - 对话历史服务
===============================
管理多轮对话历史的存储、追加、截断和持久化。
支持多个独立对话（不同场景互不污染）。
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from app.core.config import settings


# 对话存储目录
_CONV_DIR = settings.BASE_DIR / "data" / "conversations"


def _ensure_dir():
    _CONV_DIR.mkdir(parents=True, exist_ok=True)


def new_conversation_id() -> str:
    """生成一个新的对话ID"""
    return f"conv_{uuid.uuid4().hex[:12]}"


def get_history(conversation_id: str) -> List[dict]:
    """
    获取指定对话的完整历史。

    返回格式:
        [
            {"role": "user", "content": "...", "time": "2026-05-13T10:00:00"},
            {"role": "assistant", "content": "...", "time": "2026-05-13T10:00:05"},
            ...
        ]
    """
    file_path = _CONV_DIR / f"{conversation_id}.json"
    if not file_path.exists():
        return []
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def append_message(conversation_id: str, role: str, content: str):
    """
    向对话历史追加一条消息。

    参数:
        conversation_id: 对话ID
        role:            "user" 或 "assistant"
        content:         消息内容
    """
    _ensure_dir()
    history = get_history(conversation_id)
    history.append({
        "role": role,
        "content": content,
        "time": datetime.now().isoformat(),
    })
    file_path = _CONV_DIR / f"{conversation_id}.json"
    file_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def format_history_for_prompt(
    conversation_id: str,
    max_turns: Optional[int] = None,
) -> str:
    """
    将对话历史格式化为 Prompt 可用的文本。

    参数:
        conversation_id: 对话ID
        max_turns:       最大保留轮数（一轮 = 一次问答对），默认取配置

    返回:
        格式化后的历史文本。无历史时返回空字符串。
    """
    if max_turns is None:
        max_turns = settings.MAX_CONVERSATION_TURNS

    history = get_history(conversation_id)
    if not history:
        return ""

    # 保留最后 2*max_turns 条消息（一轮 = user + assistant）
    recent = history[-(max_turns * 2):]

    lines = ["## 对话历史（供上下文参考）"]
    for msg in recent:
        role_label = "学生" if msg["role"] == "user" else "助教"
        lines.append(f"{role_label}: {msg['content']}")

    return "\n".join(lines)


def delete_conversation(conversation_id: str) -> bool:
    """删除指定对话的历史文件"""
    file_path = _CONV_DIR / f"{conversation_id}.json"
    if file_path.exists():
        file_path.unlink()
        return True
    return False


def list_conversations() -> List[dict]:
    """列出所有对话（用于前端展示对话列表）"""
    _ensure_dir()
    result = []
    for f in sorted(_CONV_DIR.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        conv_id = f.stem
        history = get_history(conv_id)
        title = "新对话"
        for msg in history:
            if msg["role"] == "user":
                title = msg["content"][:30] + ("..." if len(msg["content"]) > 30 else "")
                break
        result.append({
            "conversation_id": conv_id,
            "title": title,
            "message_count": len(history),
            "updated_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    return result
