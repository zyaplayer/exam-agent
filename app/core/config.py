"""
考研伴学 Agent - 全局配置管理
==============================
基于 pydantic-settings 自动读取 .env 文件。
所有配置项均可通过环境变量覆盖，方便 Docker 部署。
"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    应用全局配置类。
    实例化时会自动从 .env 文件加载环境变量。
    变量名不区分大小写，下划线自动映射为大写环境变量。
    """

    # ---- 基础信息 ----
    PROJECT_NAME: str = "考研伴学 Agent"
    PROJECT_VERSION: str = "0.1.0"
    DEBUG: bool = True

    # ---- 项目根目录 ----
    # 自动定位到 exam_agent_project/ 根目录
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

    # ============================================
    # 模型配置：默认模型 (DeepSeek V4)
    # ============================================
    DEEPSEEK_API_KEY: str = ""
    # [TODO: 请在此填入你的 DeepSeek API Key]
    #  申请地址：https://platform.deepseek.com/api_keys

    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_MODEL_NAME: str = "deepseek-v4-flash"
    # DeepSeek 不提供 Embedding API，嵌入模型使用本地 HuggingFace 模型
    LOCAL_EMBEDDING_MODEL: str = "BAAI/bge-small-zh-v1.5"

    # ============================================
    # 模型配置：备用模型 (Qwen / 通义千问)
    # ============================================
    QWEN_API_KEY: str = ""
    # [TODO: 如果你需要使用阿里云通义千问作为备用模型，
    #   请在此填入 Qwen API Key。
    #   申请地址：https://dashscope.console.aliyun.com/apiKey]

    QWEN_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_MODEL_NAME: str = "qwen-plus"

    # [TODO: 如需接入第三个备用模型（如 OpenAI 原生、Claude 等），
    #   请在此区块下方按照相同模式新增一组环境变量，
    #   然后在 llm_manager.py 的 _PROVIDER_CONFIGS 中注册即可。]

    # ============================================
    # 数据存储配置
    # ============================================
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    RAW_DOCS_DIR: str = "./data/raw_docs"
    SUMMARY_DB_DIR: str = "./data/summary_db"

    # ============================================
    # 文档切分配置
    # ============================================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # 按 Markdown 标题切分的层级（例如设为 2 表示按 H2 切分，保留 "章" 的边界）
    # 设为 1 最粗（按 H1 切，整章一个块），设为 3 更细（按 H3 切）
    MARKDOWN_SPLIT_HEADER_LEVEL: int = 3

    # 解析器能识别的最大标题层级（防止误将正文中的 # 编号当作标题）
    MARKDOWN_HEADER_MAX_LEVEL: int = 4

    # ============================================
    # RAG 检索配置
    # ============================================
    RETRIEVAL_K: int = 4          # 向量检索返回的最相关文档数
    RETRIEVAL_THRESHOLD: float = 0.0  # 相似度阈值（0 表示不过滤）
    MAX_CONTEXT_TOKENS: int = 8000  # Prompt 最大 token 数（超出时自动截断）

    # 多轮对话历史保留的最大轮数（一轮 = 1 问 + 1 答）
    MAX_CONVERSATION_TURNS: int = 10

    # ============================================
    # API 服务配置
    # ============================================
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        """pydantic-settings 的元配置"""
        # .env 文件的搜索路径
        env_file = str(Path(__file__).resolve().parent.parent.parent / ".env")
        env_file_encoding = "utf-8"
        # 允许读取额外的环境变量（不报错）
        extra = "ignore"


# 全局单例 —— 整个项目只实例化一次
settings = Settings()
