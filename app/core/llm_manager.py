"""
考研伴学 Agent - 模型统一管理入口
==================================
提供 LLM 和 Embedding 模型的工厂函数。
对话模型通过 provider 参数切换不同厂商（DeepSeek / Qwen 等）。
嵌入模型使用本地 HuggingFace 模型（免费、离线、无需额外 API Key）。
"""

from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.config import settings


# ============================================
# 对话模型提供商注册表
# ============================================
# 格式：
#   "provider名称": {
#       "api_key":    settings 中对应的 API Key 字段值,
#       "base_url":   settings 中对应的 Base URL 字段值,
#       "chat_model": settings 中对应的对话模型名称字段值,
#   }
# 后续新增模型只需在此字典中添加一行即可。

_PROVIDER_CONFIGS = {
    "deepseek": {
        "api_key": settings.DEEPSEEK_API_KEY,
        "base_url": settings.DEEPSEEK_BASE_URL,
        "chat_model": settings.DEEPSEEK_MODEL_NAME,
    },
    "qwen": {
        "api_key": settings.QWEN_API_KEY,
        "base_url": settings.QWEN_BASE_URL,
        "chat_model": settings.QWEN_MODEL_NAME,
    },
    # [TODO: 如需添加新模型，参考上方格式在此处新增一条即可。
    #   例如添加 OpenAI 原生：
    #   "openai": {
    #       "api_key": settings.OPENAI_API_KEY,
    #       "base_url": settings.OPENAI_BASE_URL,
    #       "chat_model": settings.OPENAI_MODEL_NAME,
    #   }]
}


# ============================================
# 嵌入模型（全局单例，使用本地 HuggingFace 模型）
# ============================================
# DeepSeek 不提供 Embedding API，因此使用本地 sentence-transformers 模型。
# 首次加载时会自动从 HuggingFace 下载，之后缓存在本地。
# 模型选择：BAAI/bge-small-zh-v1.5
#   - 中文友好，轻量（~100MB），CPU 即可运行
#   - 如需更高精度可替换为 BAAI/bge-large-zh-v1.5（~400MB）

_embedding_instance: HuggingFaceEmbeddings | None = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """
    获取嵌入模型实例（全局单例，懒加载）。

    使用本地 HuggingFace sentence-transformers 模型，无需 API Key。
    首次调用时会下载模型文件（约 100MB），请耐心等待。
    后续调用直接返回缓存实例。

    返回:
        HuggingFaceEmbeddings 实例，用于将文本转为向量存入 ChromaDB。
    """
    global _embedding_instance
    if _embedding_instance is None:
        print(f"[嵌入模型] 正在加载 {settings.LOCAL_EMBEDDING_MODEL} ...")
        _embedding_instance = HuggingFaceEmbeddings(
            model_name=settings.LOCAL_EMBEDDING_MODEL,
            # 模型参数（可根据硬件调整）
            model_kwargs={"device": "cpu"},  # [TODO: 如有 GPU 改为 "cuda"]
            encode_kwargs={"normalize_embeddings": True},  # 向量归一化，提升相似度计算精度
        )
        print(f"[嵌入模型] 加载完成")
    return _embedding_instance


def get_llm(
    provider: str = "deepseek",
    temperature: float = 0.7,
    streaming: bool = True,
) -> ChatOpenAI:
    """
    根据提供商名称获取对应的对话 LLM 实例。

    参数:
        provider:    模型提供商名称，可选 "deepseek", "qwen" 等。
                     默认 "deepseek"。
        temperature: 模型温度，控制回答随机性。0 = 确定性最强，1 = 最随机。
                     默认 0.7。
        streaming:   是否启用流式输出。默认 True（配合 SSE 向前端流式推送）。
                     如果用于非流式场景（如意图识别），可设为 False。

    返回:
        ChatOpenAI 实例（兼容 OpenAI SDK 格式）。
        DeepSeek 和 Qwen 等国产模型均兼容此格式。

    异常:
        ValueError: 当传入的 provider 不在注册表中，或对应的 API Key 未配置时抛出。
    """
    if provider not in _PROVIDER_CONFIGS:
        available = ", ".join(_PROVIDER_CONFIGS.keys())
        raise ValueError(
            f"不支持的模型提供商: '{provider}'。"
            f"当前可用的提供商: {available}"
        )

    config = _PROVIDER_CONFIGS[provider]

    if not config["api_key"]:
        raise ValueError(
            f"模型提供商 '{provider}' 的 API Key 未配置。"
            f"请在 .env 文件中设置对应的 API Key 环境变量。"
        )

    return ChatOpenAI(
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["chat_model"],
        temperature=temperature,
        streaming=streaming,
    )


def list_providers() -> list[str]:
    """
    列出当前所有已注册的对话模型提供商名称。
    可用于前端下拉框展示或健康检查接口。

    返回:
        提供商名称列表，如 ["deepseek", "qwen"]
    """
    return list(_PROVIDER_CONFIGS.keys())
