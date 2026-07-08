# 考研伴学 Agent (Exam Companion Agent)

以考研为导向的 AI 智能体后端，支持 RAG 知识问答、做题练习与学习规划。

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.12 |
| 框架 | FastAPI + Uvicorn |
| AI 框架 | LangChain |
| 对话模型 | DeepSeek V4 Flash (OpenAI 兼容 API) |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 (HuggingFace, 本地 CPU) |
| Rerank 模型 | BAAI/bge-reranker-base (CrossEncoder, 本地 CPU) |
| 向量数据库 | ChromaDB (持久化) |
| 检索引擎 | BM25 (rank-bm25 + jieba) + 稠密向量加权融合 |
| 配置管理 | pydantic-settings |
| 容器化 | Docker + Docker Compose |

## 项目结构

```
exam_agent_project/
├── main.py                          # FastAPI 启动入口
├── requirements.txt                 # Python 依赖
├── Dockerfile                       # Docker 镜像
├── docker-compose.yml               # Docker 编排
├── .env.example                     # 环境变量模板
├── README.md                        # 用户文档
├── CLAUDE.md                        # 本文件 — AI 助手指南
├── TODO.md                          # 待办事项清单（工作追踪）
│
├── app/
│   ├── core/
│   │   ├── config.py                # 全局配置 (pydantic-settings)
│   │   └── llm_manager.py           # LLM 工厂 (DeepSeek + 本地 BGE 嵌入)
│   │
│   ├── agent/
│   │   ├── router.py                # 混合意图路由 (规则引擎 + LLM 分类)
│   │   ├── qa_chain.py              # RAG 问答链路 (检索→Rerank→生成)
│   │   ├── quiz_generator.py        # 做题练习链路 (简单出题 + 试卷模式)
│   │   ├── planner_chain.py         # 学习规划链路 (时间线 + 灵活模式)
│   │   ├── query_rewriter.py        # 查询重写 (指代消解)
│   │   └── prompts.py               # 所有 System/User Prompt 模板
│   │
│   ├── services/
│   │   ├── vector_store.py          # ChromaDB 管理 (混合检索 + 层级检索)
│   │   ├── reranker.py              # CrossEncoder 重排序
│   │   └── conversation_service.py  # 对话历史管理
│   │
│   ├── utils/
│   │   ├── document.py              # 文档解析 + 三级切分
│   │   ├── outline_extractor.py     # 目录大纲提取
│   │   └── token_counter.py         # Token 计数与截断
│   │
│   ├── api/
│   │   └── routes.py                # 12 个 API 接口
│   │
│   └── evaluation/
│       └── eval_rag.py              # RAG 质量评测
│
├── data/                            # 运行时数据目录
│   ├── chroma_db/                   # 向量库持久化 (自动生成)
│   ├── raw_docs/                    # 原始文档上传目录
│   ├── summary_db/                  # 教材目录大纲
│   ├── conversations/               # 对话历史 JSON
│   ├── eval/                        # 评测数据集
│   └── .bm25_index/                 # BM25 预建索引缓存
│
└── static/
    └── index.html                   # Web 前端 (双面板 + 暗色模式)
```

## 启动方式

```bash
# 1. 创建环境
conda create -n exam_agent python=3.12 -y && conda activate exam_agent

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env → 填入 DEEPSEEK_API_KEY

# 4. 启动 (国内网络需设 HF 镜像)
# Windows: $env:HF_ENDPOINT = "https://hf-mirror.com"
# macOS/Linux: export HF_ENDPOINT=https://hf-mirror.com
python main.py

# 5. 访问
# http://localhost:8000       → 前端页面
# http://localhost:8000/docs  → Swagger API 文档
```

## 架构概览

```
用户输入 → router.py (意图识别: QA/Quiz/Planner)
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
qa_chain  quiz_gen  planner
    │         │         │
    └─────────┼─────────┘
              ▼
    vector_store.py (混合检索)
      ├── BM25 关键词 (jieba 分词)
      ├── 稠密向量 (ChromaDB + BGE)
      └── 层级两阶段 (大纲定位 → 章节精搜)
              │
              ▼
    reranker.py (CrossEncoder 精排 Top-K)
              │
              ▼
    token_counter.py (截断保护)
              │
              ▼
    DeepSeek V4 Flash → SSE 流式返回
```

## 核心设计决策

1. **混合意图路由**：规则引擎优先（零成本），低置信度升级到 LLM 分类
2. **文档切分**：三级策略 — Markdown 标题切分 → 段落切分 → 递归字符切分
3. **检索管道**：BM25 + 稠密向量加权融合 → CrossEncoder Rerank 精排
4. **大纲索引**：上传文档时自动提取章节目录 → 存入 `_outline` 内部索引库 → 层级检索时先用大纲定位章节
5. **查询重写**：检测指代词（它/这/那）→ LLM 指代消解 → 完整问题检索
6. **Token 安全**：估算 total → 超限按 7:3 比例截断检索文档和对话历史
7. **所有 LLM 调用均通过 `llm_manager.py` 工厂**，provider 参数切换模型

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 核心对话 (SSE 流式) |
| POST | `/api/quiz/exam` | 试卷模式出题 |
| POST | `/api/documents/upload` | 上传文档入库 |
| GET | `/api/documents` | Collection 列表 |
| POST | `/api/collections/bind` | 教材-试卷绑定 |
| GET | `/api/collections/bindings` | 查询绑定 |
| DELETE | `/api/collections/bindings/{id}` | 删除绑定 |
| GET | `/api/conversations` | 对话列表 |
| GET | `/api/conversations/{id}` | 对话历史 |
| DELETE | `/api/conversations/{id}` | 删除对话 |
| GET | `/api/collections/aliases` | 中文名映射 |
| GET | `/api/health?deep=true` | 健康检查 |

## 代码约定

- Python 文件头有模块 docstring，说明用途
- 函数使用 Google-style docstring (Args/Returns/Raises)
- 流式函数返回 `AsyncIterator[str]`，非流式返回 `str`
- 异常处理：区分 API Key 错误 / 超时 / 通用错误
- SSE 流以 `data: [DONE]\n\n` 结尾

## 注意事项

- 嵌入模型首次下载约 100MB，需等待
- HuggingFace 国内访问需设置 `HF_ENDPOINT=https://hf-mirror.com`
- ChromaDB 1.5+ 要求同一目录共享一个 PersistentClient 实例
- DeepSeek 不提供 Embedding API，嵌入使用本地 BGE 模型
- `app/__init__.py` 在项目导入时自动设置 HF_ENDPOINT 镜像
