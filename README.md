# 考研伴学 Agent (Exam Companion Agent)

以考研为导向的智能体后端，支持 RAG 知识问答、做题练习与学习规划。

## 技术栈

| 层级 | 技术 |
|------|------|
| 语言 | Python 3.12 |
| 框架 | FastAPI + Uvicorn |
| AI 框架 | LangChain |
| 对话模型 | DeepSeek V4 Flash (API, OpenAI 兼容) |
| 嵌入模型 | BAAI/bge-small-zh-v1.5 (HuggingFace, 本地 CPU) |
| Rerank 模型 | BAAI/bge-reranker-base (CrossEncoder, 本地 CPU) |
| 向量数据库 | ChromaDB 1.5+ (持久化) |
| 检索引擎 | BM25 (rank-bm25 + jieba) + 稠密向量加权融合 |
| 配置管理 | pydantic-settings |
| 容器化 | Docker + Docker Compose |

## 功能

- **文档智能处理**：上传 PDF/Markdown/TXT/DOCX，三级切分（标题→段落→字符），自动去重，前置内容过滤，目录大纲提取
- **RAG 知识问答**：混合检索（BM25+向量）→ Rerank 精排 → 查询重写（指代消解）→ 引用标注 → 流式 SSE 返回
- **做题练习**：简单出题模式（数量/方向/题型/难度可配）+ 试卷模式（以参考真题为模板仿写）
- **学习规划**：时间线模式（三阶段倒排）+ 灵活模式（按章节编排）+ 学习卡片生成
- **多轮对话**：对话历史持久化 + 自动注入 Prompt + 对话列表管理
- **Web 前端**：双面板（聊天+知识库管理）+ 侧栏对话列表 + LaTeX/Markdown 渲染 + 暗色模式

## 快速开始（协作者指南）

### 前置条件

- Windows / macOS / Linux
- 能够访问 DeepSeek API（需 API Key）
- 能够访问 HuggingFace（或配置国内镜像）

### 第一步：安装 Conda

如果你还没有 Conda，推荐安装 Miniconda：

- Windows: 下载并安装 [Miniconda Windows 64-bit](https://docs.conda.io/en/latest/miniconda.html)
- macOS: `brew install miniconda`
- Linux: `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && bash Miniconda3-latest-Linux-x86_64.sh`

### 第二步：克隆仓库

```bash
git clone https://github.com/zyaplayer/exam-agent.git
cd exam-agent
```

### 第三步：创建 Conda 环境

```bash
conda create -n exam_agent python=3.12 -y
conda activate exam_agent
```

### 第四步：安装依赖

```bash
pip install -r requirements.txt
```

> 如果 HuggingFace 连接失败（国内网络），在终端先设置镜像：
> ```bash
> # Windows PowerShell
> $env:HF_ENDPOINT = "https://hf-mirror.com"
> # macOS / Linux
> export HF_ENDPOINT=https://hf-mirror.com
> ```

### 第五步：配置 API Key

```bash
# 复制环境变量模板
cp .env.example .env
```

编辑 `.env` 文件，填入你的 DeepSeek API Key：

```ini
DEEPSEEK_API_KEY=sk-你的真实Key
```

申请地址: https://platform.deepseek.com/api_keys

### 第六步：启动服务

```bash
python main.py
```

首次启动会自动：
1. 下载嵌入模型 `BAAI/bge-small-zh-v1.5`（约 100MB，仅一次）
2. 创建数据目录
3. 检查 API Key 配置

看到以下输出表示启动成功：

```
[启动] 嵌入模型已就绪
[启动] DeepSeek API Key 已配置
[启动] 考研伴学 Agent v0.1.0
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 第七步：打开测试页

浏览器访问: **http://localhost:8000**

API 文档: **http://localhost:8000/docs**

## 准备测试教材

在 `data/raw_docs/` 目录放入教材文件（PDF/Markdown/TXT/DOCX），然后：

**方式 A：Web 前端上传**
切换到「📚 知识库管理」Tab → 选择文件 → 选择 Collection → 上传入库

**方式 B：命令行上传**

```bash
# 创建教材大纲目录（供学习规划使用）
mkdir -p data/summary_db

# 创建教材目录大纲文件（例）
cat > data/summary_db/高数教材.markdown << 'EOF'
# 高等数学
## 第一章 函数与极限
### 1.1 函数的概念与性质
### 1.2 数列的极限
...
EOF

# 上传教材本体
curl -X POST http://localhost:8000/api/documents/upload \
  -F "file=@data/raw_docs/高数教材.pdf" \
  -F "collection_name=高数教材"
```

## 运行评测

项目内置 30 道高数评测题用于检测 RAG 质量：

```bash
# 首次运行（建立基线）
python -m app.evaluation.eval_rag --baseline

# 修改代码后对比
python -m app.evaluation.eval_rag --compare
```

评测指标：
- **faithfulness**（忠实度）：回答是否来自检索片段
- **context_precision**（检索精确度）：检索片段是否都相关
- **answer_quality**（答案质量）：回答与标准答案的一致性

## 项目结构

```
exam_agent_project/
├── main.py                      启动入口
├── requirements.txt              依赖清单
├── Dockerfile                    容器镜像
├── docker-compose.yml            容器编排
├── .env.example                  环境变量模板
│
├── app/
│   ├── core/
│   │   ├── config.py             全局配置
│   │   └── llm_manager.py        LLM 工厂（DeepSeek + BGE）
│   ├── agent/
│   │   ├── prompts.py            Prompt 模板库
│   │   ├── router.py             混合意图路由
│   │   ├── qa_chain.py           RAG 问答链路
│   │   ├── planner_chain.py      学习规划链路
│   │   ├── quiz_generator.py     做题练习链路
│   │   └── query_rewriter.py     查询重写（指代消解）
│   ├── services/
│   │   ├── vector_store.py       ChromaDB 管理（混合检索 + 去重）
│   │   ├── reranker.py           CrossEncoder 重排序
│   │   └── conversation_service.py 对话历史管理
│   ├── utils/
│   │   ├── document.py           文档解析 + 三级切分
│   │   ├── outline_extractor.py  目录大纲提取
│   │   └── token_counter.py      Token 计数与截断
│   ├── api/
│   │   └── routes.py             12 个 API 接口
│   └── evaluation/
│       └── eval_rag.py           RAG 评测脚本
│
├── data/
│   ├── eval/                     评测数据集
│   ├── summary_db/               教材目录大纲
│   ├── chroma_db/                向量库（运行时生成）
│   ├── raw_docs/                 原始文档（协作者自行准备）
│   └── conversations/            对话历史（运行时生成）
│
├── static/
│   └── index.html                前端页面
│
└── test.py                       临时调试脚本
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 核心对话（QA/Quiz/Planner SSE 流式） |
| POST | `/api/quiz/exam` | 试卷模式出题 |
| POST | `/api/documents/upload` | 上传文档入库 |
| GET | `/api/documents` | Collection 列表 |
| POST | `/api/collections/bind` | 创建教材-试卷绑定 |
| GET | `/api/collections/bindings` | 查询绑定 |
| GET | `/api/conversations` | 对话列表 |
| GET | `/api/health?deep=true` | 健康检查（深度） |

## Docker 部署

```bash
docker compose up -d
```

## License

MIT
