# Commspec RAG — 通信协议规范检索增强生成系统

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Milvus-2.4-00D4AA?logo=milvus" alt="Milvus">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi" alt="FastAPI">
  <img src="https://img.shields.io/badge/Next.js-14-000000?logo=next.js" alt="Next.js">
  <img src="https://img.shields.io/badge/Docker-✓-2496ED?logo=docker" alt="Docker">
  <img src="https://img.shields.io/badge/License-BSD--3--Clause-blue.svg" alt="License">
</p>

面向 **3GPP + O-RAN 通信协议规范**的专用检索增强生成（RAG）系统，支持中英韩三语智能问答、跨规范引用解析、Release 版本对比与多跳推理。

## ✨ 核心特性

- **🔍 混合检索** — Dense (BGE-M3) + Sparse (Milvus BM25) 双路召回，NN Router 精排
- **📚 多跳推理** — LLM 缺口分析 → 子查询生成 → 自动串联多篇规范
- **🔗 交叉引用** — 自动识别 `TS 38.413 §8.3.1` 并补充被引规范内容
- **📅 Release 感知** — 检测 R17/R18 版本意图，支持多版本对比问答
- **🌐 多语言** — 中/英/韩三语查询 → LLM 翻译 → 英文检索 → 回译回答
- **📡 在线补充** — 离线不足时自动触发 Google CSE / TSpec-LLM
- **📊 监控运维** — Prometheus 指标 + `/metrics` 端点 + 管理后台
- **✅ 幻觉防护** — 答案溯源验证，每个论断需标注规范出处

## 🏗 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      Commspec RAG Pipeline                      │
├──────────┬──────────┬──────────┬──────────┬──────────┬─────────┤
│ Step 0   │ Step 1   │ Step 3   │ Step 3.2 │ Step 3.5 │ Step 3.7│
│ 多语言   │ 查询扩展 │ 混合检索 │ 多跳检索 │ 交叉引用 │ Release │
│ i18n     │ LLM      │ Milvus   │ MultiHop │ CrossRef │ 感知    │
├──────────┼──────────┼──────────┼──────────┼──────────┼─────────┤
│ Step 3.8 │ Step 4   │ Step 5   │ Step 5.5 │ Step 6   │         │
│ 在线补充 │ Prompt   │ LLM 生成 │ 回译     │ 答案验证 │         │
│ Google   │ 构建     │          │ i18n     │ Verifier │         │
└──────────┴──────────┴──────────┴──────────┴──────────┴─────────┘
```

| 层级 | 模块 | 说明 |
|------|------|------|
| 📥 Ingestion | `ingestion/` | DOCX → Markdown → 分块 → BGE 嵌入 → Milvus |
| 🔍 Retriever | `retriever/` | 混合检索 + NN Router + 多跳 + 交叉引用 + 在线补充 |
| ✨ Generator | `generator/` | Prompt 模板 + LLM 客户端 + 答案验证 + 多语言 |
| 🌐 API | `api/` | REST (FastAPI) + MCP 工具 + Prometheus /metrics |
| 🖥 Frontend | `frontend/` | Next.js 14 用户端 + 管理后台 (5 页面 + 暗色双主题) |

## 📁 目录结构

```
commspec_rag_project/
├── src/
│   ├── config/             # pydantic-settings 配置
│   ├── ingestion/          # 文档摄取（extractor, splitter, embedder, manifest）
│   ├── retriever/          # 检索（milvus_store, search, router, multi_hop, cross_ref）
│   ├── generator/          # 生成（pipeline, prompt, llm_client, verifier, i18n）
│   ├── api/
│   │   ├── rest/           #   REST API + 管理端点 + 反馈
│   │   └── mcp/            #   MCP 工具（可选）
│   ├── utils/              # 监控 (Prometheus) + 设备检测
│   ├── main.py             # FastAPI 入口
│   └── cli.py              # CLI 入口
├── frontend/               # Next.js 14 前端
│   ├── app/                #   / (用户端) + /admin/* (管理后台)
│   ├── components/         #   ThemeToggle
│   └── lib/                #   API 客户端 + 主题
├── scripts/                # bulk_ingest, download_specs
├── deploy/                 # Dockerfile + docker-compose
├── tests/                  # 测试 + 评测集 (70 题)
├── docs/
│   └── architecture.md     # 架构文档
├── data/                   # 运行时数据 (不入库)
│   ├── documents/R18/      #   3GPP DOCX 源文件 (400 篇)
│   ├── vectors/            #   向量索引
│   ├── manifest/           #   摄入清单（SHA256 + 版本号）
│   ├── checkpoint/         #   断点续传进度
│   └── cache/              #   嵌入缓存 (embedding_cache.db)
├── .env.example            # 环境变量模板
├── requirements.txt        # Python 依赖
└── .gitignore
```

## 🚀 快速开始

### 前置条件

- **Python 3.11+**
- **Docker Desktop**（运行 Milvus）
- **pandoc**（DOCX → Markdown 转换）
- **Node.js 18+**（前端开发，可选）

### 1. 克隆与安装

```bash
git clone https://github.com/Eric-HK-L/commspec_rag_project.git
cd commspec_rag_project

# Python 虚拟环境
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 前端依赖
cd frontend && npm install && cd ..
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，至少配置：
#   LLM_API_KEY=sk-xxx          (LLM API 密钥)
#   LLM_BASE_URL=https://...    (API 端点)
#   LLM_MODEL=gpt-4o-mini       (模型名)
```

### 3. 启动 Milvus

```bash
cd deploy
docker compose up -d
# 确认 Milvus 运行: docker ps | grep milvus
```

### 4. 准备文档

将 3GPP R18 + O-RAN 规范 `.docx` 文件放入 `data/documents/R18/<series>/`：

```
data/documents/R18/
├── 21_series/
├── 22_series/  (25 篇)
├── 23_series/  (86 篇)
├── 24_series/  (55 篇)
├── 36_series/  (85 篇)
└── 38_series/  (148 篇)
```

### 5. 构建知识库

```bash
# 全量摄入（首次部署）
python scripts/bulk_ingest.py

# 增量摄入（日常维护，默认模式：仅处理新增/修改文档）
# 直接运行即可，脚本自动跳过已处理文件
python scripts/bulk_ingest.py

# 断点续传（摄入中断后恢复）
python scripts/bulk_ingest.py --resume-from-checkpoint
```

### 6. 启动服务

```bash
# 后端 API (端口 8000)
python -m src.main
# 或: uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload

# 前端开发服务器 (端口 3000)
cd frontend && npm run dev
```

| 服务 | 地址 |
|------|------|
| API 文档 (Swagger) | http://localhost:8000/docs |
| 用户前端 | http://localhost:3000 |
| 管理后台 | http://localhost:3000/admin |
| Prometheus 指标 | http://localhost:8000/metrics |

默认管理员账号：`admin` / `linux123`

### Docker 一键部署

```bash
cd deploy
# 启动 Milvus + API + 前端（数据摄入需手动执行）
docker compose up -d
# 后端 :8000  + 前端 :3000  + Milvus :19530

# 数据摄入（在宿主机上执行）
python scripts/bulk_ingest.py
```

## 🔧 技术栈

| 组件 | 选型 |
|------|------|
| LLM | OpenAI 兼容 API（GPT-4o / DeepSeek / Qwen） |
| 嵌入模型 | BAAI/bge-m3（多语言，1024-dim，稠密+稀疏双向量） |
| 向量数据库 | Milvus 2.4（Dense + Sparse BM25） |
| 文档处理 | Docling + python-docx + pandoc |
| API 框架 | FastAPI + Uvicorn |
| 前端 | Next.js 14 + Tailwind CSS + Recharts |
| 监控 | prometheus_client |
| 配置 | pydantic-settings + python-dotenv |
| 部署 | Docker Compose |

## 📖 参考论文

- [Chat3GPP — arXiv:2501.13954](https://arxiv.org/abs/2501.13954)
- [Telco-RAG — arXiv:2404.15939](https://arxiv.org/abs/2404.15939)

## 📚 更多文档

| 文档 | 说明 |
|------|------|
| [架构设计](./docs/architecture.md) | 系统架构、模块详解、数据流、关键决策 |
| [离线部署指南](./docs/offline-deployment.md) | 内网环境离线安装（pip wheels + Docker 镜像） |
| [硬件兼容性](./docs/hardware-compatibility.md) | 跨平台运行指南（Intel / NVIDIA GB10 / Apple Silicon） |
| [Phase 计划](./docs/plans/) | 项目演进路线图（Phase 1-4） |

## 📄 License

[BSD-3-Clause](LICENSE)
