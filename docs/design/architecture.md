# CommSpec RAG — 系统架构文档

> 版本：v2.3 | 最后更新：2026-07-15

---

## 目录

1. [系统概览](#1-系统概览)
2. [架构总览](#2-架构总览)
3. [模块详解](#3-模块详解)
   - [3.1 配置层](#31-配置层-srcconfig)
   - [3.2 文档摄入层](#32-文档摄入层-srcingestion)
   - [3.3 检索层](#33-检索层-srcretriever)
   - [3.4 生成层](#34-生成层-srcgenerator)
   - [3.5 API 服务层](#35-api-服务层-srcapi)
   - [3.6 前端层](#36-前端层-frontend)
   - [3.7 工具层](#37-工具层-srcutils)
4. [数据流](#4-数据流)
5. [关键设计决策](#5-关键设计决策)
6. [部署架构](#6-部署架构)
7. [附录：设计遗产](#7-附录设计遗产)

---

## 1. 系统概览

CommSpec RAG 是一个**生产级 3GPP 通信标准专用 RAG 系统**，提供：

- **精准检索** — Dense 向量 + BM25 关键词双路混合检索，RRF 融合排序
- **智能问答** — 检索增强生成，含幻觉验证、交叉引用解析、多跳检索、Release 版本感知
- **多语言支持** — 中文/韩文查询自动翻译为英文检索，回答回译为源语言
- **多通道访问** — REST API（Web UI 交互）+ MCP 工具（AI Agent 调用）
- **跨平台部署** — Linux x86_64 / NVIDIA GB10 (ARM64 + CUDA) / macOS Apple Silicon
- **离线友好** — 完整离线部署方案，适配公司内网环境

### 技术栈

| 层级       | 选型                                                                |
| ---------- | ------------------------------------------------------------------- |
| API 框架   | FastAPI + Uvicorn                                                   |
| LLM 集成   | OpenAI SDK（base_url + api_key + model 三参数统一切换）             |
| 嵌入模型   | BGE-M3（多语言，1024-dim，稠密+稀疏双向量）                         |
| 向量数据库 | Milvus 2.4+（Dense + BM25 混合检索；BGE-M3 原生 sparse 向量待迁移） |
| 文档处理   | Pandoc（DOCX → Markdown）                                           |
| 前端       | Next.js 16（React 19 + Tailwind CSS 4）+ react-markdown             |
| 部署       | Docker Compose（etcd + MinIO + Milvus + API + Frontend）            |

### 数据目录结构

所有运行时数据从 `DATA_DIR` 环境变量派生（默认 `./data/`），子路径均从该根目录推导：

```
{DATA_DIR}/                          ← 环境变量 DATA_DIR, 默认 data/
├── documents/                       ← DOCUMENTS_DIR（独立可配，默认 data/documents）
│   └── R18/{21,22,23,24,36,38}_series/  ← 3GPP DOCX 源文件
├── vectors/
│   └── bm25_index.pkl              ← BM25 稀疏检索索引
├── manifest/
│   └── ingestion_state.json        ← 摄入清单（SHA256 + 版本号）
├── cache/
│   └── embedding_cache.db          ← 嵌入向量 SQLite 缓存
├── checkpoint/
│   └── chunks_checkpoint.pkl       ← 提取阶段断点（避免重复 DOCX 解析）
├── feedback.db                      ← 用户反馈 SQLite 数据库
└── processed/                       ← 中间产物（.gitkeep 保留目录）
```

> **外置存储**：如需将数据目录挂载到 NAS/外置硬盘，在 `.env` 中设置 `DATA_DIR=/mnt/nas/3gpp-rag-data`，所有子路径自动切到该位置。`DOCUMENTS_DIR` 保持独立可配，以支持文档源与运行时数据物理分离。

---

## 2. 架构总览

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          用户接口层                                       │
│  ┌──────────────────┐  ┌──────────────────────┐  ┌───────────────────┐  │
│  │   Next.js 前端     │  │   REST API (FastAPI)  │  │  MCP Tools       │  │
│  │   (port 3000)     │  │   (port 8000)         │  │  (AI Agent 调用)  │  │
│  └────────┬─────────┘  └──────────┬───────────┘  └─────────┬─────────┘  │
│           │                       │                        │            │
├───────────┼───────────────────────┼────────────────────────┼────────────┤
│           │               API 编排层                        │            │
│           └───────────────────────┬────────────────────────┘            │
│                                   │                                     │
├───────────────────────────────────┼─────────────────────────────────────┤
│                          生成层 (Generator)                              │
│  ┌─────────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────────┐   │
│  │ RAGPipeline │ │  LLMClient   │ │ Verifier │ │ i18n / Release   │   │
│  │ (主编排器)   │ │ (OpenAI SDK) │ │ (幻觉校验)│ │ (多语言+版本感知) │   │
│  └─────────────┘ └──────────────┘ └──────────┘ └──────────────────┘   │
│                                   │                                     │
├───────────────────────────────────┼─────────────────────────────────────┤
│                          检索层 (Retriever)                              │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────┐ ┌────────────────┐  │
│  │HybridRetriever│ │ MilvusStore  │ │ Cross-Ref  │ │ MultiHop /     │  │
│  │Dense+BM25+RRF│ │ (向量库后端)  │ │ (引用解析)  │ │ Online Suppl.  │  │
│  └──────────────┘ └──────────────┘ └────────────┘ └────────────────┘  │
│                                   │                                     │
├───────────────────────────────────┼─────────────────────────────────────┤
│                        文档摄入层 (Ingestion)                             │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐              │
│  │Extractor │→│ Splitter │→│ Embedder │→│ Orchestrator │              │
│  │DOCX→MD   │ │标题分块   │ │向量生成   │ │ 全流程编排    │              │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘              │
│  ┌──────────────────┐ ┌─────────────────┐ ┌──────────────────────┐    │
│  │ EmbeddingCache   │ │ MPSChunkedEmbedder│ │ Manifest / Incremental│  │
│  │ (向量缓存)        │ │ (Apple GPU 安全)  │ │ (状态追踪+增量更新)   │    │
│  └──────────────────┘ └─────────────────┘ └──────────────────────┘    │
│                                   │                                     │
├───────────────────────────────────┼─────────────────────────────────────┤
│                          基础设施层                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌────────────────────────┐    │
│  │ Milvus   │ │ BM25     │ │ Prometheus│ │ Pydantic Settings     │    │
│  │ (向量库)  │ │ (关键词)  │ │ (监控)    │ │ (配置管理)            │    │
│  └──────────┘ └──────────┘ └──────────┘ └────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────┘
```

### 源码目录映射

```
src/
├── config/settings.py          ← 全局配置 (Pydantic)
├── ingestion/                  ← 文档摄入管线
│   ├── extractor.py            ← DOCX → Markdown (Pandoc)
│   ├── splitter.py             ← 智能分块 (标题感知 + Grid Table 行组拆分 + 字节门禁)
│   ├── embedder.py             ← 批量嵌入生成
│   ├── orchestrator.py         ← 全流程编排
│   ├── mps_embedder.py         ← Apple MPS GPU 安全嵌入 (spawn)
│   ├── incremental.py          ← 增量索引
│   ├── manifest.py             ← 摄入状态追踪
│   ├── release_monitor.py      ← 3GPP 新版本监控
│   └── embedding_cache.py      ← 嵌入缓存
├── retriever/                  ← 检索层
│   ├── vector_store.py         ← 抽象接口
│   ├── milvus_store.py         ← Milvus 2.4+ 实现 (Dense + BM25)
│   ├── bm25_index.py           ← Python BM25 索引器
│   ├── search.py               ← HybridRetriever (Dense+BM25+RRF)
│   ├── cross_ref.py            ← 交叉引用解析 + 二次检索
│   ├── multi_hop.py            ← 多跳检索 (迭代 Agent)
│   ├── online_supplement.py    ← 在线搜索补充
│   ├── query_quality.py        ← 检索质量评估
│   └── router.py               ← NN Router 重排序
├── generator/                  ← 生成层
│   ├── pipeline.py             ← RAGPipeline (主编排器)
│   ├── llm_client.py           ← OpenAI 兼容 LLM 客户端
│   ├── prompt.py               ← 提示词模板
│   ├── verifier.py             ← 幻觉验证
│   ├── i18n.py                 ← 多语言 (zh/ko → EN → zh/ko)
│   ├── release_aware.py        ← Release 版本感知
│   └── feedback.py             ← 用户反馈分析 (统计报告生成)
├── api/                        ← API 层
│   ├── rest/router.py          ← REST API (/search, /ask, /documents)
│   ├── rest/admin_router.py    ← 管理端点
│   ├── rest/feedback.py        ← 用户反馈
│   ├── rest/middleware.py       ← 请求日志/异常处理
│   ├── rest/schemas.py         ← Pydantic 模型
│   ├── mcp/server.py           ← MCP 工具服务
│   ├── mcp/tools.py            ← MCP 工具定义
│   └── auth.py                 ← API Key 认证
├── utils/
│   ├── helpers.py              ← 硬件检测 + 平台适配
│   └── monitoring.py           ← Prometheus 指标
├── main.py                     ← FastAPI 服务入口
└── cli.py                      ← CLI 工具 (stats/ingest/incremental/feedback)

tests/
└── eval/                       ← 检索质量评测
    ├── run_eval.py             ← 评测执行 (完整 RAGPipeline 检索)
    ├── metrics.py              ← Recall@K / MRR / NDCG 指标
    └── test_set.json           ← 70 题 3GPP QA 评测集
```

---

## 3. 模块详解

### 3.1 配置层 (`src/config/`)

**[settings.py](file://src/config/settings.py)** — 基于 pydantic-settings 的环境变量管理，所有配置从 `.env` 自动加载。

**核心配置项**：

| 分类   | 关键项                                                            | 说明                                          |
| ------ | ----------------------------------------------------------------- | --------------------------------------------- |
| LLM    | `llm_base_url`, `llm_api_key`, `llm_model`                        | OpenAI 兼容三参数，覆盖公司自建/公有云/Ollama |
| 嵌入   | `embedding_provider`, `embedding_device`, `local_embedding_model` | api/local 切换，auto/cuda/mps/cpu 设备选择    |
| 向量库 | `milvus_host`, `milvus_port`, `milvus_collection_name`            | Milvus 连接参数                               |
| 摄入   | `chunk_size`, `chunk_overlap`, `documents_dir`                    | DOCX 文档路径 + 分块参数                      |
| 检索   | `dense_top_k`, `bm25_top_k`                                       | 混合检索参数                                  |

**智能属性**：

- `resolved_embedding_device` — `auto` 时自动检测平台（NVIDIA→cuda, Apple Silicon→cpu 安全降级）
- `documents_abs_dir` / `log_abs_file` — 相对路径 → 绝对路径

### 3.2 文档摄入层 (`src/ingestion/`)

#### 3.2.1 摄入策略

当前统一使用 DOCX 从零摄入（`bulk_ingest.py`），直接从 3GPP DOCX 源文件提取、分块、嵌入、入库。

| 模式         | 配置值         | 数据来源                | 适用场景                   |
| ------------ | -------------- | ----------------------- | -------------------------- |
| **从零构建** | `from_scratch` | 本地 3GPP DOCX → Milvus | 完全可控分块策略，生产默认 |

#### 3.2.2 核心模块

**[extractor.py](file://src/ingestion/extractor.py)** — DOCX → Markdown 转换（Pandoc 管线），保留表格、列表、公式结构。

**[splitter.py](file://src/ingestion/splitter.py)** — 三层自适应分块引擎，零信息丢失：

| 层级                      | 机制                                              | 触发条件           | 作用                       |
| ------------------------- | ------------------------------------------------- | ------------------ | -------------------------- |
| **Layer 1: 标题感知**     | 按 `#/##/###` 层级切分，保留父章节路径            | 默认               | 结构化语义分块             |
| **Layer 2: 内容感知拆分** | Grid Table 行组拆分 / HTML `<tr>` 拆分 / 换行拆分 | chunk > 55KB       | 在语义边界精确切割巨型表格 |
| **Layer 3: 字节截断兜底** | `_safe_truncate_bytes` 语义边界截断               | 极少数不可拆分内容 | 确保入库不崩溃             |

**Grid Table 智能拆分**：

- 无状态边界收集算法，统一处理标准表、合并单元格表、`+===+` 子表头三种异构 Pandoc 输出
- 每个子表自包含完整列头 + 文档上下文（heading + prose），独立可检索
- 实测：967 个超限 chunk → 3,697 个安全子表，99.4% ≤ 55KB，行覆盖率 99.5%

**崩溃防护**：

- 提取阶段 `_normalize_chunk_sizes` 确保 checkpoint 已是干净尺寸（≤55KB）
- 嵌入阶段 `_safe_truncate_bytes` 双重兜底
- Milvus VARCHAR 65535 硬限制，55KB 提供 10KB+ 安全边距

`chunk_size=512`, `chunk_overlap=50` 可配。

**[embedder.py](file://src/ingestion/embedder.py)** — 批量嵌入生成器，支持云端 API 和本地 BGE 模型双后端，带重试和断点续传。

**[mps_embedder.py](file://src/ingestion/mps_embedder.py)** — Apple Silicon MPS GPU 安全嵌入方案：

- 使用 Python `spawn` 而非 `fork`（Apple Metal 框架 fork-unsafe）
- 父进程永不 import torch，零 GPU 状态
- **workers=1（硬约束）**：Apple MPS 不支持多进程同时使用 GPU，多进程
  同时调用 `[MTLCommandBuffer waitUntilCompleted]` 触发 Metal 调度器死锁
- 单 worker 每 `chunks_per_worker=200` 批后退出重建 → OS 回收 GPU 内存
- GPU 内部已高度并行，M4 Max 单进程实测 ~120 t/s (3GPP 长文本)，无需多 worker
- 实际使用独立 subprocess + 单进程 MPS 直接编码（非 ProcessPoolExecutor），零 IPC 开销

**[orchestrator.py](file://src/ingestion/orchestrator.py)** — 串联 下载→转换→分块→嵌入→入库 全流程，支持 `--skip-*` 分段执行和中间结果落盘。

**[incremental.py](file://src/ingestion/incremental.py)** — 基于文件 hash 的增量更新，仅处理新增/修改文档。

**[manifest.py](file://src/ingestion/manifest.py)** — 摄入状态追踪（JSON 文件），记录已处理文档的 hash 和时间戳。

### 3.3 检索层 (`src/retriever/`)

#### 3.3.1 混合检索策略

```
用户查询
    │
    ▼
┌─────────────────────────────────────────────┐
│  Dense 检索 (Milvus IVF_FLAT, COSINE)       │  ← 语义相似度，Top-100
│  + BM25 检索 (Python rank-bm25)             │  ← 关键词精确匹配，Top-100
│  → RRF 融合排序 (k=60)                       │  ← 默认策略，不可绕过
│  → 最终 Top-K 输出                           │
└─────────────────────────────────────────────┘
```

**为什么必须混合检索**：3GPP 规范文本密度极高，不同章节共享大量相同术语（PDU Session、AMF、gNB 等），纯语义检索会产生"词像意不像"的系统性误匹配。BM25 提供关键词锚点防止语义漂移。

#### 3.3.2 核心模块

**[vector_store.py](file://src/retriever/vector_store.py)** — 抽象接口，定义 `SearchResult` / `Chunk` 数据结构和 `VectorStore` ABC。

**[milvus_store.py](file://src/retriever/milvus_store.py)** — Milvus 2.4+ 实现：

- Dense 检索：IVF_FLAT 索引，COSINE 度量，1024-dim
- BM25 检索：Python rank-bm25 实现，`rebuild_bm25_from_collection()` 从 Milvus 全量重建
- `hybrid_search()` — 双路检索 + Python 侧 RRF 融合（`_rrf_fuse()` 静态方法）
- `get_documents_summary()` / `get_document_chunks()` — 文档管理 CRUD 支持

**[bm25_index.py](file://src/retriever/bm25_index.py)** — Python BM25 索引器，支持构建/保存/加载/搜索，持久化到 `{DATA_DIR}/vectors/bm25_index.pkl`。

**[search.py](file://src/retriever/search.py)** — `HybridRetriever` 统一入口：

- Milvus 原生 `hybrid_search()` 优先
- BM25 不可用时自动降级 Dense-only
- `search_with_context()` 预留相邻 chunk 扩展

**[cross_ref.py](file://src/retriever/cross_ref.py)** — 3GPP 交叉引用解析：

- 正则识别：TS 38.413 / TR 38.901 / §5.2.1 / Table 7.3.1-1 / Figure 4.1-1
- 去重后对每个引用发起二次检索，补充被引文档上下文
- 递归深度限制 MAX_REF_DEPTH=2 防止无限循环

**[multi_hop.py](file://src/retriever/multi_hop.py)** — 多跳检索迭代 Agent：

- Round 1: 标准检索 → LLM 缺口分析 → 生成子查询
- Round 2-3: 并行二次检索 → 合并去重
- `needs_multi_hop()` 启发式：spec_number 多样性 < 0.25 时触发

**[online_supplement.py](file://src/retriever/online_supplement.py)** — 在线搜索补充，离线检索低分时自动触发 Google 片段 / TSpec-LLM。

**[query_quality.py](file://src/retriever/query_quality.py)** — 检索质量评估（密度、多样性、覆盖率），低质量时触发降级策略。

**[router.py](file://src/retriever/router.py)** — NN Router 重排序（Telco-RAG 架构复刻），1024-dim 嵌入 + 18-dim 结构化特征 → 18 路 Series 预测。

### 3.4 生成层 (`src/generator/`)

#### 3.4.1 RAG Pipeline 完整链路

```
用户查询 (任意语言)
    │
    ▼
┌──────────────────────────────────────────────────────────────┐
│ Step 0: 多语言处理 (i18n.py)                                   │
│   detect_language() → zh/ko → translate_to_english() → EN    │
├──────────────────────────────────────────────────────────────┤
│ Step 1: 查询扩展 (prompt.py)                                   │
│   LLM 改写查询 → 补充 3GPP 同义词/缩写                          │
├──────────────────────────────────────────────────────────────┤
│ Step 2: 嵌入生成                                               │
│   云端 API (默认) / 本地 BGE-M3 (local模式)                    │
├──────────────────────────────────────────────────────────────┤
│ Step 3: 混合检索 (HybridRetriever)                             │
│   Dense + BM25 → RRF 融合                                     │
├──────────────────────────────────────────────────────────────┤
│ Step 3.2: 多跳检索 (multi_hop.py) [条件触发]                    │
│   needs_multi_hop() → LLM 缺口分析 → 子查询 → 补充结果          │
├──────────────────────────────────────────────────────────────┤
│ Step 3.5: 交叉引用解析 (cross_ref.py)                           │
│   提取引用 → 去重 → 二次检索 → 补充上下文                        │
├──────────────────────────────────────────────────────────────┤
│ Step 3.6: 检索质量评估 (query_quality.py)                       │
│   密度/多样性/覆盖率 → 噪声过滤 → 低质量降级                    │
├──────────────────────────────────────────────────────────────┤
│ Step 3.7: Release 版本感知 (release_aware.py)                   │
│   检测 "R17 vs R18" 意图 → 按版本过滤/分组 → 对比提示词          │
├──────────────────────────────────────────────────────────────┤
│ Step 3.8: 在线搜索补充 (online_supplement.py) [可选]            │
│   离线检索分 < 阈值 → Google/TSpec-LLM 在线补充                  │
├──────────────────────────────────────────────────────────────┤
│ Step 4: RAG 提示词构造 (prompt.py)                              │
│   System + Context + Release注释 + 在线补充 → LLM              │
├──────────────────────────────────────────────────────────────┤
│ Step 5: LLM 生成 (llm_client.py)                               │
│   OpenAI 兼容 API → 流式/非流式                                 │
├──────────────────────────────────────────────────────────────┤
│ Step 5.5: 多语言回译 (i18n.py)                                  │
│   EN 答案 → translate_from_english() → 用户源语言               │
├──────────────────────────────────────────────────────────────┤
│ Step 6: 答案验证 (verifier.py)                                  │
│   Claim 提取 → 规范溯源检查 → 无支撑标注 ⚠️                     │
└──────────────────────────────────────────────────────────────┘
```

#### 3.4.2 核心模块

**[pipeline.py](file://src/generator/pipeline.py)** — `RAGPipeline` 主编排器：

- `ask(query)` — 完整 RAG 问答（12 步全链路）
- `search(query)` — 仅检索
- `_warmup()` — 预热嵌入模型避免首次查询延迟

**[llm_client.py](file://src/generator/llm_client.py)** — `LLMClient`：

- `chat(messages)` — OpenAI SDK 统一调用，`base_url + api_key + model` 三参数切换
- `embed(texts)` — 根据 `embedding_provider` 自动选择云端 API 或本地 BGE 模型

**[i18n.py](file://src/generator/i18n.py)** — 多语言支持：

- `detect_language()` — 字符集检测（CJK → zh, Hangul → ko, 其他 → en）
- `translate_to_english()` — LLM 查询翻译（保留 3GPP 术语）
- `translate_from_english()` — LLM 回答回译

**[release_aware.py](file://src/generator/release_aware.py)** — Release 版本感知：

- `detect_release_intent()` — 正则匹配 "R17 vs R18" / "Release 18"
- `filter_by_release()` / `group_by_release()` — 按版本过滤/分组
- `build_release_context()` — 构建版本感知上下文和提示词

**[verifier.py](file://src/generator/verifier.py)** — 幻觉验证：

- Claim 提取（TS 编号正则 + LLM 结构化输出）
- 支撑校验（语义相似度 + 关键词交叉验证）
- 无支撑比例 > 阈值时追加全局警告

**[prompt.py](file://src/generator/prompt.py)** — 提示词模板：

- `build_rag_prompt()` — 层级上下文注入 + Release 注释 + 在线上下文
- `build_query_expansion_prompt()` — 查询改写/扩展

### 3.5 API 服务层 (`src/api/`)

#### REST API 端点

| 端点                            | 方法   | 说明                            |
| ------------------------------- | ------ | ------------------------------- |
| `/api/v1/health`                | GET    | 服务健康检查 + chunk 统计       |
| `/api/v1/search`                | POST   | Dense+BM25 混合检索             |
| `/api/v1/search/count`          | POST   | 检索结果计数                    |
| `/api/v1/search/batch`          | POST   | 批量并行检索（最多 10 条）      |
| `/api/v1/ask`                   | POST   | RAG 问答（含来源+验证）         |
| `/api/v1/ask/stream`            | POST   | SSE 流式生成（逐 token 推送）   |
| `/api/v1/documents`             | GET    | 文档列表（分页+筛选）           |
| `/api/v1/documents/{id}`        | GET    | 文档详情                        |
| `/api/v1/documents/{id}/chunks` | GET    | 文档 Chunk 列表                 |
| `/api/v1/documents/{id}`        | DELETE | 删除文档                        |
| `/api/v1/stats`                 | GET    | 系统统计（Release/Series 分布） |
| `/api/v1/feedback`              | POST   | 提交 👍/👎 用户反馈             |
| `/api/v1/feedback`              | GET    | 反馈列表查询（分页+筛选）       |
| `/api/v1/feedback/stats`        | GET    | 反馈汇总统计（好评率）          |
| `/metrics`                      | GET    | Prometheus 指标                 |

#### 中间件栈

```
Request → CORS → RequestLogging → Prometheus → APIKey → Router → Response
```

**[router.py](file://src/api/rest/router.py)** — 核心 REST 端点实现。

**[admin_router.py](file://src/api/rest/admin_router.py)** — 管理端点（摄入触发、状态查询）。

**[feedback.py](file://src/api/rest/feedback.py)** — 用户 👍/👎 反馈收集。

**[schemas.py](file://src/api/rest/schemas.py)** — Pydantic 请求/响应模型。

#### 监控与可观测性

**[monitoring.py](file://src/utils/monitoring.py)** — Prometheus 指标：

- `record_search()` / `record_ask()` / `record_llm_call()` — 延迟/调用量/错误率
- `record_multi_hop()` / `record_error()` — 高级检索/异常计数

### 3.6 前端层 (`frontend/`)

前端采用 **DeepSeek 式对话界面** 设计，与搜索引擎式布局有本质区别：

**核心设计**：

- **对话式消息流** — 用户消息蓝色气泡居右，助手回答居左，Markdown 富文本渲染（表格/代码块/列表）
- **底部输入栏** — 固定于页面底部，支持 Enter 发送、Shift+Enter 换行
- **流式回答** — SSE 逐 token 推送，实时 Markdown 渲染
- **精排开关** — 输入栏左侧 🎯/⚡ 按钮，用户可选择启用/关闭 Cross-Encoder Reranker（质量 vs 速度）
- **检索溯源隐藏** — 引用依据折叠在每条回答下方，点击展开查看源规范/章节/分数
- **对话历史侧边栏** — 固定浮层（z-40），遮罩背景，滑入动画

**页面路由**：

| 路由                    | 页面           | 说明                                      |
| ----------------------- | -------------- | ----------------------------------------- |
| `/`                     | HomePage       | 对话式主界面（Hero → 消息流）             |
| `/admin`                | AdminPage      | 管理仪表盘（统计 + 快速操作）             |
| `/admin/search`         | SearchTestPage | 检索测试台（检索结果 + LLM 回答双栏对比） |
| `/admin/documents`      | DocListPage    | 文档管理（筛选/分页/删除）                |
| `/admin/documents/[id]` | DocDetailPage  | 文档详情（元数据 + Chunk 列表）           |
| `/documents`            | DocBrowserPage | 公开文档浏览（只读）                      |

**技术栈**：

- Next.js 16.2 + React 19.2 + TypeScript
- Tailwind CSS 4 + 明暗主题（CSS 变量 + next-themes 风格 context）
- react-markdown v10 — 助手回答富文本渲染
- SSE (Server-Sent Events) — `/api/v1/ask/stream` 流式生成
- localStorage — 对话历史持久化

**关键文件**：

```
frontend/
├── app/
│   ├── page.tsx                     ← 对话主界面 (HomePage)
│   ├── layout.tsx                   ← 根布局 (Nav + ThemeProvider)
│   ├── globals.css                  ← 全局样式 + 主题变量
│   ├── documents/page.tsx           ← 公开文档浏览器
│   └── admin/
│       ├── page.tsx                 ← 管理仪表盘
│       ├── search/page.tsx          ← 检索测试台 (双栏对比)
│       └── documents/[id]/page.tsx  ← 文档详情 + Chunk 列表
├── components/
│   └── ThemeToggle.tsx              ← 明暗主题切换按钮
└── lib/
    ├── api.ts                       ← REST API + SSE 客户端封装
    ├── theme.tsx                    ← ThemeProvider (React Context)
    └── useConversationHistory.ts    ← localStorage 对话历史 Hook
```

### 3.7 工具层 (`src/utils/`)

**[helpers.py](file://src/utils/helpers.py)** — 硬件检测 + 平台适配：

- `detect_platform()` — 检测 NVIDIA / Apple Silicon / 其他
- `get_embedding_device_config()` — 返回设备推荐配置

**[monitoring.py](file://src/utils/monitoring.py)** — Prometheus 指标（见 3.5）。

**Grafana Dashboard** — `deploy/grafana-dashboard.json`：9 面板覆盖请求速率、检索延迟、LLM Token 用量、错误率。导入 Grafana 后配合 Prometheus `/metrics` 端点即可使用。

**[auth.py](file://src/api/auth.py)** — API Key 认证中间件。

#### MCP 工具

**[mcp/tools.py](file://src/api/mcp/tools.py)** + **[mcp/server.py](file://src/api/mcp/server.py)** — 暴露 4 个 MCP 工具供 AI Agent 调用：

- `search_specifications` — 规范检索
- `get_specification_details` — 规范详情
- `ask_3gpp_expert` — RAG 专家问答
- `list_releases` — 可用 Release 列表

### 3.6 工具层 (`src/utils/`)

**[helpers.py](file://src/utils/helpers.py)** — 硬件检测：

- `get_hardware_info()` — 识别 OS、CPU 架构、GPU 类型、统一内存
- `get_best_device()` — 自动选择最优嵌入设备（NVIDIA→cuda, Apple Silicon→cpu 安全降级）

**[monitoring.py](file://src/utils/monitoring.py)** — Prometheus 指标导出：

- `record_search()` / `record_ask()` / `record_llm_call()` — 延迟/调用量/错误率
- `record_multi_hop()` / `record_error()` — 高级检索/异常计数

---

## 4. 数据流

### 4.1 问答请求完整链路

```
用户 POST /api/v1/ask {"query": "5G NR PDU Session 建立流程"}
    │
    ▼
FastAPI router → get_pipeline() → RAGPipeline.ask()
    │
    ├─ [i18n] detect_language() → "zh" → translate_to_english() → EN query
    ├─ [expand] LLM 查询扩展 → 补充 3GPP 同义词
    ├─ [embed] LLMClient.embed() → 1024-dim float32
    ├─ [search] HybridRetriever.search() → Milvus Dense + BM25 → RRF → Top-10
    ├─ [multi-hop] needs_multi_hop()? → LLM 缺口分析 → 子查询检索
    ├─ [cross-ref] extract_references() → 二次检索补充 → 合并
    ├─ [quality] evaluate_quality() → filter_noise()
    ├─ [release] detect_release_intent() → 版本过滤/分组
    ├─ [online] should_supplement()? → Google/TSpec-LLM 补充
    ├─ [prompt] build_rag_prompt() → System + Context + Release注释
    ├─ [llm] LLMClient.chat() → 流式/非流式
    ├─ [i18n] translate_from_english() → 中文回答
    └─ [verify] AnswerVerifier.verify() → 溯源检查 + 警告

    ▼
AskResponse { answer, sources[10], verified, coverage, warnings }
```

### 4.2 摄入数据流

```
DOCX 文件 (data/documents/R18/)
    │
    ▼
┌──────────────────────────────────────┐
│ Extractor (Pandoc)                  │
│   .docx → .md (保留表格/列表/公式)     │
├──────────────────────────────────────┤
│ Splitter (HeaderAwareSplitter)        │
│   三层自适应:                           │
│   ① 标题层级切分 (默认)                  │
│   ② Grid Table 行组/HTML<tr>/换行拆分    │
│      (chunk > 55KB 触发)               │
│   ③ _safe_truncate_bytes 语义截断 (兜底) │
│   每个 chunk 注入: spec_id, series,   │
│   release, section_id, chunk_index   │
├──────────────────────────────────────┤
│ Embedder (BatchEmbedder)              │
│   BGE-M3 / OpenAI API → 1024-dim     │
├──────────────────────────────────────┤
│ MilvusStore.insert()                  │
│   Dense 向量 + 元数据写入             │
├──────────────────────────────────────┤
│ BM25 索引重建                         │
│   rebuild_bm25_from_collection()     │
│   → {DATA_DIR}/vectors/bm25_index.pkl  │
└──────────────────────────────────────┘
```

---

## 5. 关键设计决策

### 5.1 为什么 Dense + BM25 混合检索是默认策略

3GPP 规范文本密度极高，不同章节共享大量相同术语。纯语义检索会产生系统性误匹配——"词像意不像"。BM25 提供的关键词锚点是防止语义漂移的最后一道防线。RRF 融合不是可选的增强，是默认检索策略。

### 5.2 为什么用 Python BM25 而非 Milvus 原生 Sparse Vector

Milvus 2.4+ 的原生 BM25（基于 Tantivy）需要 Sparse Vector 字段 + BM25 函数索引，在 ARM64 Docker 镜像中支持不完整。Python rank-bm25 实现更可控，与 Dense 检索统一在 Python 侧 RRF 融合，架构更简洁。

**待迁移**：BGE-M3 原生输出学习到的词法稀疏向量（与 dense 联合训练，精度高于统计 BM25）。迁移后 Python BM25 索引器可移除，sparse 信号直接存入 Milvus `SPARSE_FLOAT_VECTOR` 字段，由 Milvus 原生 `hybrid_search()` 一笔查询完成 Dense + Sparse 双路检索。

### 5.3 为什么 Apple Silicon 用 spawn + workers=1

Apple 明确：`fork()` 后使用 Metal/GPU 框架是 undefined behavior。即使父进程仅在 CPU 上 import torch，也会初始化 Metal 内部符号。`spawn` 创建全新 OS 进程，独立加载模型，安全无副作用。

**workers=1 的硬约束**：Apple MPS 不支持多进程同时使用 GPU。多进程同时调用 `[MTLCommandBuffer waitUntilCompleted]` 会触发 Metal GPU 调度器死锁——所有 worker 永久阻塞在 `__psynch_cvwait`。这是 Apple Metal 框架层面的限制，与 Python multiprocessing 实现无关。单 worker 已足够：GPU 内部高度并行，M4 Max ~75 t/s（3GPP 长文本），约 24 分钟可完成 108K chunks。

### 5.4 为什么 LLM 统一走 OpenAI 兼容 API

`base_url + api_key + model` 三参数覆盖公司自建 API / OpenAI / Azure / Ollama。只要目标 API 兼容 `/v1/chat/completions`，零代码修改，仅改变量。

### 5.5 为什么选 BGE-M3

BGE-M3 在同一 1024-dim 向量空间支持 100+ 语言。中文查询可直接检索英文 3GPP 规范，跨语言相似度达 0.857，无需 LLM 查询翻译。同时输出稠密+稀疏双向量，为后续 Milvus 原生 hybrid_search 迁移提供基础。

### 5.6 为什么不用 FAISS

Milvus 2.4+ 原生支持 Dense + BM25 双路检索，FAISS 不支持 BM25。在 3GPP 领域 BM25 是硬需求，FAISS-only 验证出来的 Dense-only 结果本身就是精度缺陷的，没有验证意义。

---

## 6. 部署架构

### Docker Compose 服务拓扑

```
                    ┌──────────────┐
                    │   Frontend   │  Next.js :3000
                    └──────┬───────┘
                           │
                    ┌──────┴───────┐
                    │   API        │  FastAPI :8000
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │  Milvus  │ │  MinIO   │ │   etcd   │
        │  :19530  │ │ :9000/01 │ │  :2379   │
        └──────────┘ └──────────┘ └──────────┘
```

### 支持的部署平台

| 平台                            | 嵌入设备                     | Docker 架构   |
| ------------------------------- | ---------------------------- | ------------- |
| Linux x86_64 (Intel/AMD)        | `cpu`                        | `linux/amd64` |
| NVIDIA GB10 (ARM64 + Blackwell) | `auto` → `cuda`              | `linux/arm64` |
| macOS Apple Silicon             | `mps` (spawn) / `cpu` (安全) | 本地开发      |

### 离线部署

完整的离线部署方案参见 **[docs/deployment/offline-deployment.md](../deployment/offline-deployment.md)**，支持外网制备 → U 盘/网盘传输 → 内网一键安装。

---

## 7. 附录：设计遗产

本系统的架构设计参考了 5 个开源 3GPP RAG 项目的核心优势：

| 参考项目            | 融入本项目的关键设计                                               |
| ------------------- | ------------------------------------------------------------------ |
| **SpecPilot**       | Docling DOCX→MD 双转换链，按 Release×Series 二维矩阵组织输出       |
| **gpp-RAG-app**     | ETL 流水线（4 步）设计，Milvus 集成模式，Docker Compose 部署       |
| **Telco-RAG**       | 双阶段检索（Dense + NN Router），在线/离线混合，Next.js 前端架构   |
| **Chat3GPP**        | RRF 融合算法（`1/(k+rank+1)`），LangChain RAG 集成，提示词构造模式 |
| **3GPP MCP Server** | MCP 工具设计模式，智能缓存策略（<500ms），TSpec-LLM 对接           |

> **相关文档**：[硬件兼容性指南](./hardware-compatibility.md) | [离线部署手册](../deployment/offline-deployment.md)
