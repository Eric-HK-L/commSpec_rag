---
title: "附录A: 配置参考"
tags: [config, env, reference]
---

# 附录 A — 配置参考

## A.1 环境变量完整列表

所有配置项均通过 `.env` 文件或环境变量设置，由 `pydantic-settings` 自动加载。

### LLM 配置
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LLM_BASE_URL` | `https://api.openai.com/v1` | LLM API 端点 (兼容 OpenAI 格式) |
| `LLM_API_KEY` | `sk-your-key-here` | API 密钥 |
| `LLM_MODEL` | `gpt-4o-mini` | 模型名称 |
| `LLM_TEMPERATURE` | `0.0` | 生成温度 (0=确定性) |
| `LLM_MAX_TOKENS` | `2048` | 最大输出 token |
| `LLM_TIMEOUT` | `60.0` | API 超时秒数 |

### 嵌入模型
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `EMBEDDING_PROVIDER` | `local` | `local` (BGE-M3) 或 `api` (OpenAI) |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | API 模式的云端模型 |
| `EMBEDDING_DIMENSION` | `1024` | 嵌入向量维度 |
| `EMBEDDING_DEVICE` | `auto` | `auto`/`cuda`/`mps`/`cpu` |
| `LOCAL_EMBEDDING_MODEL` | `BAAI/bge-m3` | 本地模型 HuggingFace ID |

### Cross-Encoder 精排
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RERANKER_ENABLED` | `true` | 是否启用第二阶段精排 |
| `RERANKER_MODEL` | `BAAI/bge-reranker-v2-m3` | 精排模型 |
| `RERANKER_TOP_K` | `100` | 送入 reranker 的候选数 |
| `RERANKER_DEVICE` | `auto` | 运行设备 |

### 向量数据库
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VECTOR_DB` | `milvus` | 向量数据库类型 |
| `MILVUS_HOST` | `localhost` | Milvus 主机地址 |
| `MILVUS_PORT` | `19530` | Milvus gRPC 端口 |
| `MILVUS_COLLECTION_NAME` | `TeleComm_specs` | Collection 名称 |

### 文档处理
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DOCUMENTS_DIR` | `data/documents` | 文档源目录 |
| `DATA_DIR` | `data` | 数据根目录 |

### 摄入分块 (Phase 5 新增 — INGESTION__ 前缀)

以下配置仅重摄入 (`bulk_ingest.py`) 时生效，修改后必须重新执行摄入。

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `INGESTION__CHUNK_MODE` | `dynamic` | `fixed` (段落累计) / `dynamic` (表格/正文分离) |
| `INGESTION__CHUNK_SIZE` | `1024` | fixed 模式的字符上限 (fallback) |
| `INGESTION__CHUNK_OVERLAP` | `100` | 分块重叠字符数 |
| `INGESTION__TABLE_MAX_CHARS` | `5000` | dynamic 模式下表格 chunk 上限 |
| `INGESTION__PROSE_MAX_CHARS` | `1500` | dynamic 模式下纯文本 chunk 上限 |
| `INGESTION__MAX_CHUNK_CHARS` | `8000` | BGE-M3 8192 token 安全上限 |
| `INGESTION__BATCH_SIZE` | `64` | Milvus 写入批大小 |

### 检索
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `MAX_SEARCH_RESULTS` | `20` | 最终返回结果数 |
| `DENSE_TOP_K` | `100` | Dense 检索候选数 |
| `BM25_TOP_K` | `100` | BM25 检索候选数 |
| `SIMILARITY_THRESHOLD` | `0.7` | 相似度阈值 |
| `ENABLE_NN_ROUTER` | `false` | 启用 NN 路由器 |
| `ENABLE_ONLINE_SEARCH` | `false` | 启用在线搜索补充 |
| `GOOGLE_API_KEY` | `` | Google Custom Search API Key |
| `GOOGLE_CSE_ID` | `` | Google CSE ID |
| `TSPEC_LLM_URL` | `` | TSpec-LLM API 端点 |
| `ONLINE_SCORE_THRESHOLD` | `0.6` | 离线分低于此值触发在线补充 |

### Release 监控
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `RELEASE_MONITOR_INTERVAL_MINUTES` | `120` | 文档变更检测间隔 (0=禁用) |

### API 服务
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `API_HOST` | `0.0.0.0` | 监听地址 |
| `API_PORT` | `8000` | 监听端口 |
| `API_WORKERS` | `1` | Uvicorn worker 数 |

### 日志
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `LOG_LEVEL` | `INFO` | `DEBUG`/`INFO`/`WARNING`/`ERROR` |
| `LOG_FILE` | `logs/app.log` | 日志文件路径 |

### MPS 内存控制 (Apple Silicon)
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `PYTORCH_MPS_HIGH_WATERMARK_RATIO` | `0.5` | MPS 内存上限比例 |
| `PYTORCH_MPS_LOW_WATERMARK_RATIO` | `0.3` | MPS 内存下限比例 |

## A.2 典型配置场景

### 场景 1: 开发环境 (OpenAI + 本地 Milvus)
```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=gpt-4o-mini
EMBEDDING_PROVIDER=local
MILVUS_HOST=localhost
LOG_LEVEL=DEBUG
```

### 场景 2: 离线生产环境 (vLLM + GPU)
```bash
LLM_BASE_URL=http://gpu-server:8000/v1
LLM_API_KEY=not-needed
LLM_MODEL=Qwen2.5-7B-Instruct
EMBEDDING_PROVIDER=local
EMBEDDING_DEVICE=cuda
MILVUS_HOST=milvus-prod
LOG_LEVEL=INFO
```

### 场景 3: 完整功能 (含在线搜索)
```bash
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-xxx
ENABLE_ONLINE_SEARCH=true
GOOGLE_API_KEY=xxx
GOOGLE_CSE_ID=xxx
TSPEC_LLM_URL=https://tspec-llm.3gpp.org/query
RELEASE_MONITOR_INTERVAL_MINUTES=60
```

## A.3 配置读取逻辑

```python
# src/config/settings.py
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",              # 自动从 .env 加载
        env_file_encoding="utf-8",
        extra="ignore",               # 忽略未定义变量
    )

settings = Settings()  # 全局单例, 惰性加载
```

`settings.embedding_device = "auto"` 的自动检测逻辑:
- NVIDIA GPU 可用 → `cuda`
- Apple Silicon (MPS) → `cpu` (安全默认, 避免 MPS 内存泄漏)
- 其他 → `cpu`

可通过设置 `EMBEDDING_DEVICE=mps` 强制执行 MPS 加速。

## A.4 配置项分类（必填 vs 可选）

### 🔴 必填项（不填系统无法正常工作）

| 变量 | 说明 | 不填的后果 |
|------|------|-----------|
| `LLM_BASE_URL` | LLM API 端点 | RAG 问答功能不可用 |
| `LLM_API_KEY` | API 密钥 | LLM 调用返回 401 |
| `LLM_MODEL` | 模型名称 | 使用默认 gpt-4o-mini（可能不可用） |

### 🟡 建议配置（使用推荐默认值即可，但建议按环境调整）

| 变量 | 默认值 | 建议调整 |
|------|--------|----------|
| `MILVUS_HOST` | `localhost` | 生产环境改为实际服务器地址 |
| `LOG_LEVEL` | `INFO` | 开发用 DEBUG，生产用 INFO |
| `EMBEDDING_DEVICE` | `auto` | 有 GPU 设 `cuda`，Mac 可设 `mps` |
| `RERANKER_ENABLED` | `true` | CPU 环境可设 `false` 加快速度 |

### 🟢 可选（仅特定功能需要）

| 变量 | 用途 | 何时需要 |
|------|------|----------|
| `ENABLE_ONLINE_SEARCH` | 在线搜索补充 | 需要在线搜索结果时 |
| `GOOGLE_API_KEY` | Google 搜索 | 使用 Google CSE 在线补充时 |
| `GOOGLE_CSE_ID` | Google CSE | 同上 |
| `TSPEC_LLM_URL` | TSpec-LLM | 使用 3GPP 官方 RAG 在线补充时 |
| `RELEASE_MONITOR_INTERVAL_MINUTES` | Release 监控 | 需要自动检测文档变更时（0=禁用） |
| `API_WORKERS` | Worker 数 | 高并发场景 |

## A.5 常见配置错误

| 错误 | 症状 | 正确写法 |
|------|------|----------|
| `LLM_BASE_URL` 末尾多了 `/v1/v1` | 404 Not Found | `https://api.openai.com/v1`（不要重复 /v1） |
| `LLM_MODEL` 名称错误 | Model not found | 确认模型实际名称（如 `gpt-4o-mini` 而非 `gpt4o-mini`） |
| 布尔值用大写 | `True` `False` 不生效 | 用全小写 `true` / `false` |
| `MILVUS_HOST` 写错 | Connection refused | 确认 Milvus 实际 IP/域名 |
| `EMBEDDING_DEVICE=mps` 但非 Mac | RuntimeError: MPS not available | 设为 `cpu` 或 `cuda` |
| `DATA_DIR` 使用了相对路径 | 摄入找不到文档 | 用绝对路径或确认相对路径基于项目根目录 |
| `.env` 中引号包裹值 | 值包含引号字符 | 不要用引号: `KEY=value` 而非 `KEY="value"` |
| 忘记设置 `HF_ENDPOINT`（国内） | HuggingFace 连接超时 | `HF_ENDPOINT=https://hf-mirror.com` |

## A.6 配置优先级

系统加载配置的优先级（由高到低）:

```
1. 系统环境变量 (export LLM_MODEL=xxx)
2. .env 文件
3. pydantic-settings 默认值
```

> 如果同时在 Shell 和 `.env` 中设置了同一变量，Shell 环境变量优先。排查配置问题时先 `env | grep LLM` 检查。
