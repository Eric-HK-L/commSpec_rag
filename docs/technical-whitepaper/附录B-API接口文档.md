---
title: "附录B: API 接口文档"
tags: [api, rest, endpoints, reference]
---

# 附录 B — API 接口文档

## B.1 通用说明

- **Base URL**: `http://localhost:8000`
- **格式**: JSON (request/response)
- **认证**: API Key (可选, 通过 `X-API-Key` Header)
- **日期格式**: ISO 8601 (UTC)

### 通用响应格式
```json
{
  "success": true,
  "data": { ... },
  "error": null,
  "total": 0
}
```

## B.2 Search API

### 语义搜索
```
GET /api/v1/search?q={query}&top_k=20
```
**响应**: `APIResponse[SearchResponse]`
```json
{
  "success": true,
  "data": {
    "query": "PDU Session Establishment",
    "results": [
      {
        "chunk_id": "abc123",
        "text": "...",
        "score": 0.92,
        "spec_number": "38.413",
        "release": "R18",
        "parent_section_id": "8.3.1",
        "parent_title": "PDU Session Resource Setup"
      }
    ],
    "total": 20
  }
}
```

## B.3 Ask API (RAG 问答)

### 标准问答
```
POST /api/v1/ask
Content-Type: application/json
{
  "query": "What is PDU Session Establishment?",
  "top_k": 20,
  "reranker_enabled": true
}
```
**响应**: `APIResponse[AskResponse]`
```json
{
  "success": true,
  "data": {
    "query": "What is PDU Session Establishment?",
    "answer": "PDU Session Establishment is...",
    "sources": [{ ... }],
    "verified": true,
    "warnings": [],
    "coverage": 0.85,
    "expanded_query": "PDU Session Resource Setup NGAP NAS SMF..."
  }
}
```

### 流式问答 (SSE)
```
POST /api/v1/ask/stream
Content-Type: application/json
{ "query": "...", "top_k": 20 }
```
**响应**: `text/event-stream`
```
data: {"type": "start"}
data: {"type": "token", "content": "PDU"}
data: {"type": "token", "content": " Session"}
...
data: {"type": "sources", "sources": [...]}
data: {"type": "done"}
```

## B.4 Documents API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/documents?offset=0&limit=100` | 文档列表 |
| GET | `/api/v1/documents/{doc_id}` | 文档详情 |
| DELETE | `/api/v1/documents/{doc_id}` | 删除文档 |
| GET | `/api/v1/documents/specs` | 规范编号列表 |

## B.5 Admin API

详见 [[10-管理控制台#10.3 Admin API 端点]]

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/admin/stats` | 系统统计 |
| GET | `/api/v1/admin/manifest` | Manifest 清单 |
| DELETE | `/api/v1/admin/manifest/{key}` | 删除 manifest 条目 |
| POST | `/api/v1/admin/ingest/trigger` | 触发摄入 |
| GET | `/api/v1/admin/ingest/status` | 摄入状态 |
| GET | `/api/v1/admin/logs` | 系统日志 |
| GET | `/api/v1/admin/system` | 系统健康 |
| GET | `/api/v1/admin/config` | 配置视图 |

## B.6 Feedback API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/v1/feedback` | 提交反馈 |
| GET | `/api/v1/feedback?offset=0&limit=20` | 反馈列表 |
| GET | `/api/v1/feedback/stats` | 反馈统计 |

## B.7 MCP API (Model Context Protocol)

端点路径: `/mcp/*`

为 AI 编程助手提供标准化的工具调用接口，支持:
- `search_3gpp` — 语义搜索
- `ask_3gpp` — RAG 问答
- `list_documents` — 文档列表

## B.8 System API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/metrics` | Prometheus 指标 |

### 健康检查响应
```json
{
  "status": "ok",
  "milvus": true,
  "uptime_seconds": 12345
}
```

## B.9 错误码

| HTTP Status | 含义 | 常见原因 |
|-------------|------|----------|
| 200 | 成功 | - |
| 400 | 请求参数错误 | 缺少必填参数 |
| 401 | 未授权 | API Key 无效 |
| 404 | 资源不存在 | doc_id 未找到 |
| 422 | 请求体验证失败 | JSON 格式错误 |
| 500 | 服务器内部错误 | LLM 超时 / Milvus 异常 |
| 503 | 服务不可用 | Milvus 未连接 |
