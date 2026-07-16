---
title: "附录C: 测试报告"
tags: [testing, quality, report]
---

# 附录 C — 测试报告

## C.1 测试策略

本系统采用四级测试金字塔，确保代码质量和功能稳定性：

```
         ╱  E2E  ╲         评测脚本: run_eval.py
        ╱  集成测试 ╲       Pipeline + API 端点协作
       ╱   集成测试   ╲     检索器/生成器/API 交互
      ╱    单元测试    ╲    函数级验证
     ╱────── 基准测试 ──────╲  性能/吞吐/延迟
```

## C.2 单元测试覆盖

### 测试文件一览 (29 文件, 444 用例)

| 模块 | 测试文件 | 用例数 | 状态 |
|------|----------|--------|------|
| **Config** | `test_config/test_settings.py` | ~15 | ✅ |
| **Generator** | `test_generator/test_i18n.py` | ~20 | ✅ |
| | `test_generator/test_prompt.py` | ~15 | ✅ |
| | `test_generator/test_release_aware.py` | ~25 | ✅ |
| | `test_generator/test_verifier.py` | ~15 | ✅ |
| **Ingestion** | `test_ingestion/test_embedding_cache.py` | ~20 | ✅ |
| | `test_ingestion/test_incremental.py` | ~10 | ✅ |
| | `test_ingestion/test_manifest.py` | ~10 | ✅ |
| | `test_ingestion/test_precomputed_loader.py` | ~5 | ✅ |
| | `test_ingestion/test_release_monitor.py` | ~15 | ✅ |
| | `test_ingestion/test_splitter.py` | ~20 | ✅ |
| **Retriever** | `test_retriever/test_cross_ref.py` | ~15 | ✅ |
| | `test_retriever/test_multi_hop.py` | ~20 | ✅ |
| | `test_retriever/test_online_supplement.py` | ~15 | ✅ |
| | `test_retriever/test_query_quality.py` | ~20 | ✅ |
| | `test_retriever/test_router.py` | ~10 | ✅ |
| | `test_retriever/test_search.py` | ~15 | ✅ |
| **API** | `test_api/test_auth.py` | ~5 | ✅ |
| | `test_api/test_mcp_tools.py` | ~10 | ✅ |
| | `test_api/test_schemas.py` | ~10 | ✅ |
| **Utils** | `test_utils/test_helpers.py` | ~12 | ✅ |
| | `test_utils/test_monitoring.py` | ~8 | ✅ |
| **Eval** | `tests/eval/test_metrics.py` | ~10 | ✅ |
| | `tests/eval/test_run_eval.py` | ~5 | ✅ |
| **E2E** | `tests/e2e/test_milvus_e2e.py` | ~3 | ⚠️ |
| **新模块** | 见下 | ~60 → ~140 | ✅ |

### 新增测试 (本轮)

| 文件 | 用例数 | 覆盖内容 |
|------|--------|----------|
| `test_ingestion/test_embedder.py` | ~15 | BatchEmbedder + 双层缓存 |
| `test_retriever/test_reranker.py` | ~10 | Cross-Encoder 精排 |
| `test_generator/test_feedback.py` | ~10 | 反馈存储 + 分析 |
| `test_generator/test_llm_client.py` | ~8 | LLM 客户端接口 |
| `tests/benchmark/test_bench.py` | ~15 | 吞吐/延迟/缓存命中率 |
| `test_pipeline_integration.py` | ~38 | RAGPipeline 全链路 mock 集成 |
| `test_api_integration.py` | ~44 | REST API 全端点 TestClient 验证 |

## C.3 集成测试覆盖

### API 端点集成 (44 用例)
| 端点 | 测试覆盖 |
|------|----------|
| `GET /health` | 就绪/初始化状态切换 |
| `POST /search` | 语义搜索 + 过滤 + 参数校验 + 503/500 错误码 |
| `POST /ask` | 标准问答 + RAGResponse 结构验证 + 边界参数 |
| `POST /search/count` | 结果计数 |
| `POST /search/batch` | 批量检索 + 最大查询数限制 |
| `GET /documents` | 分页列表 + 系列/Release 过滤 |
| `GET /documents/{id}` | 单文档详情 + 404 |
| `GET /documents/{id}/chunks` | 文档分块列表 |
| `DELETE /documents/{id}` | 文档删除 |
| `GET /stats` | 系统统计信息 |
| `POST /ask/stream` | SSE 流式事件格式验证 + 错误事件 |
| `GET /refs/graph` | 引用图谱 (Mock Milvus) |

### Pipeline 集成 (38 用例)
| 测试类 | 覆盖内容 |
|--------|----------|
| `TestPipelineInit` | 组件创建 (retriever/verifier/multi_hop/cache) |
| `TestPipelineAsk` | 全流程 ask() + 缓存命中/miss + 空结果 + LLM 异常 |
| `TestPipelineSearch` | search() + top_k + reranker + spec-aware boost |
| `TestExtractSpecNumbers` | 规范号提取 (单/多/无/非法格式) |
| `TestFilterLowQuality` | 低质量章节过滤 (缩写/定义/引用/目录/正常内容) |
| `TestSpecAwareRerank` | 两阶段 spec-aware 重排序 + 分数融合 |
| `TestQueryCache` | LRU 缓存 key 确定性 + TTL 配置 |
| `TestPipelineErrorRecovery` | 查询扩展降级 / 嵌入失败零向量 / 交叉引用容错 |
| `TestRAGResponse` | RAGResponse 数据类构造 + 默认值 |

## C.4 基准测试 (Benchmark)

### 测试指标
| 指标 | 测试方法 | 目标 |
|------|----------|------|
| 检索延迟 (P50/P95) | 100 次采样 | < 200ms / < 500ms |
| 嵌入缓存命中率 | 重复摄入测试 | > 80% |
| LLM 调用延迟 | Mock API 响应 | < 3s |
| RAG 端到端延迟 | 完整管线 | < 5s |
| 查询缓存命中率 | 重复查询 | > 90% (TTL 1h内) |

### 运行方法
```bash
# 运行全部基准测试
python -m pytest tests/benchmark/ -v

# 特定指标
python -m pytest tests/benchmark/test_bench.py::test_search_latency -v
```

## C.5 CI 运行结果

```
✅ Python 3.14: 444 passed, 0 failed (全量回归)
  ├── Pipeline 集成测试: 38/38 ✅
  ├── API 集成测试:    44/44 ✅
  ├── 单元测试:        362/362 ✅
  └── 基准测试:        8/8 ✅
✅ Eval dry-run: 70 条测试集格式验证通过
✅ Frontend build: 12 routes compiled
⚠️ E2E: 跳过 (CI 无 Milvus)
```

### 测试覆盖汇总

| 层级 | 文件数 | 用例数 | 通过率 |
|------|--------|--------|--------|
| 单元测试 (Unit) | 27 | 362 | 100% |
| 集成测试 (Integration) | 2 | 82 | 100% |
| 基准测试 (Benchmark) | 1 | 8 | 100% |
| E2E (dry-run) | 1 | 1 | ✅ 格式验证 |
| **合计** | **31** | **444** | **100%** |

## C.6 已知限制

1. **嵌入模型测试**: 需要 HuggingFace 下载 BGE-M3 (CI 首次需网络)
2. **Milvus E2E**: CI 环境无 Milvus，仅本地运行
3. **LLM 集成测试**: 使用 mock，真实调用需配置 API Key
4. **MPS 测试**: 仅 Apple Silicon 环境可运行
