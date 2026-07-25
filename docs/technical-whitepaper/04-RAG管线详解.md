---
title: "04: RAG 管线详解"
tags: [rag, pipeline, query, retrieval, generation]
---

# 04 — RAG 管线详解

## 4.1 管线总览

RAG Pipeline 是系统的核心编排引擎，位于 `src/generator/pipeline.py`。它将检索、生成、验证三大阶段串联为一个完整的问答流程。

**入口类**: `RAGPipeline`
**核心方法**: `ask(query, reranker_enabled=True) → RAGResponse`

## 4.2 完整执行流程

### Step 0: 查询缓存检查
```python
cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
cached = self._query_cache.get(cache_key)  # TTLCache(maxsize=256, ttl=3600)
```
- 相同查询 1 小时内直接返回缓存结果
- 使用 `cachetools.TTLCache` 实现，LRU 淘汰策略

### Step 1: 多语言处理
```python
query_lang = detect_language(query)        # → "zh"/"en"/"ja"/...
search_query = translate_to_english(query, query_lang, self._llm)
```
- 调用 LLM 进行翻译（非英文 → 英文）
- 英文查询直接跳过
- 参见 [[06-多语言与国际化]]

### Step 2: 查询扩展
```python
expanded_query = self._expand_query(search_query)
# 调用 LLM: build_query_expansion_prompt(query) → 扩展关键词
```
- LLM 将简短查询扩展为包含同义词/相关术语的关键词串
- 提高召回率，尤其在电信领域术语变体多的情况下
- 失败降级：返回原始查询

### Step 3: 混合检索
```python
# 候选池大小自适应
search_top_k = settings.reranker_top_k if reranker_enabled else settings.max_search_results

results = self._retriever.search(expanded_query, query_embedding)
# Dense: Milvus ANN (IVF_FLAT, metric=COSINE)
# BM25:  本地 pickle 索引, TF-IDF 加权
# RRF 融合: score = Σ 1/(k + rank_i) for each retriever i
```

**RRF 融合策略** (Reciprocal Rank Fusion):
- 对每个文档，计算其在 Dense 和 BM25 列表中的排名
- 融合得分 = 1/(60+dense_rank) + 1/(60+bm25_rank)
- k=60 的经验值，可调

### Step 3.1: 分类列举查询 — 多角度分解检索 (Phase 5 新增)
```python
if _is_taxonomy_query(search_query):
    results = self._taxonomy_decompose_search(search_query, query_embedding, results)
```
- 触发条件：查询含 "格式/分类/种类/列出/汇总" 等关键词
- LLM 将问题分解为 3-5 个子查询（每个覆盖一个类别维度）
- 子查询并行搜索，结果合并去重（原始结果排前，子查询补充追加后）
- 确保分类列举类问题覆盖所有维度，不遗漏任何格式/类型

### Step 3.1b: 元数据加权 boost (Phase 5 新增)
```python
if _is_taxonomy_query(search_query):
    results = self._apply_metadata_boost(results)
```
- authoritative spec → ×1.3
- parameter_table / definition → ×1.2
- 两者可叠加，最大 boost ×1.56
- 优先展示物理层权威定义和参数表类型的 chunk

### Step 3.9: 相邻 chunk 上下文扩展 (Phase 5 新增)
```python
if _is_taxonomy_query(search_query):
    self._expand_adjacent_chunks(results, top_n=10, window=3)
else:
    self._expand_adjacent_chunks(results)  # top_n=5, window=2
```
- 为 Top-N 命中 chunk 拉取同文档相邻 chunk (±window)
- 分类列举问题扩大范围 (top_n=10, window=3)，解决表格/列表碎片化导致的召回不完整
- 相邻文本存入 `adjacent_chunks` 字段，拼入 LLM 上下文

### Step 3.2: 多跳检索 (条件触发)
```python
if needs_multi_hop(results):
    results = self._multi_hop.search(search_query, query_embedding)
```
- 触发条件：多样性 < 阈值 (chunks 集中在少数几个 spec)
- 流程：分析 chunk 摘要 → LLM 识别信息缺口 → 生成子查询 → 二次检索 → 合并去重
- 失败降级：继续使用单跳结果
- 参见 [[05-检索增强技术]]

### Step 3.4: 交叉引用解析
```python
results = self._resolve_cross_refs(results, max_refs=5)
```
- 扫描 Top-10 chunk 中的引用模式 (如 "see TS 38.413 §8.3.1")
- 对每个引用执行定向检索 → 补充到结果列表
- 限制 max_refs=5 控制延迟
- 参见 [[05-检索增强技术]]

### Step 3.5: 检索质量评估
```python
quality = evaluate_quality(results)
action = diagnose_quality(quality, len(results))
results = filter_noise(results)
```
- 三维评估：密度 (Density)、多样性 (Diversity)、覆盖度 (Coverage)
- 低质量诊断：建议重写查询 / 扩展检索 / 在线补充
- 噪声过滤：移除低分 + 低信息密度 chunks
- 参见 [[05-检索增强技术]]

### Step 3.6: Cross-Encoder 精排 + Spec-Aware
```python
results = self._post_process_results(query, expanded_query, query_embedding, results, top_k, reranker_enabled)
```
两步混合精排：
1. **Cross-Encoder Reranker** (`BAAI/bge-reranker-v2-m3`): 逐对计算 query-chunk 相关性
   - 融合公式: `final = α × norm_reranker + (1-α) × norm_original` (α=0.6)
   - 保护原始 RRF 排序信号，避免窄域区分度不足
2. **Spec-Aware 定向加权**: 对 LLM 识别的目标规范号 (如 38.413) 进行 ×2.0 boost + 定向检索补充

### Step 3.7: Release 感知
```python
release_intent = detect_release_intent(search_query)
if release_intent.type != "none":
    results, release_note = build_release_context(results, release_intent)
    release_note = build_release_note_for_prompt(release_intent, release_note)
```
- LLM 分析查询中的 Release 意图 (R18/R17/Comparative)
- 筛选/排序目标 Release 的 chunks
- 生成 Release 说明注入 Prompt (如 "以下内容基于 R18 规范")

### Step 3.8: 在线搜索补充
```python
if settings.enable_online_search and self._online.enabled:
    if self._online.should_supplement(best_score, len(results)):
        online_results = self._online.supplement_if_needed(search_query, best_score, len(results))
        online_context = self._online.format_as_context(online_results)
```
- 触发条件: `best_score < 0.6` 或 `len(results) < 5`
- 来源: TSpec-LLM (官方) → Google CSE `site:3gpp.org` (补充)
- 结果注入 RAG Prompt 的独立段 (标注 "外部来源")

### Step 4: 构建 RAG Prompt
```python
messages = build_rag_prompt(search_query, results, extra_system_note=release_note, online_context=online_context)
```
Prompt 结构:
```
[System] 你是 3GPP 协议专家...
[System] {release_note}
[System] {online_context}  ← 可选, 外部来源
[Context] Source 1: {spec_number} §{section_id} {title}\n{text}
[Context] Source 2: ...
[User] {query}
```

### Step 5: LLM 生成
```python
answer = self._llm.chat(messages)  # 英文答案
```
- 同步调用 OpenAI-compatible API
- 记录 Prompt/Completion token 数到监控

### Step 5.5: 多语言回译
```python
if query_lang != "en":
    answer = translate_from_english(answer, query_lang, self._llm)
```

### Step 6: 答案验证
```python
verification = self._verifier.verify(answer, results)
```
- 检查答案中的技术声明是否在检索结果中有支撑
- 输出: `{answer, verified: bool, warnings: list[str], coverage: float}`
- 覆盖度 < 阈值 → 标记 warning

## 4.3 性能数据

| 阶段 | 典型耗时 | 说明 |
|------|----------|------|
| 查询缓存命中 | < 1ms | 缓存命中直接返回 |
| i18n 翻译 | 200-500ms | LLM 翻译调用 |
| 查询扩展 | 200-500ms | LLM 扩展 |
| 查询嵌入 | 10-50ms | BGE-M3 单条 |
| 混合检索 | 50-100ms | Dense + BM25 + RRF |
| 多跳检索 | +200-500ms | 条件触发 |
| 交叉引用 | +50-200ms | 条件触发 |
| Cross-Encoder 精排 | 100-300ms | 从 100 条精选 20 条 |
| LLM 生成 | 1-3s | 取决于模型和输出长度 |
| i18n 回译 | 200-500ms | 非英文查询 |
| **总计 (无缓存)** | **2-5s** | - |
| **总计 (缓存命中)** | **< 10ms** | - |

## 4.4 容错设计

1. **查询扩展失败** → 使用原始查询
2. **嵌入生成失败** → 零向量 (1024-dim 全零)
3. **多跳检索失败** → 使用单跳结果
4. **Cross-Encoder 精排失败** → 使用原始 RRF 排序
5. **在线搜索失败** → 仅用离线结果
6. **LLM 生成失败** → 返回错误提示
7. **答案验证失败** → 返回未验证标记

## 4.5 扩展点

- **Pipeline 初始化时注入自定义 LLM**: `RAGPipeline(vector_store, llm_client=my_client)`
- **检索器可替换**: 任何实现 `VectorStore` 接口的对象
- **Prompt 模板可定制**: 修改 `src/generator/prompt.py` 中的模板
- **精排权重可调**: `ALPHA=0.6` 可改为 0-1 之间任意值

## 4.6 管线调试与诊断

### 4.6.1 逐阶段诊断方法

当 RAG 问答效果不理想时，按以下顺序逐阶段排查：

```
Step 0: 查询缓存 → 是否命中了过期/错误的缓存？
  诊断: 在日志中搜索 "cache hit" / "cache miss"
  
Step 1: i18n 翻译 → 非英文查询是否正确翻译？
  诊断: 日志搜索 "translate_to_english" 查看翻译前后文本
  
Step 2: 查询扩展 → 扩展后的查询是否增加了有效关键词？
  诊断: API 响应中的 expanded_query 字段
  
Step 3: 混合检索 → 返回的 chunks 是否相关？
  诊断: Admin 搜索测试页面单独测试 search（不开 Ask）
  
Step 3.2: 多跳 → 是否触发了多跳？补充了什么？
  诊断: 日志搜索 "Multi-hop triggered"
  
Step 3.4: 交叉引用 → 解析了哪些引用？补充了几条？
  诊断: 日志搜索 "cross_ref" / "Cross-reference resolved"
  
Step 3.5: 质量评估 → 检索质量评分如何？
  诊断: 日志搜索 "evaluate_quality" / "diagnose_quality"
  
Step 3.6: 精排 → reranker 是否生效？分数融合是否正常？
  诊断: 检查 chunks 返回顺序是否合理（Top-3 应高度相关）
  
Step 3.7: Release 感知 → 是否正确识别了 Release 意图？
  诊断: 日志搜索 "detect_release_intent" / "release_intent"
  
Step 3.8: 在线补充 → 是否触发了在线搜索？补充了哪些？
  诊断: 日志搜索 "online_supplement" / "should_supplement"
  
Step 4-5: Prompt + LLM → Prompt 内容是否完整？LLM 响应是否合理？
  诊断: 日志搜索 "build_rag_prompt" / "LLM response"
  
Step 6: 验证 → 答案是否通过验证？覆盖度多少？
  诊断: API 响应中的 verified / coverage / warnings 字段
```

### 4.6.2 常见管线问题速查

| 症状 | 最可能出问题的阶段 | 排查优先级 |
|------|-------------------|-----------|
| 问答完全无关 | Step 3 检索 | ①先看 search 结果是否相关 |
| 答案不完整 | Step 3.2 多跳未触发 | ②检查多样性是否低于阈值 |
| 缺少关键规范引用 | Step 3.4 交叉引用 | ③检查引用是否被正确提取 |
| 答案语言错误 | Step 1/5.5 i18n | ④检查翻译链 |
| 答案包含幻觉 | Step 4 Prompt / Step 6 验证 | ⑤检查 prompt 模板 + 验证结果 |
| 非 R18 内容混入 | Step 3.7 Release 感知 | ⑥检查 release_intent 检测 |
| 离线环境结果少 | Step 3.8 在线补充 | ⑦检查 online_search 配置 |
| 非英文查询效果差 | Step 1 + Step 5.5 | ⑧检查翻译质量和回译准确性 |

### 4.6.3 LLM 调用追踪

```bash
# 查看最近的 LLM 调用
grep "LLM" logs/app.log | tail -20

# 典型日志格式：
# [INFO] LLM call: model=gpt-4o-mini, prompt_tokens=1234, completion_tokens=256, duration=1.5s
# [ERROR] LLM API call failed: ConnectionError (retry 1/3)
# [WARNING] LLM response truncated: max_tokens=2048 exceeded
```

### 4.6.4 Prompt 调试

当怀疑 Prompt 模板有问题时：

```bash
# 方法 1: 查看 API 响应中的 expanded_query（确认查询扩展效果）
curl -s -X POST http://localhost:8000/api/v1/ask \
  -H 'Content-Type: application/json' \
  -d '{"query":"PDU Session","top_k":10}' | \
  python -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',{}).get('expanded_query','N/A'))"

# 方法 2: 开启 DEBUG 日志级别查看完整 prompt 构建过程
LOG_LEVEL=DEBUG python -m src.main
# 日志中会输出: build_rag_prompt() → 系统提示词 + 上下文 chunks + 用户查询
```
