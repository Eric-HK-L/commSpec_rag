# 检索质量根因分析与优化路线图

> **以代码为准**：本仓库 `docs/` 下多份文档已过期（如
> `docs/open_source/3gpp-rag-rel18-vs-3gpp-rag-project.md` 声称"chunk 未保留 section_number/
> section_path / 无图增强检索"）。这些在当前代码里**均已实现**（见下"代码核对结论"）。
> 本文所有判断均以 `src/` 当前实现为准，遇到与文档冲突处以代码为准。

> 结论先行：当前链路**第一阶段的混合检索（Dense+BM25+RRF）召回不足是主要瓶颈**，
> Cross-Encoder 精排已经把初检 Recall@5 从 0.457 救到 0.729（+0.27），说明精排器工作正常、
> 问题出在"送入精排器的候选池"和"候选池内部的排序"。优化重心应放在 **嵌入/分块/ANN 索引/BM25
> 词法检索** 这些上游环节，而不是继续堆下游精排。

## 0.0 代码核对结论（本次分析的事实基线）

逐文件核对 `src/` 当前实现，以下为**已确认的代码事实**（与部分过期文档相反）：

| 事实                                                | 代码位置                                                                                                                                     | 说明                                                                                     |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| ✅ section_number/section_title/section_path 已保留 | `vector_store.py` Chunk/SearchResult、`splitter.py` `_build_section_path`                                                                    | 过期文档说"缺失"，实际已实现并入库                                                       |
| ✅ 离线 xref 图已实现并接入                         | `ingestion/xref_graph_builder.py`、`retriever/graph_expander.py`、`planner.py:221`                                                           | 已生成 162MB 图（47,806 节点/224,564 边），planner 已加载                                |
| ✅ small-to-big 父上下文已实现                      | `milvus_store.py` parent_text/parent_chunk_id、`planner.expand_parent_context`                                                               | —                                                                                        |
| ⚠️ 嵌入文本 = 纯正文                                | `embedder.py:embedding_text()` 返回 `chunk.text`                                                                                             | 标题/路径**未**编入向量（见 P0-1，仍待 A/B）                                             |
| ⚠️ chunk 级 `summary` 字段不存在                    | `vector_store.py` Chunk、`milvus_store.py` schema 均无 `summary`                                                                             | `docs/optimization/chunk-llm-summary.md` 方案**未落地**                                  |
| ⚠️ ANN 索引 IVF_FLAT nprobe=32                      | `milvus_store.py:221` nlist=1024、`:448` nprobe=32                                                                                           | 只探测 ~3% 的簇                                                                          |
| ⚠️ BM25 = Python rank-bm25，`split()` 分词          | `bm25_index.py`                                                                                                                              | 未用 Milvus 原生 BM25                                                                    |
| ❌ **增量/全量摄入分块参数不一致**                  | `orchestrator.py:64` 用 `IngestionConfig`(chunk_size=1024/overlap=200)；`incremental.py:120` 用 `HeaderAwareSplitter()` 硬编码默认(2500/100) | 见 P0-6                                                                                  |
| ❌ **增量摄入未调用 classify_chunk**                | `incremental.py:131-144` 直接 split→embed→insert                                                                                             | 增量文档 content_type/spec_role/topic_domain 全空，元数据 boost 与低质过滤对增量文档失效 |

## 0. 评测数据定位瓶颈

`tests/eval/eval_report.md`（70 题，plan() 全链路）：

| 指标            | 值         | 目标  | 结论                 |
| --------------- | ---------- | ----- | -------------------- |
| 初检 Recall@5   | **0.4571** | 诊断  | 第一段检索漏召回严重 |
| 重排后 Recall@5 | 0.7286     | ≥0.80 | 未达标               |
| Recall@10       | 0.7976     | ≥0.85 | 未达标               |
| Recall@20       | 0.8881     | ≥0.90 | 接近                 |
| MRR             | 0.8724     | ≥0.70 | ✅                   |
| NDCG@10         | 0.8953     | ≥0.75 | ✅                   |
| 多跳 Recall@5   | 0.5903     | —     | 最弱环节             |

关键解读：

1. **初检 Recall@5 = 0.457**：混合检索（Dense + BM25 + RRF）返回的 top-5 里，只有 45.7% 的题
   把正确规范排进前 5。候选池是 100（`reranker_top_k=100`），Recall@20 重排后 0.888 说明
   **约 11% 的题正确规范根本没进 100 候选池**（ANN/嵌入/分块层面的彻底漏召），其余 ~43% 是
   "进了池子但排得靠后"（排序问题）。
2. **精排器是当前最强的一环**（0.457→0.729），继续在 reranker 上投入边际收益已递减。
3. **多跳 Recall@5=0.59**：触发启发式过窄 + 缺口分析输入太薄（见 P1-5）。

> 另需注意评测指标的粒度：Recall 目前按 `spec_number`（整篇文档）计算，不校验**章节**是否命中。
> "正确规范进了 top-5" 不等于"正确章节进了 top-5"。真实答案质量可能比 0.73 更差。建议升级评测
> 到章节级 Recall + 答案级 groundedness（见 P2-3）。

---

## P0 — 第一阶段召回的直接杠杆（低成本/纯配置/可 A/B）

### P0-1 嵌入文本构成：标题 + 层级路径是否编入向量（最该先做 A/B）

**现状**：`src/ingestion/embedder.py:embedding_text()` 只返回 `chunk.text`（纯正文），
并用注释明确拒绝把 `section_title / section_path` 拼进嵌入输入，理由是"domain shift"。

**现状确认（以代码为准）**：`section_number/section_title/section_path` 元数据**已经在
Chunk 里**（`vector_store.py`），只是 `embedding_text()` 只取 `chunk.text`、**没有把它们拼进
向量**。过期的对比文档把"元数据缺失"列为根因是错的——元数据不缺，缺的是"是否编入向量"这个
**尚未被 A/B 验证**的决策。两个方向（纯正文 vs 标题+路径+正文）目前都没有 70 题评测集支撑。

**问题**：3GPP 规范高度同质——"PUSCH power control" 和 "PUCCH power control" 的正文向量
天然接近；而真正能区分的上下文（`§7.1.1` / `§7.2.1`、`Uplink power control` vs
`Downlink power control`）被丢弃。对"纯数字参数表"类 chunk（PRACH preamble 表）更是致命：
正文几乎全是 `1151 15 30 48...`，没有任何文字能对齐查询词。

**业界佐证**：Anthropic 的 [Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
（把 chunk 的上下文摘要拼到 chunk 前再嵌入）实测减少 **49%** 检索失败；对协议这种层级文档，
"上下文" 就是 `section_number + section_title + section_path`。

**A/B 实测结论（2026-08，61k chunks 重嵌入）**：`section_path + text` 模式**显著变差**——
RAN 子集重排 Recall@5 从 0.8465 跌到 0.7544（**-0.092**，8 题下降/2 题提升）。层级路径是
泛化文本，拼在前会稀释正文中"PRACH/DMRS/CSI-RS"等具体术语的 dense 信号。**原 `embedding_text`
注释的"domain shift"判断是正确的，纯正文是更优选择**。故不再推荐把 section_path 编入向量；
表格/公式的上下文注入应走 chunk 设计层（见 `chunk-design-analysis.md`），而非简单拼接路径。

### P0-2 ANN 索引 recall：IVF_FLAT nprobe 只搜了 3% 的簇

**现状**：`src/retriever/milvus_store.py:create_collection()` 建 `IVF_FLAT, nlist=1024`；
`search_dense()` 用 `nprobe=32`。即每次查询只探测 `32/1024 ≈ 3.1%` 的聚类。

**问题**：1024 维向量 + nlist=1024 下，nprobe=32 是**明显的近似召回折损点**。对于 3GPP 这种
"同一个词在不同 spec 反复出现、语义近邻跨簇分布"的语料，正确答案很可能落在未探测的簇里，
这正是"初检 Recall@5=0.457"的隐形贡献者之一（ANN 直接漏掉，连精排器都救不回）。

**建议**（按成本从低到高）：

1. 先只调 `nprobe` 到 128~256（重建索引**不必要**，改 search 参数即可，秒级验证），看初检 Recall 变化。
2. 若仍不足，改 `HNSW` 索引（`M=16~32, efConstruction=200`，`ef=128`），对 1024 维 + 中等规模
   语料 recall 明显优于 IVF_FLAT，代价是内存略增 + 需重建索引。

### P0-3 chunk 偏大，稀释 Dense 向量

**现状**：`src/config/settings.py`（`IngestionConfig`，经 `orchestrator.py:64` 注入 splitter）：
`chunk_size=1024`（触发长章节二次切分的阈值）、`prose_max_chars=1500`、`table_max_chars=5000`、
`max_chunk_chars=8000`（BGE-M3 token 安全上限）。即正文 chunk 可达 ~~1500 字符，表格/公式作为
"原子块"常整块成 chunk（最多 5000~~8000 字符）。

**问题**：Dense bi-encoder（BGE-M3）对长 chunk 会把几百行表格/整段章节的信息"平均"进一个
向量，导致细粒度查询（"preamble format 4 的 CP 长度"）难以命中。这违背了 small-to-big 的
原则——**"小" chunk 用于精确召回，"大" 父上下文用于喂给 LLM**。当前项目 small-to-big 的
"父上下文"机制（`parent_text`/`parent_context`）已经建好，但"小 chunk"本身并不小（1500~5000 字符）。

**建议**：把用于**嵌入+召回**的小 chunk 收缩到 ~~256~~512 token（正文 ~~400~~600 字符，表格
按行组切到 ~1000 字符且每片保留表头），父上下文继续用现有机制拼接。这与
[Jina 的 long-context 分块讨论](https://jina.ai/news/still-need-chunking-when-long-context-models-can-do-it-all/)
一致：长上下文模型仍需要分块，但关键是"召回用小块、生成用大块"。

### P0-4 BM25 分词太原始，且未用 Milvus 原生 BM25

**现状**：`src/retriever/bm25_index.py` 用 Python `rank-bm25`，分词是 `text.lower().split()`；
Milvus 侧只存 Dense（`milvus_store.py` 注释"BM25 待后续启用"）。

**问题**：

1. `split()` 把 `PRACH-preamble`、`N2-interface`、`38.211`、`TS 38.413` 这类**技术复合词/
   规范号**拆成无意义的碎片或连在一起，词法召回对协议术语极不友好（无词干化、无连字符处理）。
2. Python BM25 与 Milvus 双库，RRF 在 Python 侧拼装，过滤/一致性维护成本高。

**建议**：

1. 短期：给 `_tokenize` 换成 `wordpunct` 分词 + 保留连字符两侧独立 token + 小写 + 可选
   kstem（`rank_bm25` 只吃 token 列表，纯函数替换，无需改 Milvus schema）。
2. 中期：迁移到 [Milvus 原生 BM25](https://milvus.io/docs/full-text-search.md)
   （`SPARSE_FLOAT_VECTOR` + analyzer），或直接启用 **BGE-M3 自带的稀疏（lexical）权重**
   替代 rank-bm25——BGE-M3 本身就是 dense+sparse 双输出，用它的 sparse 与 dense 同源对齐，
   比外挂 BM25 更协调。

### P0-6 增量/全量摄入分块参数不一致（代码 bug，需先修）

**现状**：`orchestrator.py:64` 用 `IngestionConfig` 实例化 splitter（`chunk_size=1024`、
`chunk_overlap=200`）；但 `incremental.py:120` 用 `HeaderAwareSplitter()` **硬编码默认值**
（`max_chunk_chars=2500`、`chunk_overlap_chars=100`）。此外 `incremental.py` 摄入后**不调用
`classify_chunk`**，导致增量文档的 `content_type/spec_role/topic_domain` 全为空。

**影响**：

1. 同一篇文档，全量摄入与增量摄入切出的 chunk 边界、overlap、大小都不同 → 向量空间/嵌入不一致，
   正是 `embedding_text()` 注释里极力避免的"多路径不一致"问题。
2. 增量文档无元数据 → `_apply_metadata_boost`（×1.3/×1.2）和 `filter_low_quality` 对它们失效，
   增量文档的权威参数表无法获得加权，低质章节也过滤不掉。

**建议**：把 splitter 构造逻辑抽成一个共享工厂（按 `IngestionConfig` 统一构造），两处调用；
增量路径补齐 `classify_chunk` 标注。这是纯正确性修复，不影响算法选型。

### P0-5 RRF 融合权重未调

**现状**：`settings.rrf_k_dense=60, rrf_k_sparse=60`（等权）；代码注释自己也写了
"3GPP 领域 Dense 通常优于 BM25，可设 k_dense=40 / k_sparse=120"。

**建议**：用 70 题评测集扫 2~3 组（如 40/120、30/150、50/80），取初检 Recall@5 最优。纯配置，
可和 P0-1 的 A/B 一起跑。

---

## P1 — 定向增强（中等工作量，ROI 高）

### P1-1 低语义密度 chunk 摘要注入（表格"上下文前置"）

已有一份完整设计 `docs/optimization/chunk-llm-summary.md` 但**未落地**（Milvus schema 无
`summary` 字段）。核心问题与 P0-1 同源：纯数字表/公式块嵌入质量极低。

建议先做**规则版**（零 LLM 成本）：对 `content_type == "parameter_table"` 的 chunk，在嵌入文本
前注入 `表格标题/编号 + 列头 + 所在 section 说明`（splitter 已能拿到这些元数据）。规则版验证
有效后，再对规则覆盖不了的 ~5-10% chunk 用本地小 LLM（Qwen2.5-7B）补摘要。

### P1-2 缩写表太小（23 条 → 数百条）

`src/retriever/glossary.py` 只有 23 个缩写；3GPP 每篇规范 §3.1 都有上百条。Telco-RAG 论文
[arXiv:2404.15939](https://arxiv.org/abs/2404.15939) 明确把缩写展开列为高性价比召回手段。

建议：离线从每篇规范 `§3 Definitions and abbreviations` 章节抽取 `缩写 → 全称` 对，
生成 `data/processed/abbreviations.json`，查询侧展开复用现有 `expand_abbreviations()`。

### P1-3 查询侧多视角 RRF（single-query → multi-query）

当前只有 `_taxonomy_decompose_search()` 在"分类列举"题上做多子查询；普通题只有单条
`expanded_query`。参考实现 `3gpp-rag-rel18` 的 multi-query RRF（3-5 个子查询并行检索后 RRF 合并）
在 rel18 上显著提升 recall。建议把多子查询从"taxonomy 专用"泛化为"所有查询可选"：
缩写/全称、近义术语、相关协议层各一个视角。

### P1-4 Reranker 截断与 chunk 大小错配

`src/retriever/reranker.py:max_length=1024`，而 chunk 可达 8000 字符。表格答案若在表中段，
reranker 根本看不到。两条路（都与 P0-3 联动）：

- 小 chunk 化后自然缓解；
- 或 reranker 改打 `(query, summary/section_title + 正文头部)` 而非全文。

### P1-5 多跳触发与缺口分析太薄

`src/retriever/multi_hop.py`：

- 触发条件 `needs_multi_hop` 仅在 spec 多样性 ≤0.25 时开启，跨规范但"多样性刚好达标"的题不触发；
- 缺口分析只喂给 LLM **每条 chunk 前 120 字符 × 8 条**的摘要，信息量不足以判断缺口。
  建议：缺口分析输入改为 `section_number + section_title + 首 300 字符`，并把触发从
  "多样性启发式"改为"LLM 判定 + 关键词（对比/关联/影响）"。

---

## P2 — 结构性升级（大工作量，长期方向）

### P2-1 Late Chunking / ColBERT 晚交互

对长章节，[Jina Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models/)
先把整段（含跨句上下文）过一遍长上下文编码器，再按 chunk 边界做 mean-pooling，保留跨 chunk
上下文——对"定义在 §7.4.1、参数表在 §7.4.2"这类拆分很友好。BGE-M3 本身支持 8192 token，
可评估 late chunking；更高上限是 [ColBERT 晚交互](https://blog.milvus.io/ai-quick-reference/what-is-colbert-and-how-does-it-differ-from-standard-biencoder-approaches)
（[jina-colbert-v2](https://arxiv.org/abs/2408.16672) 多语言），词级交互对技术术语匹配更精准，
但需要 Milvus 多向量支持 + 全量重建。

### P2-2 层级索引 / GraphRAG 深化

已建成 162MB 的 `xref_graph.json`（47,806 节点 / 224,564 边），但它是**单层 JSON 全量加载**
（`graph_expander.py`），且只做"命中后沿 REFERENCES 边扩展"。可再进一步：

- **文档级摘要索引**：每篇 spec 一个摘要向量，先路由到正确 spec，再在该 spec 内做 chunk 检索
  （两级检索，显著降低跨 spec 混淆，参考 [LightRAG 层级节点](https://github.com/HKUDS/LightRAG)）。
- 顺带优化 162MB JSON 的加载（惰性加载 / 换 SQLite / 只加载 38.xxx 子图），降低启动内存与延迟。
- 对齐 [LARAG（Link-Aware Retrieval，面向带超链接技术文档）](https://arxiv.org/html/2605.07517v1)
  的思路：3GPP 的跨规范引用正是最强的结构信号，离线图要作为**主动召回通道**而不只是被动扩展。

### P2-3 评测升级（章节级 + 答案级）

当前 `metrics.py` 只算 `spec_number` 级 Recall。建议：

1. 测试集每条补 `expected_sections` 的**严格命中判定**（`section_number` 前缀匹配）；
2. 引入答案级 groundedness/faithfulness（如 RAGAS 的 faithfulness + context precision），
   直接对齐"用户觉得效果不好"的体感。

---

## 建议执行顺序

1. **第 0 轮（半天，纯 bug 修复）**：统一增量/全量 splitter 构造 + 增量补 `classify_chunk`
   （P0-6）——正确性优先，且让后续所有评测/调优跑在一致的索引上。
2. **第 1 轮（1-2 天，纯配置/脚本，立刻做）**：
   `nprobe 32→128/256`（P0-2）、RRF k 扫描（P0-5）、`embedding_text` A/B（P0-1）——三者都走
   现有 `tests/eval/run_eval.py` 出指标，不写新代码或少写。
3. **第 2 轮（3-5 天，小改）**：BM25 分词升级（P0-4-1）、表格 chunk 上下文前置规则版（P1-1）、
   缩写表扩容（P1-2）。
4. **第 3 轮（1-2 周，重建索引）**：小 chunk 化 + HNSW + BGE-M3 原生 sparse（P0-3/2/4-2）、
   多视角 RRF（P1-3）、多跳触发/缺口分析（P1-5）。
5. **第 4 轮（长期）**：Late chunking/ColBERT、层级索引、评测升级（P2）。

每轮都用 `tests/eval/run_eval.py --fresh` 回归，并保留"初检 Recall@5"作为第一段检索的
独立诊断指标（当前 0.457，是全部工作的北极星）。
