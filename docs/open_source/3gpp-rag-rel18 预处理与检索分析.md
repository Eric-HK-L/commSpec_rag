分析 `open_sources/3gpp-rag-rel18-main` 项目在3GPP规范文档预处理（Section Chunking）与检索调用（Hybrid + Graph Enhanced）方面的独到设计，覆盖切片规则、FAISS索引构建、交叉引用图构建、多查询融合与图增强检索全流程。

# 1 项目概览
`3gpp-rag-rel18-main` 是一个面向3GPP NR Release-18技术规范（38.xxx系列）的RAG检索技能（Cline Skill形态），目标是把规范文档转化为可精确引用、可跨规范追溯的问答能力。其核心定位是解决传统Vector RAG在3GPP规范场景下“~50%准确率、chunk切断语义、无知识积累”的痛点，目标准确率>90%。

项目代码集中在 `agents/skills/3gpp-rag-search/` 下，关键模块：

|模块|职责|
| ---- | ---- |
|`src/section_chunker.py`|Markdown → Section Chunk JSON|
|`src/batch_chunker.py`|批量切片入口|
|`src/build_faiss_index.py`|Chunk → Embedding → FAISS索引|
|`src/build_xref_graph.py`|Chunk → Cross-Reference Graph（离线）|
|`src/rag_search_engine.py`|混合检索引擎（FAISS + 关键词 + RRF）|
|`src/graph_enhanced_search.py`|检索结果 + 图扩展|
|`src/abbreviation_resolver.py`|缩写字典 + 多查询生成|

##2 文档预处理流程
## 2.1 整体管线
```
Markdown 文档（含 YAML front matter）
↓
SectionChunker.chunk_markdown()
｜解析 #~###### 标题 → 章节树 → 叶子节点 chunk
↓
data/specs/chunks/{doc_id}.json （1 文档 = 1 chunk 文件，content 直接内嵌）
├─→ build_faiss_index.py → data/embeddings/chunks.index + metadata + content_map
└─→ build_xref_graph.py → resources/3GPP_Wiki/graph/xref_graph.json + .md
```

## 2.2 Section-Based Chunking（章节级切片）
切片规格定义在 `docs/CHUNKING_SPEC.md`，实现于 `section_chunker.py`。这是该项目最独到的设计之一。

### 2.2.1 设计动机
|旧方式问题|新方式解决|
| ---- | ---- |
|`end_index` 用 +100 魔法数字|`end` = 下一标题行号，精确定位|
|无 section 大小限制，613 行 section 整块返回|`max_lines` 超限则按子章节/强制拆分|
|Tree 只存索引，需回查原文|chunk 内直接保存 content，无需回查|
|索引与内容不一致|content 内嵌，索引即内容|

### 2.2.2 切片规则
1. 标题驱动：按 `# ~ ######` 标题层级构建章节树
2. 只生成叶子节点 chunk：非叶子（有子章节的）不单独成 chunk，避免父子内容重叠
3. `section_number` 严格保留：从标题正则提取（如 `4.1`、`7.1.1`、`A.1`），`chunk_id = {doc_id}{section_number.replace('.','_')}`
4. `section_path` 层级路径：`"7 Uplink Power control > 7.1 PUSCH > 7.1.1 UE behaviour"`，为检索提供上下文
5. 超长拆分：`max_lines`（默认200）超限时，先按子章节拆；无章节则强制按 `max_lines` 拆，`chunk_id` 加 `_p1/_p2` 后缀，但 `section_number` 保持原值
6. YAML front matter：提取为文档级元数据（`doc_number/version/release/title`），不生成chunk

### 2.2.3 chunk JSON 结构
```json
{
  "doc_id": "38213-i70",
  "doc_number": "38.213",
  "version": "V18.7.0",
  "release": "Rel-18",
  "chunks": [
    {
      "chunk_id": "38213-i70_4_1",
      "section_number": "4.1",
      "section_title": "Cell search",
      "section_path": "4 Synchronization procedures > 4.1 Cell search",
      "content": "## 4.1 Cell search\n...(完整原文)...",
      "line_start": 184,
      "line_end": 243,
      "line_count": 59,
      "token_estimate": 236,
      "parent_section_number": "4"
    }
  ]
}
```
关键点：`content` 字段直接保存章节原文，检索后无需二次回查原文文件，彻底消除“索引-内容不一致”风险。

### 2.2.4 质量校验
切片后自动校验：全行覆盖、chunk 间无间隙、父子关系一致、`max_lines` 合规、`content` 非空。

## 2.3 FAISS 索引构建
`build_faiss_index.py` 把 chunk 转为向量索引：
1. **嵌入文本构造（关键）**
```python
embed_text = f"{section_title} {section_path} {content_preview}"
# content_preview = content[:500]
```
把 `section_title + section_path + content` 前500字拼接为嵌入文本。这意味着层级路径（如 `"Uplink Power control > PUSCH > UE behaviour"`）被编入向量，提升同主题不同章节的区分度。
2. 模型：`paraphrase-multilingual-MiniLM-L12-v2`（384-dim，多语言，轻量）
3. 索引类型：`IndexFlatIP`（内积 = 归一化后的余弦相似度），`normalize_embeddings=True`
4. 输出三件套：
    - `chunks.index`：FAISS 索引
    - `chunks_metadata.json`：元数据（`chunk_id/doc_id/section_number/section_path/content_preview`）
    - `chunks_content.json`：`{id: full_content}` 映射，检索时直接取完整内容

## 2.4 Cross-Reference Graph 构建（最核心创新）
`build_xref_graph.py` 离线构建规范间交叉引用图，这是该项目区别于普通 RAG 的最大亮点。整个构建分5个阶段，最终输出一个含邻接索引的JSON图，检索时只需O(1)邻接查询即可发现跨规范关联。

### 2.4.1 图的数据模型
图由两类节点和五类边组成：
**节点类型**
|节点类型|id 格式|含义|关键属性|
| ---- | ---- | ---- | ---- |
|SPEC_SECTION|`{doc_id}_{section_number}`|一个chunk = 一个节点|`spec, section, title, path, doc_id, parent_section`|
|IE|`IE_{ie_name}`|ASN.1 信息元素定义|`name, asn1_type, asn1_def, spec, section, doc_id`|

**边类型**
|边类型|方向|含义|权重|来源阶段|
| ---- | ---- | ---- | ---- | ---- |
|REFERENCES|chunk → chunk|跨spec / 同spec规范引用（最强连接）|cross-spec 1.0 / 同spec 0.6|阶段3|
|PARENT_CHILD|父章节 → 子章节|章节树结构|0.5|阶段2|
|SIBLING|兄弟章节 → 下一兄弟|同父连续章节|0.3|阶段2|
|NEXT_SECTION|顶层章节 → 下一顶层|顶层连续编号章节（如4→5）|0.4|阶段2|
|DEFINES|section → IE|章节定义了某信息元素|0.8|阶段4|

### 2.4.2 阶段1：节点生成与索引构建
扫描 `data/specs/chunks/*.json` 的所有chunk文件，为每个chunk创建 `SPEC_SECTION` 节点。同时构建4个索引：
1. `section_index`: `(spec, section_number) → [chunk_id, ...]`
    - 同时把 `parent_section_number` 也索引进来（用于 `PARENT_CHILD` 边查找）
    - 只索引有 `section_number` 的chunk（IE chunk 的 `section_number=""` 被排除，防止爆炸）
2. `doc_to_spec`: `doc_id → spec`（如 `38331-i60 → 38.331`）
3. `doc_chunks`: `doc_id → [chunk, ...]`
4. `references_sections`: `doc_id → §2 References 章节内容`（ground truth，不参与边生成）

`doc_id` 到 spec 号的转换函数 `doc_id_to_spec_with_part()` 处理多part规范（如 `38101-1-ia0 → 38.101-1`）。

### 2.4.3 阶段2：结构边生成（PARENT_CHILD / SIBLING / NEXT_SECTION）
对每个文档的chunk列表：
**PARENT_CHILD 边**
- 遍历每个chunk的 `parent_section_number`
- 通过 `section_index[(spec, parent_section)]` 找到父chunk
- 生成父→子边，权重0.5

**SIBLING 边**
- 按 `parent_section_number` 分组同父兄弟
- 按 `section_number` 数字排序（`[int(x) for x in s.split('.')]`）
- 相邻兄弟生成兄→弟边，权重0.3

**NEXT_SECTION 边**
- 只对顶层章节（无 `parent_section_number`）
- 按 `section_number` 排序
- 若 `n2 == n1 + 1`（连续编号），生成 `n1 → n2` 边，权重0.4
- 非连续编号（如4→6）不生成

### 2.4.4 阶段3：引用边生成（REFERENCES）— 最核心
这是图的价值所在。对每个chunk content（跳过§2 References章节），用多模式正则提取规范引用。

**引用提取正则（6个模式，按精度从高到低）**
```python
# 模式 0（最高精度）："Clause X.Y.Z of TS 38.AAA" – spec + clause 联合提取
CLAUSE_OF_TS_PATTERN = re.compile(
    r'(?:Clause|clause|section|Section)\s+([\d\.A-Za-z]*)\s+of\s+(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})(?:-\d)?',
    re.IGNORECASE
)
# 匹配 "5.3A.1", "8.1A", "6a" 等 letter-suffixed section

# 模式 0b："Clause X.Y.Z of [N, TS 38.AAA]" – bracket 形态
CLAUSE_OF_BRACKET_PATTERN = re.compile(
    r'(?:Clause|clause|section|Section)\s+([\d\.A-Za-z]*)\s+of\s+\[(\d+),\s+(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})\]',
    re.IGNORECASE
)

# 模式 1："TS 38.214" + 邻近 clause – spec + 邻近上下文 clause
SPEC_REF_PATTERN = re.compile(
    r'(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})(?:-\d)?',
    re.IGNORECASE
)
# 匹配后在 ±50~100 字上下文里用 CLAUSE_REF_PATTERN 找 clause

# 模式 2："[6, TS 38.214]" – bracket 引用
BRACKET_REF_PATTERN = re.compile(
    r'\[(\d+),\s+(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})\]',
    re.IGNORECASE
)

# 模式 3："clause 9.2.5.4" – 同 spec 内引用（无 TS 前缀）
SAME_SPEC_CLAUSE_PATTERN = re.compile(
    r'(?:clause|section|subsection|sub-clause)\s+([\d\.A-Za-z]*)',
    re.IGNORECASE
)
```

**去重策略（关键）**：高精度模式匹配后，把匹配区间 `(start, end)` 记入 `matched_spans` 集合，后续低精度模式跳过已匹配区间，避免同一引用被多次提取。

**同 spec clause 排除规则**：模式3匹配 `clause X.Y.Z` 时，检查该匹配所在行是否含 TS `\d{2}\.\d{3}`（`TS_REF_INLINE_PATTERN`）。若同行有TS引用，说明是cross-spec，跳过（留给模式0/1处理）；否则才算同spec引用，用 `source_spec` 作为目标spec。

**NR spec过滤**：所有模式只处理 `38.` 开头的spec（排除36.xxx LTE）。

**clause尾部清理**：`clause.rstrip('.')` 去除尾部点（如 `6.3.2. → 6.3.2`）。

**section → chunk 匹配（`find_chunks_by_section()`）**
把提取到的 `(target_spec, clause)` 解析到具体 `chunk_id`，匹配优先级：
1. 精确匹配：`("38.213", "9.2.3")` 直接查 `section_index`
2. 前缀匹配：`"9.2.3"` 开头的子章节（`9.2.3.1`, `9.2.3.2`）
3. 上级回退：`9.2.3 → 9.2 → 9`，逐级向上找
4. 上级前缀匹配：上级章节前缀开头的所有章节

**IE fallback**：若clause匹配不到，查 `parent_section_ie_index[(spec, clause)]`（IE chunk 的 `parent_section` 精确匹配），取第一个，防止爆炸。

**边生成**：对每个目标 `chunk_id` 生成 `REFERENCES` 边，含：
- `weight`：cross-spec 1.0 / 同spec 0.6
- `evidence`：引用原文片段（±20~60字）
- `target_spec / target_clause / is_cross_spec`

**去重**：`(from, to, target_spec, clause)` 四元组去重。

### 2.4.5 阶段3.5：IE名字匹配（隐式跨 spec 关联）
除了显式的clause引用，还通过IE名字共现建立隐式跨spec关联：
1. 离线提取：对每个spec，用ASN.1正则提取所有IE名字集合 `ie_names_by_spec[spec]`
2. IE chunk索引：`(spec, ie_name) → chunk_id`
3. 宽上下文收集：对chunk中每个TS引用，收集±300字上下文 `wide_context_by_spec[spec]`
4. 名字匹配：
    - 对每个目标spec的IE名字，在±300字上下文里搜索
    - 命中（±300字内）→ 权重0.8
    - 未命中但在chunk全文命中 → 权重0.5（fallback，有FP风险但保recall）
    - 都不命中 → 跳过
5. 生成 `chunk → IE_chunk` 的 `REFERENCES` 边，`target_clause = "IE:{ie_name}"`

**价值**：即使chunk没有显式写 `clause X.Y.Z of TS 38.AAA`，只要提到了某IE名字（如 `CSI-RS-ResourceMapping`），就能关联到定义该IE的chunk。

### 2.4.6 阶段4：IE定义提取（DEFINES 边）
用ASN.1正则提取信息元素定义：
```python
ASN1_PATTERN = re.compile(
    r'(\w[\w-]*)\s*::=\s*'
    r'(INTEGER|SEQUENCE|ENUMERATED|BIT\s+STRING|OCTET\s+STRING|CHOICE|BOOLEAN)',
    re.IGNORECASE
)
```
- 过滤：IE名<3字符或为通用词（`true/false/null/integer/...`）跳过
- 对每个IE定义生成 `IE_{name}` 节点（首次定义才建节点，去重）
- 生成 `section_chunk → IE_node` 的 `DEFINES` 边，权重0.8

### 2.4.7 阶段5：adjacency 邻接索引构建
把所有边转为邻接索引，供检索时O(1)查询：
```python
adjacency = defaultdict(lambda: defaultdict(list))
for edge in all_edges:
    adjacency[edge["from"]][edge["type"]].append(edge["to"])
# 转为普通 dict 以便 JSON 序列化
adjacency = {k: dict(v) for k, v in adjacency.items()}
```
最终 `adjacency[chunk_id]["REFERENCES"]` 返回该chunk引用的所有目标 `chunk_id` 列表。

### 2.4.8 输出文件
**`xref_graph.json`（代码用）**
```json
{
  "metadata": {
    "total_nodes": 5000,
    "total_edges": 12000,
    "edges_by_type": {
      "REFERENCES": 3000,
      "REFERENCES_CROSS_SPEC": 1500,
      "PARENT_CHILD": 4000,
      "SIBLING": 2000,
      "NEXT_SECTION": 500,
      "DEFINES": 2500
    }
  },
  "nodes": [...],
  "edges": [...],
  "adjacency": {
    "38331-i60_6_3_2": {
      "REFERENCES": ["38214-i70_5_2_2", "38212-i70_6_1"],
      "PARENT_CHILD": ["38331-i60_6_3_2_1"],
      "DEFINES": ["IE_CSI-RS-ResourceMapping"]
    }
  },
  "references_sections": {"38331-i60": "§2 References 原文..."}
}
```
**`xref_graph.md`（LLM 用摘要）**
- 统计表（节点/边数）
- Spec 别节点数
- Cross-Spec 参照矩阵：`source_spec → {target_spec: count}`，揭示规范间引用强度
- IE 定义表（前100个）

### 2.4.9 图构建的关键设计决策
1. 只处理38.xxx（NR）：排除36.xxx LTE，聚焦5G NR
2. §2 References章节跳过：不参与 `REFERENCES` 边生成（避免噪声），单独存为ground truth
3. IE chunk索引隔离：`section_number=""` 的IE chunk不进 `section_index`，防止前缀匹配爆炸
4. `parent_section` IE fallback：clause匹配失败时，用parent_section精确匹配IE chunk（取第一个，防爆）
5. ±300宽上下文：IE名字匹配用宽上下文（±300字），而非evidence的窄上下文（±80字），避免漏匹配
6. 权重分级：cross-spec REFERENCES 1.0 > 同spec 0.6 > DEFINES 0.8 > PARENT_CHILD 0.5 > NEXT_SECTION 0.4 > SIBLING 0.3

# 3 检索与调用流程
## 3.1 四步决策流
```
用户问题
↓
Step 1：问题分析 + 多查询构造（缩写扩展 + 3-5 视角子查询）
↓
Step 2：混合检索（FAISS 向量 + 关键词 + RRF 融合）
↓
Step 3：Graph Enhanced Search（沿 REFERENCES 边扩展跨 spec chunk）
↓
Step 4：NotebookLM 风格回答（强制内联引用 [1][2] + References 表）
```

## 3.2 Step 1：多查询构造
`abbreviation_resolver.py` + `rag_search_engine.expand_abbreviations()`：
1. 从 `data/dictionary/abbreviations.md` 加载缩写字典
2. 识别问题中的3GPP缩写（`LTM → L1/L2 Triggered Mobility`）
3. 生成3-5个不同视角子查询：
    - 缩写 + 全称
    - 相关技术术语
    - 相关协议层
    - 相关概念
多查询一次脚本调用并行检索，RRF合并，避免重复加载模型/索引。

## 3.3 Step 2：混合检索
`rag_search_engine.py` 实现 FAISS + 关键词 + RRF：
### 3.3.1 向量检索
```python
query_vec = model.encode([query], normalize_embeddings=True)
scores, ids = faiss_index.search(query_vec, k)
# 从 metadata + content_map 取完整内容
```
### 3.3.2 关键词检索
直接扫描chunk JSON文件，按title/path/content关键词命中加权：
```python
total_score = title_matches * 3 + path_matches * 2 + content_matches * 1
```
标题命中权重最高（×3），路径次之（×2），内容最低（×1）。

### 3.3.3 RRF融合
```python
rrf_score = weight / (rrf_k + rank)  # rrf_k=60
```
`keyword_weight = vector_weight = 0.5`。多查询模式下，每个子查询的keyword/vector结果分别累加RRF分数。

### 3.3.4 结果自动落盘
检索结果自动写入 `output/search_results.json`，含chunk完整content，LLM直接读文件作答，无需子agent。

## 3.4 Step 3：Graph Enhanced Search（关键差异化）
`graph_enhanced_search.py` 在RAG检索后必须执行：
1. 读取 `output/search_results.json` 获取RAG命中chunk
2. 对每个命中chunk，在 `xref_graph.json` 的 `adjacency` 中查 `REFERENCES` 边
3. 沿边发现跨spec的关联chunk（如38.331命中 → 沿REFERENCES发现38.214 clause 5.2.2、38.212 UCI encoding）
4. 过滤掉已命中spec，只保留cross-spec新发现
5. 把新发现chunk的content合并进结果

**价值**：纯向量检索容易漏掉跨规范的核心定义。例如问CSI-RS配置，向量检索可能只命中38.331的RRC IE，但CSI reporting的物理层细节在38.214、UCI编码在38.212——图扩展能把这些跨规范关联补回来。

## 3.5 Step 4：NotebookLM 风格回答
**强制规则**：
1. 内联引用：所有技术事实后加 `[1][2]`
2. References表：每个引用含 `Section / Section Hierarchy / Cited Content（原文摘录） / Relevance`
3. 结构化回答：概述 → 详细说明 → Cross-Layer Reference → 注意事项 → 核心摘要
4. 文档可信度优先级：详细说明 > 概述；官方定义 > 一般描述；Physical: `38.211 > 38.212 > 38.213 > 38.214`
5. 语言匹配：用问题语言回答，但 `Cited Content` 保留英文原文

# 4 关键设计要点总结
## 4.1 chunk 即内容
chunk JSON直接内嵌完整content，检索后无需回查原文，消除索引-内容不一致。这是相比“Tree只存索引”方案的根本改进。

## 4.2 离线图 vs 在线二次检索
交叉引用图离线构建一次，检索时只是查邻接表（O(1)），而非运行时正则提取 + 二次向量检索。可靠性高、延迟低、不依赖向量召回。

## 4.3 层级路径编入向量
嵌入文本 = `title + section_path + content[:500]`，把章节层级上下文编入向量表示，提升同主题不同章节的区分度。

## 4.4 多查询 RRF
缩写扩展 + 多视角子查询一次并行检索，RRF合并，既提升recall又避免重复加载开销。

## 4.5 IE节点与ASN.1感知
从ASN.1提取信息元素节点，建立跨spec的IE名字引用边，这对RRC/MAC层IE与物理层过程的关联特别有效。

# 5 数据目录结构
```
data/
├─ specs/
│  └─ chunks/
│     └─ 38331-i60.json  # 1文档 = 1 chunk JSON
├─ embeddings/
│  ├─ chunks.index         # FAISS
│  ├─ chunks_metadata.json
│  └─ chunks_content.json  # id → full content
└─ dictionary/
   └─ abbreviations.md      # 缩写字典

resources/3GPP_Wiki/graph/
├─ xref_graph.json  # 代码用（含 adjacency）
└─ xref_graph.md    # LLM 用摘要

output/
└─ search_results.json  # 检索结果自动落盘
```

# 6 局限性
- 嵌入模型较轻量（MiniLM 384-dim），语义表达能力弱于 BGE-M3
- 关键词检索是全量扫描chunk JSON，规模大时性能不佳（无倒排索引）
- 图扩展只做1-hop，且只沿 `REFERENCES` 边，未利用 `PARENT_CHILD/SIBLING` 做上下文补全
- 无 reranker 精排
- 无多语言翻译管线（依赖 LLM 直接按问题语言作答）

