对比 3gpp-rag-rel18-main 与当前 3GPP_RAG_project 在文档切片、元数据、交叉引用、嵌入与检索流程上的差异，定位协议 recall 不足的根因，并给出可落地的改进建议与优先级排序。

# 1 问题陈述
当前项目对 3GPP 协议的 recall 很不理想：很多问题难以回答好，甚至有些回答与实际完全不搭。
经对比分析 `open_sources/3gpp-rag-rel18-main` 项目，发现其在文档预处理与检索调用方面有几处独到设计，正是当前项目缺失或薄弱的环节。

# 2 逐项对比
## 2.1 文档切片
|维度|当前项目（`src/ingestion/splitter.py`）|3gpp-rag-rel18（`section_chunker.py`）|差距|
| ---- | ---- | ---- | ---- |
|切分依据|Markdown 标题树 + 叶子节点|Markdown 标题树 + 叶子节点|基本一致|
|表格/公式保护|有（Grid/Pipe/Math 原子块占位符）|无专门保护|当前项目更优|
|超长拆分|段落边界 + 字节上限自适应|`max_lines` 强制拆 + `_p1/_p2`|当前项目更细致|
|chunk 自身 `section_number`|未保留|保留（`section_number`）|关键缺失|
|`section_path` 层级路径|未保留（只有 `parent_title`）|保留（`7 > 7.1 > 7.1.1`）|关键缺失|
|`chunk_id` 可读性|自增数字|`{doc_id}_{section_number}`|rel18 更可追溯|
|content 内嵌|是（Milvus 存储）|是（chunk JSON）|一致|

## 2.2 交叉引用
|维度|当前项目（`src/retriever/cross_ref.py`）|3gpp-rag-rel18（`build_xref_graph.py`）|
| ---- | ---- | ---- |
|构建时机|运行时（检索后正则提取）|离线（摄入时一次构建）|
|实现方式|正则提取引用 → 二次向量检索|离线图 + 邻接表 O(1) 查询|
|可靠性|依赖向量能否召回被引用章节|确定性图边，不依赖向量|
|延迟|每引用一次二次检索（高）|邻接表查询（极低）|
|边类型|无（只有"引用→二次检索"）|`REFERENCES/PARENT_CHILD/SIBLING/NEXT`|
|IE/ASN.1 感知|无|有（IE 节点 + `DEFINES` 边）|
|跨 spec 发现|依赖二次向量检索 recall|沿 `REFERENCES` 边确定性发现|

## 2.3 嵌入
|维度|当前项目|3gpp-rag-rel18|差距|
| ---- | ---- | ---- | ---- |
|模型|BGE-M3 (1024-dim，强)|MiniLM (384-dim，轻)|当前项目更优|
|嵌入文本|chunk 原文|`title + section_path + content[:500]`|rel18 更优|
|层级路径输入向量|否|是|关键差距|

## 2.4 检索流程
|维度|当前项目|3gpp-rag-rel18|差距|
| ---- | ---- | ---- | ---- |
|混合检索|Milvus Dense + Python BM25 + RRF|FAISS + 关键词扫描 + RRF|当前项目更优（有倒排）|
|Reranker|Cross-Encoder 精排（有）|无|当前项目更优|
|多查询|LLM 查询扩展（单查询改写）|缩写字典 + 3-5 视角子查询 RRF|rel18 多查询更系统|
|Spec-aware|有（规范号定向检索 + 加权）|无|当前项目更优|
|多跳|有（`multi_hop.py`）|无|当前项目更优|
|图增强|无|有（Graph Enhanced Search）|关键差距|
|相邻 chunk 扩展|有（`get_adjacent_chunks`）|无|当前项目更优|
|在线补充|有（Google/TSpec-LLM）|无|当前项目更优|

## 2.5 回答生成
| 维度       | 当前项目    | 3gpp-rag-rel18                                | 差距        |
| -------- | ------- | --------------------------------------------- | --------- |
| 引用强制     | 软引导     | 强制内联 `[1][2]` + References 表                  | rel18 更严格 |
| 文档可信度优先级 | 无       | Physical: `38.211 > 38.212 > 38.213 > 38.214` | rel18 更精准 |
| 多语言      | 有（翻译管线） | 依赖 LLM 直接匹配                                   | 当前项目更优    |

# 3 recall 不足根因分析
结合对比，当前项目"回答与实际完全不搭"的核心根因按影响排序：

## 3.1 根因 1：交叉引用是"在线二次检索"而非"离线图"（影响最大）
当前 `cross_ref.py` 在检索后用正则从命中 chunk 提取引用，再对每个引用发起二次向量检索。问题：
1. 二次检索本身可能召回不到被引用章节（向量检索的 recall 限制），导致跨 spec 核心定义丢失
2. 延迟高：每个引用一次 embedding + 检索，串行执行
3. 只扫 Top-10 chunk 的引用，覆盖有限
4. 无 IE/ASN.1 感知：RRC IE 与物理层过程的关联无法建立

3GPP 规范的精髓就是跨规范引用（38.331 RRC → 38.214 物理层 → 38.212 编码）。当前方式无法可靠建立这条链路，是"回答不搭"的最大原因。

## 3.2 根因 2：chunk 元数据缺失 `section_number` / `section_path`
当前 chunk 只有 `parent_section_id` / `parent_title`，没有 chunk 自身的 `section_number` 和层级路径。后果：
1. 嵌入向量缺少层级上下文：同主题不同章节的 chunk 向量区分度不足
2. 无法做 section 精确匹配：图构建、spec-aware 定向检索都依赖 `section_number`
3. 引用追溯困难：回答中无法精确标注 `§7.1.1`

## 3.3 根因 3：嵌入文本未输入层级路径
当前嵌入文本 = chunk 原文。rel18 把 `section_title + section_path + content[:500]` 编入向量。后果：
- "PUSCH power control" 的 chunk 和 "PUCCH power control" 的 chunk 向量可能很接近（都是 power control），但 `section_path` 能区分上下文
- 层级路径能帮助向量模型理解 chunk 在规范中的位置

## 3.4 根因 4：无图增强检索
即使有 cross_ref 二次检索，它也是"被动的"（只扩展命中 chunk 里显式写出的引用）。rel18 的图增强是"主动的"：沿 `REFERENCES` 边发现跨 spec 关联，包括 IE 名字匹配建立的隐式关联。

# 4 改进建议（按优先级排序）
## 4.1 P0：构建离线 Cross-Reference Graph
目标：把交叉引用从"在线二次检索"改为"离线图 + 邻接查询"，这是提升 recall 最关键的改进。

### 4.1.1 为什么需要离线图
当前 `cross_ref.py` 的在线方式有三个致命问题：
1. 二次检索依赖向量 recall：提取到 TS 38.214 clause 5.2.1 后，用向量检索找该章节，但向量检索本身可能召回不到 → 跨 spec 关联丢失
2. 延迟高：每个引用一次 embedding + 检索，串行执行，5 个引用 = 5 次检索
3. 无结构边：只有"引用→二次检索"，没有 `PARENT_CHILD/SIBLING/DEFINES` 等结构关系

离线图把所有引用关系摄入时一次构建，检索时只查邻接表（O(1)），确定性、低延迟、不依赖向量。

### 4.1.2 图的数据模型

**节点**
|节点类型|id 格式|含义|
| ---- | ---- | ---- |
|SPEC_SECTION|`{doc_id}_{section_number}`|一个 chunk = 一个节点|
|IE（可选）|`IE_{ie_name}`|ASN.1 信息元素定义|

**边**
|边类型|权重|作用|
| ---- | ---- | ---- |
|REFERENCES|cross-spec 1.0 / 同 spec 0.6|跨规范引用（最强连接）|
|PARENT_CHILD|0.5|章节树结构|
|SIBLING|0.3|同父兄弟章节|
|NEXT_SECTION|0.4|顶层连续编号章节|
|DEFINES（可选）|0.8|section → IE 定义|

### 4.1.3 构建流程（5 阶段）
**阶段 1：节点生成 + 索引构建**
新增 `src/ingestion/xref_graph_builder.py`。扫描所有 chunk（需先完成 P1 的元数据增强，chunk 携带 `section_number`），为每个 chunk 创建节点。同时构建索引：
- `section_index`：`(spec, section_number) → [chunk_id, ...]`（核心索引，用于引用解析）
- `doc_to_spec`：`doc_id → spec`
- `parent_section_ie_index`：`(spec, parent_section) → [chunk_id]`（IE fallback）

**阶段 2：结构边生成**
- `PARENT_CHILD`：遍历 chunk 的 `parent_section_number`，通过 `section_index` 找父 chunk，生成父子边
- `SIBLING`：按 parent 分组，`section_number` 排序，相邻兄弟生成边
- `NEXT_SECTION`：顶层章节按编号排序，连续编号（n2==n1+1）生成边

**阶段 3：引用边生成（REFERENCES）— 核心**
对每个 chunk 的 content（跳过 §2 References 章节），用多模式正则提取引用：
```python
# 模式 0（最高精度）："Clause X.Y.Z of TS 38.AAA"
CLAUSE_OF_TS = r'(?:Clause|clause|section|Section)\s+([\d\.A-Za-z]*)\s+of\s+(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})'

# 模式 0b："Clause X.Y.Z of [N, TS 38.AAA]"
CLAUSE_OF_BRACKET = r'(?:Clause|clause|section|Section)\s+([\d\.A-Za-z]*)\s+of\s+\[(\d+),\s+(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})\]'

# 模式 1："TS 38.214" + 邻近 clause
SPEC_REF = r'(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})'
# 匹配后在 ±50~100 字上下文找 clause

# 模式 2："[6, TS 38.214]"
BRACKET_REF = r'\[(\d+),\s+(?:3GPP\s+)?TS\s+(\d{2}\.\d{3})\]'

# 模式 3："clause 9.2.5.4"（同 spec，排除同行有 TS 引用）
SAME_SPEC_CLAUSE = r'(?:clause|section|subsection)\s+([\d\.A-Za-z]*)'
```
关键策略：
- 去重：高精度模式匹配后标记区间，低精度模式跳过已匹配区间
- 同 spec 排除：模式 3 匹配时检查同行是否有 TS 引用，有则跳过（留给模式 0/1）
- NR 过滤：只处理 38.xxx
- section → chunk 匹配：精确 > 前缀（子章节）> 上级回退 > 上级前缀
- 边去重：`(from, to, target_spec, clause)` 四元组去重

**阶段 3.5：IE 名字匹配（可选，隐式跨 spec 关联）**
- 离线提取每个 spec 的 ASN.1 IE 名字集合
- 对含 TS 引用的 chunk，在 ±300 字上下文搜索目标 spec 的 IE 名字
- 命中建立 `chunk → IE_chunk` 的 `REFERENCES` 边（±300 字内 0.8，全文 0.5）

**阶段 4：IE 定义提取（可选）**
- ASN.1 正则 `(\w[\w-]*)\s*::=\s*(INTEGER|SEQUENCE|...)` 提取 IE
- 生成 `IE_{name}` 节点 + `section → IE` 的 `DEFINES` 边

**阶段 5：adjacency 邻接索引**
```python
adjacency = defaultdict(lambda: defaultdict(list))
for edge in all_edges:
    adjacency[edge["from"]][edge["type"]].append(edge["to"])
```
输出 `data/processed/xref_graph.json`，含 `nodes/edges/adjacency`。

### 4.1.4 检索时集成
新增 `src/retriever/graph_expander.py`，在 `pipeline.py` 的 Step 3.5 替换当前 `_resolve_cross_refs`
```python
# 旧：正则提取 + 二次向量检索（慢、不可靠）
# 新：邻接表查询（O(1)，确定性）
class GraphExpander:
    def __init__(self, graph_path):
        self.graph = json.load(open(graph_path))
        self.adjacency = self.graph["adjacency"]
        self.node_map = {n["id"]: n for n in self.graph["nodes"]}

    def expand(self, results, max_per_chunk=5):
        """命中 chunk 沿 REFERENCES 边扩展跨 spec 关联。"""
        seen_ids = {r.chunk_id for r in results}
        expanded = []
        for r in results[:10]:  # Top 10
            adj = self.adjacency.get(r.chunk_id, {})
            for ref_id in adj.get("REFERENCES", [])[:max_per_chunk]:
                if ref_id in seen_ids:
                    continue
                node = self.node_map.get(ref_id, {})
                # 只保留 cross-spec 新发现
                if node.get("spec") != r.spec_number:
                    chunk = self._load_chunk(ref_id)
                    if chunk:
                        expanded.append(chunk)
                        seen_ids.add(ref_id)
        return expanded
```
`pipeline.py` 集成：
```python
# Step 3.5：图增强检索（替换 _resolve_cross_refs）
if self._graph_expander:
    expanded = self._graph_expander.expand(results)
    results = results + expanded
```

### 4.1.5 预期收益
|指标|当前（在线二次检索）|改进后（离线图）|
| ---- | ---- | ---- |
|跨 spec 关联可靠性|依赖向量 recall|确定性图边|
|延迟|每引用一次检索（~200ms/次）|邻接查询（~1ms）|
|边类型|1 种（引用）|5 种（`REFERENCES/PARENT_CHILD/SIBLING/NEXT_SECTION/DEFINES`）|
|IE/ASN.1 感知|无|有|
|跨 spec 发现|被动（只扩展显式引用）|主动（沿 REFERENCES 边 + IE 名字匹配）|

### 4.1.6 实施注意
- 依赖 P1：图构建需要 chunk 携带 `section_number`，必须先完成元数据增强
- 正则调优：3GPP 文档格式多样，正则需在真实数据上验证，可能有漏匹配/误匹配
- 图规模评估：全量 38.xxx 系列的 adjacency 可能较大，需评估内存占用（可按需加载）
- 增量更新：新增文档时需增量更新图（重新构建或合并）

## 4.2 P1：chunk 元数据增强 `section_number` / `section_path`
目标：让每个 chunk 携带自身 `section_number` 和层级路径。

实现：修改 `src/ingestion/splitter.py` 的 `HeaderAwareSplitter`：
1. `SectionNode` 已有 `sec_id`，在生成 Chunk 时传入
2. Chunk dataclass 新增字段：
    - `section_number: str`（如 `"7.1.1"`）
    - `section_title: str`（如 `"UE behaviour"`）
    - `section_path: str`（如 `"7 Uplink Power control > 7.1 PUSCH > 7.1.1 UE behaviour"`）
3. `_split_by_tree` 中为叶子节点构造 `section_path`（沿 parent 链向上拼接）
4. Milvus schema 新增对应标量字段
5. 需要重建索引（reindex）

预期收益：
- 为 P0 的图构建提供 `section_number` 基础
- 嵌入文本可编入层级路径（见 P2）
- 回答可精确标注 `§section_number`

## 4.3 P1：嵌入文本编入层级路径
目标：把 `section_title + section_path` 输入向量。

实现：修改 `src/ingestion/embedder.py` / `mps_embedder.py` 的嵌入文本构造：
```python
# 旧
embed_text = chunk.text

# 新
embed_text = f"{chunk.section_title} {chunk.section_path} {chunk.text[:500]}"
```
预期收益：同主题不同章节的向量区分度提升，减少"power control"类查询把 PUSCH/PUCCH 混淆的问题。

注意：需要重建索引。

## 4.4 P2：多查询 RRF 检索
目标：从"单查询 LLM 改写"升级为"多视角子查询并行 RRF"。

实现：参考 rel18 的 `multi_query_search`：
1. 维护 3GPP 缩写字典（可从规范 §3.1 提取）
2. 查询时识别缩写，生成 3-5 个子查询：
    - 缩写 + 全称
    - 相关技术术语
    - 相关协议层
3. 每个子查询独立 Dense+BM25 检索
4. RRF 合并所有子查询结果

预期收益：提升 recall，特别是缩写类查询（如"LTM procedure"）。

注意：当前已有 `_taxonomy_decompose_search`（分类分解），可扩展为通用多查询。

## 4.5 P2：图增强检索集成到 pipeline
目标：在 pipeline 中集成图扩展，作为 cross_ref 的替代。

实现：`pipeline.py` 的 Step 3.5：
```python
# 旧
results = self._resolve_cross_refs(results, max_refs=5)

# 新
results = self._graph_expand(results)  # 邻接表查询
```

`_graph_expand` 逻辑：
1. 加载 `xref_graph.json`
2. 对 Top-N 命中 chunk，查 adjacency 的 `REFERENCES` 边
3. 过滤已命中 spec，只保留 cross-spec 新发现
4. 加载新 chunk 的 content，合并进结果

预期收益：确定性发现跨 spec 关联，不依赖向量 recall。

## 4.6 P3：回答生成强化引用与可信度优先级
目标：减少"回答不搭"。

实现：修改 `src/generator/prompt.py`：
1. 强制内联引用格式 `[1][2]`
2. 加入文档可信度优先级提示：
    > 文档优先级：详细说明 > 概述；官方定义 > 一般描述
    > Physical: 38.211 > 38.212 > 38.213 > 38.214
3. References 表格式：`Section / Section Hierarchy / Cited Content / Relevance`

预期收益：LLM 生成时更倾向引用高可信度章节。

# 5 实施路线图
|阶段|任务|依赖|预期收益|
| ---- | ---- | ---- | ---- |
|阶段 1|P1 chunk 元数据增强 + 重建索引|无|为图构建打基础|
|阶段 2|P0 离线 Xref Graph 构建|阶段 1|跨 spec 引用确定性|
|阶段 3|P1 嵌入文本输入路径 + 重建索引|阶段 1|向量区分度提升|
|阶段 4|P2 图增强检索集成|阶段 2|跨 spec 召回提升|
|阶段 5|P2 多查询 RRF|无|recall 提升|
|阶段 6|P3 回答引用强化|无|减少不搭回答|

# 6 预期效果
实施 P0 + P1 后，预期解决以下典型问题：
|问题类型|当前表现|改进后预期|
| ---- | ---- | ---- |
|跨 spec 查询（如 CSI-RS 配置）|只命中 RRC IE，漏物理层|沿 REFERENCES 边发现 38.214/38.212|
|缩写查询（如 LTM）|召回不全|多查询 RRF 覆盖全称 + 相关术语|
|同主题混淆（PUSCH vs PUCCH）|向量接近|`section_path` 编入向量区分|
|引用追溯|无法精确标注 `§section`|chunk 携带 `section_number`|
|回答不搭|无可信度优先级|Physical spec 优先级提示|

# 7 风险与注意
- 重建索引成本：P1/P3 需要全量 reindex，3GPP 规范量大时耗时
- 图构建正则精度：引用提取正则需针对 3GPP 文档格式调优，可能有漏匹配/误匹配
- 图规模：全量 38.xxx 系列的图可能较大，需评估 adjacency 内存占用
- 向后兼容：chunk 元数据新增字段需兼容现有 Milvus collection（可能需重建 collection）

# 8 当前项目已具备的优势（无需改动）
对比当中当前项目也有明显优势，改进时应保留：
- BGE-M3 嵌入模型（1024-dim）强于 rel18 的 MiniLM
- Cross-Encoder Reranker 精排
- 表格/公式原子保护切片
- Milvus 原生混合检索（有倒排，性能优于全量扫描）
- Spec-aware 定向检索 + 加权
- 多跳检索
- 相邻 chunk 上下文扩展
- 在线搜索补充
- 多语言翻译管线

改进方向是补齐离线图 + 元数据增强，而非替换现有管线。