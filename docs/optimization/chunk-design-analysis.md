# Chunk 设计现状分析与优化方向

> 以代码 + 当前 Milvus 索引实测数据为准（2026-08 抽样 61,323 chunks）。

## 状态更新（Stage 1 实施后）

| 项                                                  | 状态                                      |
| --------------------------------------------------- | ----------------------------------------- |
| content_type 收紧（去 `_contains_table_reference`） | ✅ 已实施（parameter_table 48%→40.8%）    |
| 大表行组语义单元化（table_max_chars 2000）          | ❌ 已回滚（实测 R@5 -4.6%，整表优于行组） |
| 图片 VLM/OCR 理解                                   | ⛔ 已改为"展示方案"（见 P0-3 更正）       |

> 下方正文为原始分析，P0-1 的"行组拆分"建议已被 Stage 1 实测推翻，保留仅为记录思考过程。

## 0. 数据事实基线

| 指标                  | 实测                                                                          |
| --------------------- | ----------------------------------------------------------------------------- |
| chunk 字符数          | 均值 1582 / 中位 1076 / p90 3863 / p99 7896 / **max 8000**                    |
| 碎片 (<200 字符)      | **8.0%**（尽管已有 min_chunk=300 合并）                                       |
| 大 chunk (>3000 字符) | **13.1%**                                                                     |
| content_type 分布     | overview 45.6% / **parameter_table 48.0%** / definition 2.9% / procedure 3.5% |

**表格/图片格式按数据源分布（关键区分）：**

| 数据源                  | 表格格式              | 数量   | 图片链接   | 数量      |
| ----------------------- | --------------------- | ------ | ---------- | --------- |
| 3GPP marked（36/38 系） | **pipe 表** `\|...\|` | 58,447 | `![](...)` | **3,803** |
| O-RAN marked            | **HTML 表** `<table>` | 214    | `![](...)` | 106       |

## 1. 问题清单（按影响排序）

### P0-1 表格 chunk 语义稀释 + content_type 过度分类（影响最大）

**现象**：

- `classify_chunk` 的 `_contains_table_reference()` 只要正文出现 "Table 6.3.3.2-1" 字样就把 chunk 判成 `parameter_table`，导致**近一半 chunk（48%）都是 parameter_table**。
- 后果：检索侧 `parameter_table ×1.2` 加权形同虚设（几乎全命中，无区分度）；且真正的参数表 chunk 被淹没在"只是引用了表号"的普通正文里。
- 一张几百行的参数表作为 5000+ 字符的原子 chunk 喂给 BGE-M3，向量被整体平均化——查询 "PRACH preamble format 4" 要对上一个含全部 format 的大表向量，天然匹配弱。

**根因**：

1. content_type 判定用"文本里有没有 Table 引用"这种弱信号，而非"chunk 本身是否以表结构为主体"。
2. 表格作为原子块整块嵌入，缺乏"行组 + 表头 + caption + section 上下文"的语义单元化。

**修复方向**：

1. content_type 收紧：只认真实表结构（pipe/grid/HTML 表分隔行 + 表头特征），去掉 `_contains_table_reference` 这条过度宽松的规则。
2. 大表按行组切成更小的自描述单元（每片带表头 + caption + 所在 section 路径），提升 dense 匹配精度。

### P0-2 公式块低语义密度

**现象**：`$$...$$` 数学块被原子保护成独立 chunk。采样到 `$$\Delta_{OTAREFSENS} = 41.1 - 10 \cdot \log_{10}(...)$$` 这类**孤立公式块**——几乎无语义文字，查询 "OTAREFSENS 定义" 匹配不上公式块，反而匹配到被切开的相邻正文。

**根因**：公式块被原子保护，切断了"公式 ←→ 变量含义 ←→ 紧邻说明文字"的语义关联。PHY 规范（38.211/212/213/214）公式密集，对 L1 查询致命。

**修复方向**：公式不自成 chunk，而是与"变量定义 + 所在 section 标题 + 紧邻说明"拼接后再嵌入（等价于给公式注入上下文）。

### P0-3 图片展示（更正：3GPP 是 jpg + 自带描述，非 emf + 信息丢失）

> ⚠️ 本节为对早期误判的更正。

**更正**：此前误判 3GPP 图片是 `.emf` 且信息全丢。实测：

- **38/36 系图片是 `.jpg`**（38 系 2970 张，浏览器可直接显示）；O-RAN 才是 `.emf`（106 张）。
- 图片的 **alt 文字本身就是详细描述**（如 "Diagram illustrating the uplink-downlink timing relation..."），
  且 `_clean_markdown` 已把它保留为 `[图: <描述>]` 存进 chunk。
- 故图片的**文字信息早就在 chunk 里、可被检索**（LLM 已能基于描述回答），丢的只是 jpg 文件链接（展示能力）。

**现状**：chunk 文本里是 `[图: Figure 5.3A-1: Definition of ...]`，图片路径（jpg）被 `_clean_markdown` 删除。

**修复方向（展示而非理解，无需 VLM/OCR）**：

1. 方案 A（重摄入）：`_clean_markdown` 保留 jpg 链接 + 图片目录静态托管 + 前端渲染 `![](url)`。
2. 方案 B（不重摄入）：离线构建 `{Figure 编号/描述 → jpg 路径}` 映射 + 静态托管 + 前端反查渲染。

> 收益是**用户体验（直接看图）**，**不是 Recall**（描述已在 chunk）。O-RAN 的 emf 例外，量小暂缓。

### P1-1 HTML 表格原子保护缺失（仅 O-RAN，小问题）

**现象**：`_protect_atomic_blocks` / `_segment_by_atomic_blocks` 只保护 pandoc 的 Grid/Pipe/Math 块，**不保护 HTML `<table>`**。O-RAN marked 数据是 HTML 表格（214 个），prose 切分时可能把 HTML 表从中间切断。

**根因**：splitter 是为 pandoc 输出（DOCX→Grid/Pipe）设计的，未覆盖 HuggingFace 数据集的 HTML 表格式。已有 `_split_html_table_rows` 处理"超大 HTML 表"，但原子保护环节漏了 HTML。

**修复方向**：在原子块检测中增加 HTML table（`<table>...</table>` 或 `<tr>` 行组）识别。低优先级（仅 214 张 O-RAN 表）。

### P1-2 碎片 chunk（8%）

**现象**：min_chunk=300 合并后仍有 8% <200 字符。这些多为**小表格/小公式原子块**（`_is_atomic_chunk` 保护、不允许合并），孤立存在。

**修复方向**：小原子块（<200 字符的小表/小公式）允许与紧邻正文合并，或注入上下文再嵌入（与 P0-2 同源）。

## 2. 与嵌入文本构成的关系

上述 P0-1/P0-2/P1-2 的共性是：**chunk 缺少自描述上下文**。这正与 `docs/optimization/retrieval-quality-analysis.md` 的 P0-1（embedding_text 是否编入 `section_path`）是同一根问题的两个切面：

- **嵌入层**：把 `section_path + section_title` 拼进向量（正在 A/B 验证）。
- **分块层**：让 chunk 本身自描述（表带 caption+表头、公式带变量定义、图带描述）。

两者互补，不是替代。

## 3. 建议执行顺序

| 轮次 | 任务                                                       | 成本      | 依赖           |
| ---- | ---------------------------------------------------------- | --------- | -------------- |
| 0    | content_type 收紧（去 `_contains_table_reference` 误分类） | ✅ 已实施 | 无             |
| 1    | 大表行组语义单元化（table_max_chars 2000）                 | ❌ 已回滚 | 实测 R@5 -4.6% |
| 2    | HTML table 原子保护                                        | 重切分    | 无             |
| 3    | 图片展示方案（保留 jpg 链接 + 静态托管 + 前端渲染）        | 低-中     | 无需 VLM/OCR   |

> 注：轮次 0-1 已由 Stage 1 验证（收紧生效、拆分回滚）。轮次 3 已从"VLM/OCR 理解"改为"展示方案"，收益是用户体验而非 Recall（图片描述已在 chunk 内）。与检索层的 BM25 分词、多样性重排、跨协议 max_per_spec=1 共同构成优化面。
