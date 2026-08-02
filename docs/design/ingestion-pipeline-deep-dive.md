# CommSpec RAG 文档摄入管线技术详解

> 从 Word 格式的通信协议规范，到 Milvus 向量数据库中可供 AI 检索的向量数据——全流程逐层拆解。

---

## 目录

1. [概述](#1-概述)
2. [源文档](#2-源文档)
3. [Pandoc 转换](#3-pandoc-转换)
4. [元数据提取](#4-元数据提取)
5. [Markdown 后处理](#5-markdown-后处理)
6. [分块策略](#6-分块策略)
7. [分类标注](#7-分类标注)
8. [嵌入生成](#8-嵌入生成)
9. [入库 Milvus](#9-入库-milvus)
10. [Markdown 的存在形式](#10-markdown-的存在形式)
11. [增量管理](#11-增量管理)
12. [完整串联：TS 38.211](#12-完整串联ts-38211)
13. [附录](#附录)

---

## 1. 概述

### 1.1 什么是 RAG

**RAG**（Retrieval-Augmented Generation，检索增强生成）是一种让大语言模型（如 GPT、DeepSeek）能够回答"它训练时没见过"的问题的技术。

大语言模型的训练数据有截止日期，且不包含企业内部文档、行业标准等私有/专业数据。RAG 的思路是：**用户提问时，先在知识库中检索最相关的文档片段，再把它们连同问题一起交给大模型，让它基于这些材料作答。**

```
用户提问 → [检索] 从知识库找最相关的文档片段 → [增强] 把片段注入模型上下文 → [生成] 模型基于片段回答
```

本项目的知识库是 3GPP / O-RAN 通信协议规范，本文档专门讲解其中**"把规范文档变成可检索的知识"**这一环节——即从 Word 文档到向量数据库的全流程。

### 1.2 本文档在系统全景中的位置

```
┌──────────────────────────────────────────────────────┐
│                   CommSpec RAG 系统                     │
│                                                        │
│  ① 摄入管线（本文档主题）     ② 检索管线    ③ 生成管线  │
│  DOCX → Chunk → Milvus       Query → Top-K    LLM 回答 │
│       ↑                              ↑            ↑    │
│       └── 你正在阅读的部分 ──────────┘            │    │
└──────────────────────────────────────────────────────┘
```

- **摄入管线**（本文档）：负责"建库"，把文档预处理后存入 Milvus。离线执行，一次性或增量。
- **检索管线**：接收用户问题，在 Milvus 中做语义搜索，返回最相关的 Top-K 个 chunk。在线实时。
- **生成管线**：把检索结果注入提示词模板，调用 LLM 生成最终回答。在线实时。

没有摄入管线产出的高质量向量数据，检索和生成都无法正常工作——它是整个 RAG 系统的地基。

### 1.3 一句话理解

这条数据流水线的本质工作是：**把 Word 格式的通信协议规范（3GPP / O-RAN），变成 AI 能够"读懂"并快速检索的向量数据**。

一份 TS 38.211 规范有几百页，包含物理信道的帧结构、资源网格、调制方式等密集技术参数。人类工程师可以翻目录、跳章节来查找信息，但 AI 不能直接"阅读" Word 文档——它需要文档被切分成小段，每段转换成一串数学向量（embedding），存入专门做"语义搜索"的数据库。当用户问"PUSCH 的 DMRS 资源映射是怎样的？"，系统在向量数据库中查找语义最接近的文档片段（chunk），然后把最相关的几段注入到大语言模型上下文中。

### 1.4 端到端数据流全景

```mermaid
graph LR
    A["DOCX 文件"] --> B["Pandoc 转换<br/>DOCX → Markdown"]
    B --> C["Markdown 后处理<br/>清理冗余"]
    C --> D["标题感知分块<br/>按章节切分 + 原子保护"]
    D --> E["规则分类<br/>标注类型/角色/领域"]
    E --> F["嵌入生成<br/>文本 → 1024维向量"]
    F --> G["入库 Milvus<br/>向量 + 元数据持久化"]
```

| 步骤 | 输入 | 输出 | 核心模块 |
|------|------|------|----------|
| Pandoc 转换 | `38211-60.docx` | Markdown 文本字符串 | `PandocExtractor` |
| 后处理 | 原始 Markdown（含 TOC 链接、属性标记） | 干净的 Markdown | `_postprocess()` |
| 分块 | 整篇 Markdown（数十万字） | Chunk 对象列表 | `HeaderAwareSplitter` |
| 分类 | 每个 Chunk 的文本和标题 | 带 content_type/spec_role/topic_domain 标签 | `classify_chunk()` |
| 嵌入 | 每个 Chunk 的纯文本 | 1024 维浮点数向量 | `BatchEmbedder` |
| 入库 | Chunk（含文本、向量、元数据） | Milvus Collection 记录 | `MilvusStore.insert()` |

### 1.5 前置概念

| 概念 | 通俗解释 |
|------|----------|
| **Markdown** | 用纯文本表示格式的标记语言。`# 标题` 表示一级标题，`|列1|列2|` 表示表格。 |
| **Chunk** | 从规范中切出的一小段文本。比如 TS 38.211 第 6.2.3 节就是一个 chunk。 |
| **Embedding** | 把文字转成固定长度的数字（本项目 1024 个 float32）。语义相近 → 向量也近。 |
| **Milvus** | 专门存储和检索向量的数据库。类比 MySQL 存数字和字符串，Milvus 额外能存 1024 维向量并做"最近邻居"查询。 |
| **BGE-M3** | 本项目使用的嵌入模型（BAAI 开源），支持中英韩多语言，输出 1024 维向量，最多处理 8192 token。 |

---

## 2. 源文档

### 2.1 文档来源

原始输入是 3GPP 和 O-RAN 联盟发布的标准规范 DOCX 文件，存放在 `data/documents/` 目录下：

```
data/documents/
├── R18/                        # 3GPP Release 18 规范
│   ├── 21_series/ ~ 24_series/ # 需求/服务/技术实现/信令协议
│   ├── 36_series/              # LTE 无线接入
│   └── 38_series/              # NR 无线接入（共 148 个文件）
│       ├── 38211-60.docx       # TS 38.211 物理信道与调制
│       ├── 38212-i40.docx      # TS 38.212 复用与信道编码
│       ├── 38300-60.docx       # TS 38.300 NR 总体描述
│       └── ...
└── ORAN/                       # O-RAN 联盟规范
    └── O-RAN.WG4.TS.CUS.0-R005-v21.00.docx
```

### 2.2 3GPP 文件命名规则

以 `38211-60.docx` 为例：

```
38211-60.docx
│││││ ││
│││││ │└── 内部版本标识（60 = V18.4.0 对应的文件版本号）
│││││ └─── 分隔符
││││└──── 规范号第 3 位（1）→ TS 38.211
│││└───── 规范号第 2 位（1）
││└────── 规范号第 1 位（2）
│└─────── Series 号第 2 位（8）
└──────── Series 号第 1 位（3）
```

解析规则——取前 5 位数字：`38211` → spec=`38.211`，series=`38`。

> ⚠️ 文件名中的版本号（`60`、`i40`）是 3GPP 内部文件版本，**不是 Release 号**。Release 必须从文档内容头部提取（V18.4.0 → R18），详见第 4 章。

### 2.3 O-RAN 文件命名规则

`O-RAN.WG4.TS.CUS.0-R005-v21.00.docx`：
- 规范编号：`O-RAN.WG4.CUS.0`
- 版本/Release：`v21.00`（O-RAN 不区分 Release，直接用版本号标识）

---

## 3. Pandoc 转换

### 3.1 Pandoc 是什么

**Pandoc** 是一个开源的"文档瑞士军刀"——在几十种文档格式间互相转换。和普通转换工具不同的是，Pandoc 理解文档的**结构**而非**排版**。Word 里一个 14pt 加粗宋体的段落，Pandoc 不是看字体字号，而是解析其 Word 样式（如 "Heading 1"），理解"这是一个一级标题"，然后用 Markdown 的 `#` 语法来表达。

### 3.2 内部机制：AST 抽象语法树

Pandoc 的转换分两步，中间经过 AST（Abstract Syntax Tree，抽象语法树）：

```mermaid
graph LR
    A["DOCX 文件"] --> B["DOCX Reader<br/>解析 Word 样式"]
    B --> C["Pandoc AST<br/>与格式无关的<br/>文档结构树"]
    C --> D["Markdown Writer<br/>输出纯文本"]
    D --> E["Markdown 文本"]
```

**AST** 把文档表示为一棵"内容树"：根节点是 Document，子节点是各级标题（Header）、段落（Para）、表格（Table）、数学公式（Math）、图片（Image）等。每个节点记录内容+结构，不记录字体颜色等视觉属性。

有了 AST，Markdown Writer 只需遍历这棵树，把每种节点"翻译"成对应 Markdown 语法。只要 Word 样式用对了，Pandoc 就能准确识别层级关系。

### 3.3 各类内容的转换方式（以 TS 38.211 为例）

#### 标题 → `#` 标题

| Word 样式 | Markdown 输出 |
|-----------|--------------|
| Heading 1 | `# 6 NR radio interface` |
| Heading 2 | `## 6.1 General system architecture` |
| Heading 3 | `### 6.2.1 OFDM baseband signal generation` |

#### 表格 → Grid Table

这是 Pandoc 的**最大优势**。TS 38.211 Table 6.2.3-1（Resource grid parameters）经 Pandoc 输出为：

```
+------------------+------------------------------------------+-------------------+
| Parameter        | Description                              | Value             |
+==================+==========================================+===================+
| N_RB             | Number of RBs in the resource grid       | See clause 4.4.2  |
+------------------+------------------------------------------+-------------------+
| N_sc^RB          | Number of subcarriers per resource block | 12                |
+------------------+------------------------------------------+-------------------+
```

- `+---+` 定义列边界，`+===+` 定义表头分隔
- 行列结构完整保留，后续分块器将其标记为"原子块"，永不切割

#### 公式 → LaTeX Math

Word 中 OLE 嵌入的公式被转为标准 LaTeX 语法：

| Word 中看到的公式 | Pandoc 输出 |
|-------------------|------------|
| \(\mathbf{\rho}_{A}^{d}\) | `$\mathbf{\rho}_{A}^{d}$` |
| \(s_{l}^{(p,\mu)}(t)\) | `$s_{l}^{(p,\mu)}(t)$` |

#### 图片 → Markdown 引用

```
![Figure 6.2.1-1: OFDM baseband signal generation chain](media/image5.emf)
```

图片以 `.emf` 格式提取，Pandoc 不修改文件本身，只生成引用路径。

### 3.4 调用参数详解

```
pandoc 38211-60.docx -t markdown --wrap=none --markdown-headings=atx
```

| 参数 | 含义 | 为什么 |
|------|------|--------|
| `-t markdown` | 输出标准 Markdown | 含 Grid Table 和 LaTeX math。不选 `-t gfm`，因为 GFM 不支持 Grid Table |
| `--wrap=none` | 不自动折行 | **关键**。默认 72 字符折行会把表格 `+---+` 边框折断，导致分块器无法识别 |
| `--markdown-headings=atx` | `#` 风格标题 | 比默认下划线风格更容易被正则匹配 |

超时设 120 秒，大型规范（如 TS 38.331 RRC 规范，500+ 页）转换可能需近 1 分钟。

### 3.5 为什么选 Pandoc 而不是 mammoth

| 维度 | Pandoc | mammoth |
|------|--------|---------|
| 表格处理 | Grid Table，行列完整 | 简易文本，列对齐丢失 |
| 数学公式 | 标准 LaTeX math | OLE 转义字符，AI 不可读 |
| 标题层级 | 完整保留 Heading 1~9 | 部分深层标题丢失 |
| 转义污染 | 无 | 大量 `\.` `\(` `\)` 需清洗 |

### 3.6 局限性

1. **不保留排版信息**（对 RAG 场景反而是好事——只需内容语义）
2. **图片仅保留引用路径**，不转 base64
3. **需要单独安装** Pandoc（`brew install pandoc` / `apt-get install pandoc`）
4. **转换后是纯文本字符串**——存在于 Python 内存变量中，不存为磁盘 `.md` 文件（详见第 10 章）

---

## 4. 元数据提取

### 4.1 元数据字段全景表

系统为每个 chunk 维护 17 个字段：

| 字段 | 类型 | 含义 | 示例（TS 38.211） | 来源 |
|------|------|------|---------------------|------|
| `id` | INT64 主键自增 | Milvus 自动生成的唯一 ID | `45123456789` | Milvus 入库时 |
| `text` | VARCHAR(65535) | chunk 完整文本 | `"For each numerology..."` | 分块器 |
| `dense_vector` | FLOAT_VECTOR(1024) | BGE-M3 嵌入向量 | `[0.032, -0.145, ...]` | 嵌入生成 |
| `doc_id` | VARCHAR(256) | 文档唯一标识 | `38211-60` | 文件名（来源 1） |
| `series` | INT64 | 系列号 | `38` | 文件名（来源 1） |
| `spec_number` | VARCHAR(64) | 规范编号 | `38.211` | 文件名+文本头 |
| `release` | VARCHAR(32) | 3GPP Release | `R18` | 文本头（来源 2） |
| `doc_type` | VARCHAR(32) | 文档类型 | `3gpp` | 文件名（来源 1） |
| `section_number` | VARCHAR(64) | 当前章节编号 | `6.2.3` | 标题正则（来源 3） |
| `section_title` | VARCHAR(512) | 当前章节标题 | `Resource grid` | 标题（来源 3） |
| `section_path` | VARCHAR(1024) | 完整层级路径 | `6 > 6.2 > 6.2.3 Resource grid` | 章节树 |
| `parent_section_id` | VARCHAR(256) | 父章节编号 | `6.2` | 章节树 |
| `parent_title` | VARCHAR(1024) | 父章节标题 | `Physical resources` | 章节树 |
| `chunk_index` | INT64 | 同文档内 chunk 序号 | `12` | 分块器计数 |
| `content_type` | VARCHAR(32) | 内容形态标签 | `parameter_table` | 规则分类 |
| `spec_role` | VARCHAR(32) | 规范权威度标签 | `authoritative` | 规则分类 |
| `topic_domain` | VARCHAR(32) | 技术领域标签 | `phy_layer` | 规则分类 |

### 4.2 三个来源

**来源 1 — 文件名解析**（`_parse_filename()`）：
- `38211-60.docx` → 取前 5 位数字 `38211` → spec=`38.211`, series=`38`
- O-RAN 用正则 `O-RAN\.WG\d+\.\w+` 匹配

**来源 2 — 文本头解析**（`_parse_text_header()`）：
- 检查 Markdown 前 800 字符，匹配 `TS 38.211 V18.4.0` → release=`R18`
- **这是获取 Release 的唯一可靠途径**。文件名中的版本号 ≠ 3GPP Release

**来源 3 — 章节结构提取**（章节树构建时）：
- 从 Markdown 标题中解析 `section_number`、`section_title`
- 拼接 `section_path`、记录 `parent_*` 字段

### 4.3 双路径兜底

```
spec, release, version = self._parse_filename(filepath.name)      # 路径 1
text_spec, text_release, _ = self._parse_text_header(md_text)     # 路径 2
if not spec_number: spec_number = text_spec   # 文件名没解出 → 用文本头
if not release: release = text_release         # 文件名没解出 → 用文本头
```

### 4.4 TS 38.211 完整举例

文件 `38211-60.docx`，文本头 `3GPP TS 38.211 V18.4.0 (2024-09)`：

| 步骤 | 动作 | 结果 |
|------|------|------|
| ① 文件名解析 | `38211-60.docx` → spec=`38.211` | `doc_id="38211-60"`, `spec_number="38.211"`, `series=38`, `doc_type="3gpp"` |
| ② Pandoc | DOCX → Markdown | （text 来源） |
| ③ 文本头解析 | 匹配 `V18.4.0` | `release="R18"`, `version="18.4.0"` |
| ④ 分块 | 章节树切分 | `section_number`, `section_title`, `section_path`, `parent_*`, `chunk_index` |
| ⑤ 分类 | classify_chunk() | `content_type`, `spec_role`, `topic_domain` |

---

## 5. Markdown 后处理

Pandoc 的"忠实转换"会保留许多对 RAG 无意义的噪音。后处理清除以下五类内容：

### 5.1 移除目录导航链接

**问题**：Word 交叉引用被转成 `[5](#foreword)` 格式，锚点链接对 AI 无意义且浪费 token。

**处理**：正则 `\[([^\]]+?)\]\(#[^)]+\)` → 保留显示文字、去掉 URL。`[5](#foreword)` → `5`。

### 5.2 移除标题属性

**问题**：Pandoc 在每个标题后附加 `{#_Toc123456 .unnumbered}` 用于内部锚点。这对嵌入模型是噪音——标题后的乱码会被编码进向量。

**处理**：正则 `\{[#\.][^}]+\}` 删除所有属性标记。

### 5.3 图片智能处理

| 图片类型 | 示例 | 处理 |
|----------|------|------|
| 有 Figure 编号+描述的图片 | `![Figure 6.2.1-1: OFDM...](media/image5.emf)` | **保留**（有语义信息） |
| 封面多图连排 | `![](image1.emf) ![](image2.emf)` | **删除**（纯装饰） |
| 孤立无标题图片 | `![](image9.emf)` 无相邻 Figure 标题 | **删除**（无上下文） |

### 5.4 裁剪 Change History Annex

3GPP 规范末尾的版本变更记录（"Version 18.0.0: First release → Version 18.4.0: Added clause X"）是纯历史记录，无现行技术参考价值。系统检测 Annex 标题 + 变更记录特征行，从此处截断。

### 5.5 合并连续空行

Pandoc 产生的多余空行（Word 分页符痕迹）压缩为至多一个。

### 5.6 前后对比（TS 38.211 目录页）

**处理前**：
```
## Contents {#contents}
[Foreword](#foreword)
[1](#scope) Scope
![](media/image1.emf){width="3.0in"}
```

**处理后**：
```
## Contents
Foreword
1 Scope
```

> 💡 目录文字（TOC 条目）被保留而非删除——目录中的章节标题（如"NR operating bands and channel arrangement"）是高质量领域术语，对 BM25 关键词检索有帮助。

---

## 6. 分块策略

### 6.1 为什么要分块

TS 38.211 全文约 15 万字符，面临两个约束：
1. **模型限制**：BGE-M3 一次最多处理 ~8000 字符
2. **检索精度**：chunk 太大 → 召回不精确；chunk 太小 → 上下文不足

分块要解决"在哪里切"的问题。

### 6.2 核心思路：按自然骨节切分

采用**标题感知分块**——按 3GPP 规范已有的章节边界切分，而非固定字数。三步：建树 → 找叶子 → 切分。

### 6.3 章节树构建

#### 识别标题

正则 `^(#{1,8})\s+(.+)$` 识别 Markdown 标题，`^(\d+(?:\.\d+)*)\s+(.{3,})$` 提取章节编号。

#### 层级修正

Pandoc 受限于 Word 样式数（最多 6 级），深层嵌套可能被标为同级。系统用章节编号中的"."数量矫正：`"6.2.1.2"` 有 3 个点 → level 4，`max(md_level, num_level)` 取更深值为真实层级。

#### 栈式构建

类似 HTML 解析：遇更深层级入栈，遇同级/更浅层级弹出。最终每个节点知道自己的 start/end 位置。

### 6.4 树形分块

收集所有叶子节点（没有子章节的末端章节），每个叶子节点成一个 chunk：

```
6 NR radio interface
├── 6.1 General → 有子节点
│   ├── 6.1.1 Overview        ← 叶子 → chunk #1
│   └── 6.1.2 Multiple access  ← 叶子 → chunk #2
├── 6.2 Physical resources
│   ├── 6.2.1 OFDM...          ← 叶子 → chunk #3
│   └── 6.2.3 Resource grid    ← 叶子 → chunk #5
```

每个 chunk 继承完整层级上下文：`section_number`、`section_path`、`parent_section_id` 等。

### 6.5 三种原子保护结构

以下结构**永不切割**：

| 结构 | 正则特征 | 示例 |
|------|----------|------|
| Grid Table | `^\+[-=+]+\+$` | Pandoc 输出的 `+---+---+` 表格 |
| Math Block | `^\$\$$` | `$$...$$` 块级公式 |
| Pipe Table | `^\|.+\|$` | 标准 `|列1|列2|` 表格 |

分块器检测这些结构的起止边界，跨越边界的切割退回到上一个安全分割点。

### 6.6 chunk_overlap：跨边界衔接

按章节边界切分有一个隐患：某段技术说明恰好跨越两个叶子节点边界，前半段是"PUSCH 的 DMRS 配置"概述，后半段是具体的参数表。靠后的 chunk 缺少上下文，检索时可能被漏掉。

`chunk_overlap=100` 的意思是：每个 chunk 的末尾额外附带前一 chunk 的最后 100 个字符。这样相邻 chunk 之间有一小段重叠，检索时无论命中哪边都能看到衔接处的内容。

> ⚠️ 本项目 overlap 仅用于文本类 chunk，含完整表格的 chunk 不加 overlap——表格被截断反而造成信息破损。

### 6.7 Dynamic 模式

不同内容类型用不同大小：

| 类型 | 上限 | 原因 |
|------|------|------|
| 参数表 | 5000 字符 | 表格信息密度低，完整性更重要 |
| 纯文本 | 1500 字符 | 精确定义不宜淹没在大段文字中 |

绝对硬上限 8000 字符，受 BGE-M3 8192 token 限制。

### 6.8 字节安全层

Milvus VARCHAR 最多 65535 字节。系统设 55000 字节安全边距，超限时二次拆分：

1. Grid Table → 按行组拆（保留表头）
2. HTML Table → 按 `<tr>` 行拆
3. 纯文本 → 按段落边界拆

极端情况（<0.1%）仍超限时，`_safe_truncate_bytes()` 语义截断：段落末 → 行末 → 句末 → 词边界 → 硬切，加 `…`。

### 6.9 举例：TS 38.211 §6.2 分块

输入 6.2 节及其子章节的 Markdown → 构建章节树 → 产出 3 个 chunk：

| Chunk | 内容 | 大小 |
|-------|------|------|
| #8 | §6.2.1 OFDM baseband signal generation | ~2400 字符 |
| #9 | §6.2.2 Numerologies | ~800 字符 |
| #10 | §6.2.3 Resource grid（含 Grid Table） | ~3200 字符 |

父章节 6.2 的开头段落合并进第一个子章节 chunk #8。

---

## 7. 分类标注

### 7.1 三个维度

#### content_type —— 什么形态

| 值 | 触发条件 |
|----|----------|
| `parameter_table` | 含 Grid/Pipe Table |
| `definition` | 父标题含 definition/term/abbreviation |
| `procedure` | 父标题含 procedure/process/step/flow |
| `overview` | 以上都不匹配 |

#### spec_role —— 规范权威度

| 值 | 判定 |
|----|------|
| `authoritative` | 物理层/MAC/RRC 核心规范：38.211/212/213/214/321/331 |
| `overview` | 38.300（NR 总体描述） |
| `supporting` | 其他辅助规范 |

#### topic_domain —— 技术领域

| 值 | 规则 | 示例 |
|----|------|------|
| `phy_layer` | 38.2xx | 38.211, 38.212, 38.213 |
| `mac_layer` | 38.32x | 38.321, 38.322, 38.323 |
| `rrc_layer` | 38.33x | 38.331 |
| `ran_arch` | 38.4xx | 38.401, 38.413 |

### 7.2 举例

TS 38.211 §6.2.3 Resource grid 参数表 chunk：
- `content_type = "parameter_table"`（含 Grid Table）
- `spec_role = "authoritative"`（38.211 是物理层核心规范）
- `topic_domain = "phy_layer"`（38.2xx → phy_layer）

分类完全基于规则，零计算开销。标签随 chunk 存入 Milvus，检索阶段用于加权排序。

---

## 8. 嵌入生成

### 8.1 什么是嵌入（生成的原理）

**嵌入**（embedding）是把一段文字转成 1024 个浮点数。关键问题是：这些数字从哪来？

BGE-M3 是一个**预训练的神经网络模型**（约 5.68 亿参数），它被海量中英韩文本训练过，学会了"理解"一段文字的语义并将其压缩为一个固定长度的数字指纹。流程如下：

```
输入文本 → [分词器 Tokenizer] 切成 token 序列 → [Transformer 编码器] 逐层提取语义 → [池化层] 压缩为 1024 维向量
```

1. **分词**："OFDM baseband signal" → `[OF, DM, base, band, signal]` 等 token ID
2. **编码**：24 层 Transformer 逐层处理 token 序列，每层让 token 之间互相"注意"（attention 机制），逐步理解上下文关系
3. **池化**：取最后一层所有 token 向量的平均值（mean pooling），得到 1024 个 float32

关键性质：**语义相近的文字，向量在 1024 维空间中的位置也近。** 用余弦相似度衡量（值域 [-1,1]，1 表示完全相同，-1 表示完全相反）。检索时用户问题也经同一模型转成向量，在 Milvus 中找余弦距离最近的 chunk。

> 💡 可以参考类比：给世界上每本书分配一个 GPS 坐标，内容相似的书坐标接近。嵌入就是"语义 GPS"——只不过不是 2 维经纬度，而是 1024 维。

### 8.2 三层降级策略

```
云端 API ──失败──▶ 本地 BGE-M3 ──失败──▶ 零向量兜底
```

### 8.3 双层缓存

| 缓存层 | 存储 | 特点 |
|--------|------|------|
| SQLite | `data/cache/embedding_cache.db` | 批量查询，主缓存 |
| .npy 文件 | `data/cache/embeddings/{hash}.npy` | 向后兼容的备份 |

Key = `SHA256(text)[:16]`。查询顺序：SQLite → .npy → 重新计算。文件命中后回填 SQLite。

### 8.4 MPS 子进程（Apple Silicon 专用）

主进程已连接 Milvus（gRPC 全局状态），BGE-M3 在 MPS 上运行会与 gRPC 多进程冲突导致死锁。解决方案：启动独立子进程做嵌入，通过 pickle 文件传递数据，子进程不 import pymilvus。`workers=1`（MPS 不支持多进程 GPU），M4 Max 上 ~75 tokens/s。

### 8.5 分段+断点续传

每 5000 条一段，逐段嵌入→逐段入库。某段失败不影响已入库段。支持从 checkpoint 恢复。

### 8.6 嵌入文本构造

```
text_for_embedding = f"{c.section_title} {c.section_path} {c.text[:500]}"
```

`section_title` + `section_path` 提供结构上下文，`text[:500]` 取前 500 字符（核心语义通常在前部）。

---

## 9. 入库 Milvus

### 9.1 Milvus 简介

类比：MySQL 存整数/字符串/日期，Milvus 额外存 1024 维浮点数组并提供"找最近 N 个邻居"能力。本系统使用 Milvus 2.4，部署在 `localhost:19530`（Docker Compose，通过 gRPC 协议通信）。

> **gRPC** 是 Google 开发的高性能远程过程调用协议，简单理解为"程序通过网络调用另一台机器上的函数"即可。Milvus 的增删改查都走 gRPC。

### 9.2 Schema

Collection 名：`TeleComm_specs`。包含 id（INT64 主键自增）、text（VARCHAR 65535）、dense_vector（FLOAT_VECTOR 1024）、及全部 14 个标量元数据字段。

### 9.3 向量索引：IVF_FLAT 的原理

几十万个 1024 维向量全量对比太慢。IVF_FLAT 用"先粗筛再细查"策略：先训练阶段用 K-Means 把全部向量聚成 1024 个簇，每个簇有一个中心点；检索时查询向量先和 1024 个中心点比，找出最近 32 个簇，只在其中逐条精确对比。类比：图书馆 10 万本书，先分 1024 个书架（按主题聚类），找书时先确定最相关的 32 个书架，只在这 32 个书架上逐本翻看。

| 参数 | 值 | 含义 |
|------|-----|------|
| `index_type` | `IVF_FLAT` | 聚成 1024 簇，检索时搜最近几个簇 |
| `metric_type` | `COSINE` | 余弦相似度衡量距离（见下） |
| `nlist` | `1024` | 聚类中心数，太少则每簇太大，太多则粗筛也慢 |
| `nprobe` | `32`（检索时） | 搜最近 32 个簇，越多越准但越慢 |

#### 为什么用 COSINE 而不是 L2

- **L2 距离（欧几里得距离）**：衡量"向量的绝对位置差"。`[1,2,3]` 和 `[2,3,4]` 的 L2 距离约 1.73，适合数值型数据
- **COSINE 相似度**：衡量"向量的方向一致性"。值域 [-1,1]，1 表示方向完全相同。适合文本语义——一段长文本和一段短文本如果语义相同，方向接近但绝对位置可能差很多

业界共识：文本嵌入用 COSINE，图像嵌入用 L2。本项目遵循此惯例。

### 9.4 插入流程

列式插入，每批最多 **1000 条**（防止 gRPC 消息体超限）。每批插入后执行 **flush**（刷盘），强制 Milvus 将内存中的数据写入磁盘持久化，防止进程崩溃时数据丢失。

### 9.5 BM25 及混合检索融合（RRF）

#### BM25：关键词兜底

向量语义检索擅长找"意思相近"的内容，但有时用户输入的是精确术语（如"N_RB"、"SCS 30kHz"），关键词匹配更直接。**BM25** 是经典的关键词检索算法，基于词频和逆文档频率给匹配 chunk 打分。本项目在每次全量摄入后从 Milvus 全量读取文字重建 BM25 索引，持久化到 `data/vectors/bm25_index.pkl`。

#### RRF：双路结果融合

检索时同时查询两路：
- **Dense（稠密向量）**：语义相似度排序，擅长找"PUSCH 的资源分配方式"
- **Sparse（BM25 关键词）**：关键词匹配排序，擅长找"N_RB = 273"

两路各自返回 Top-K，用 **RRF（Reciprocal Rank Fusion，倒数排名融合）**合并。公式为 `RRF_score = 1/(k + rank_dense) + 1/(k + rank_sparse)`，k=60 是平滑常数。一个 chunk 在语义路排第 2、关键词路排第 5，总得分约 0.0315。最终按 RRF 得分重排取 Top-K，既不会漏掉语义相近的说明段落，也不会漏掉精确术语匹配的参数表。

### 9.6 完整记录示例

```json
{
  "id": 4512300042,
  "text": "+------------------+---+...",
  "dense_vector": [0.0321, -0.1456, ..., -0.0245],
  "doc_id": "38211-60",
  "series": 38,
  "spec_number": "38.211",
  "release": "R18",
  "section_number": "6.2.3",
  "section_title": "Resource grid",
  "section_path": "6 > 6.2 Physical resources > 6.2.3 Resource grid",
  "parent_section_id": "6.2",
  "parent_title": "Physical resources",
  "chunk_index": 12,
  "doc_type": "3gpp",
  "content_type": "parameter_table",
  "spec_role": "authoritative",
  "topic_domain": "phy_layer"
}
```

---

## 10. Markdown 的存在形式

### 10.1 核心问题

> **Markdown 仅存在于 Python 内存中，从始至终不会保存为磁盘上的 `.md` 文件。**

整个流程是一次性内存管线：

```
磁盘 DOCX ──pandoc stdout──▶ Python str ──▶ Splitter ──▶
  Chunk 对象列表 ──▶ Embedder ──▶ Milvus
```

### 10.2 为什么不存盘

- **DOCX 是唯一事实源**——存中间产物反而引入一致性问题
- **可随时重新生成**——Pandoc 转换是确定性的
- **避免磁盘膨胀**——200+ 篇规范的 .md 合计约 100MB，用不到则不存

### 10.3 数据流转全图

```
磁盘 (Persistent)：
  data/documents/R18/38_series/38211-60.docx     [事实源]
  data/checkpoint/chunks_checkpoint.pkl           [断点续传]
  data/cache/embedding_cache.db                   [嵌入缓存]
  data/manifest/ingestion_state.json              [摄入台账]
  data/vectors/bm25_index.pkl                     [BM25索引]
  Milvus (Docker)                                 [最终存储]

内存 (Transient)：
  pandoc stdout → Python str → _postprocess() → Splitter →
  list[Chunk] → classify_chunk() → Embedder → Chunk.embedding →
  MilvusStore.insert()
  所有中间产物在函数返回后被 GC 回收
```

---

## 11. 增量管理

### 11.1 IngestionManifest 台账

JSON 格式，记录每份规范的摄入状态。Key = `"{spec_number}|{release}"`：

```json
{
  "38.211|R18": {
    "spec_number": "38.211", "release": "R18",
    "latest_version": "60", "sha256": "a1b2c3...",
    "chunk_count": 287, "ingested_at": "2026-07-20T14:30:00Z"
  }
}
```

### 11.2 增量流程

```
扫描全部 .docx →
  ├─ 台账中不存在 → 【新增】→ 全流程
  ├─ 版本号相同 + SHA256 相同 → 【跳过】
  ├─ 新版本 > 旧版本 → 【替换】→ 先删旧 chunks，再入新的
  ├─ 版本相同但 SHA256 不同 → 【告警跳过】
  └─ 文件已删但台账有记录 → 【孤儿】→ 从 Milvus 删对应 chunks
```

### 11.3 Checkpoint

提取阶段最耗时（200+ 篇规范需数十分钟）。提取完成后所有 Chunk 对象序列化到 `chunks_checkpoint.pkl`。嵌入阶段若失败，`--resume-from-checkpoint` 跳过提取，直接嵌入+入库。

### 11.4 全量重建

`python scripts/bulk_ingest.py --full-rebuild`：drop collection + 清空台账 → 重跑全部。仅在分块策略变更、嵌入模型切换时使用。

---

## 12. 完整串联：TS 38.211

以 `38211-60.docx`（TS 38.211 V18.4.0，NR 物理信道与调制）为例：

### 文档信息

| 属性 | 值 |
|------|-----|
| 文件 | `38211-60.docx` (~5MB) |
| 规范 | TS 38.211 NR Physical channels and modulation |
| Release | R18（V18.4.0, 2024-09） |
| 页数 | ~180 页 |

### 7 步全流程

**Step 1 — Pandoc 转换**（~35s）：
- DOCX → ~150,000 字符 Markdown
- 含 Grid Table、LaTeX math、图片引用、TOC 链接

**Step 2 — 元数据提取**（<1s）：
- 文件名 → `doc_id="38211-60"`, `spec_number="38.211"`, `series=38`
- 文本头 → `release="R18"`

**Step 3 — 后处理**（<1s）：
- 清理 ~200 处标题属性、~150 处目录链接
- 删除封面多图、裁剪 Change History（~8000 字符）
- 处理后约 135,000 字符

**Step 4 — 分块**（~2s）：
- 构建 8 层深度的章节树
- 产出 ~280 个 chunk，平均 ~480 字符
- 最大 chunk（ASN.1 定义表）~7000 字符

**Step 5 — 分类**（<1s）：
- 参数表 chunk → `parameter_table + authoritative + phy_layer`
- 定义 chunk → `definition + authoritative + phy_layer`
- 纯规则匹配，零延迟

**Step 6 — 嵌入**：
- 嵌入文本 = `section_title + section_path + text[:500]`
- 每个 chunk 生成 1024 维 float32 向量
- SQLite + .npy 双层缓存写入

**Step 7 — 入库**：
- 280 个 chunk 随第一批 1000 条微批次一起入库
- 16 列数据写入 Milvus `TeleComm_specs` Collection

### 端到端耗时

| 步骤 | 云端 API | 本地 MPS (Apple Silicon) |
|------|----------|--------------------------|
| Pandoc 转换 | ~35s | ~35s |
| 后处理 | <1s | <1s |
| 分块 | ~2s | ~2s |
| 分类 | <1s | <1s |
| 嵌入 | ~12s | ~45s |
| 入库 | ~3s | ~3s |
| **合计** | **~53s** | **~86s** |

> 💡 云端 API 更快但依赖网络和第三方服务。本地 MPS 离线可用且数据不出本机，适合涉密场景。M4 Max 上 MPS 约 75 tokens/s，280 个 chunk 嵌入约需 45s。

---

## 附录

### A. 关键配置项

`IngestionConfig`（`.env` 中 `INGESTION__` 前缀覆盖）：

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `chunk_mode` | `dynamic` | 分块模式 |
| `table_max_chars` | `5000` | 表格上限 |
| `prose_max_chars` | `1500` | 文本上限 |
| `max_chunk_chars` | `8000` | BGE-M3 8192 token 安全边距 |
| `chunk_overlap` | `100` | chunk 重叠字符数（详见第 6.6 节） |

`Settings`（`.env` 配置）：

| 配置项 | 默认值 |
|--------|--------|
| `milvus_host:port` | `localhost:19530` |
| `milvus_collection_name` | `TeleComm_specs` |
| `documents_dir` | `data/documents` |
| `embedding_dimension` | `1024` |

### B. 数据目录总览

```
data/
├── documents/          # 原始 DOCX（事实源）
├── cache/
│   └── embedding_cache.db  # SQLite 嵌入缓存
├── checkpoint/
│   └── chunks_checkpoint.pkl  # 断点续传
├── manifest/
│   └── ingestion_state.json   # 摄入台账
├── vectors/
│   └── bm25_index.pkl   # BM25 索引
└── feedback.db          # 用户反馈
```

### C. 常用命令

```bash
# 全量重建
python scripts/bulk_ingest.py --full-rebuild

# 增量更新
python scripts/bulk_ingest.py

# 只处理 38 系列
python scripts/bulk_ingest.py --series 38

# 从 checkpoint 断点续传
python scripts/bulk_ingest.py --resume-from-checkpoint

# 查看 Milvus 统计
python -m src.cli stats
```
