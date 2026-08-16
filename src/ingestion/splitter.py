"""标题感知文档分块器 — Markdown / 纯文本通用，结构感知（表格/公式原子化保护）.

支持两种策略:
  1. header_split: 解析 #~#### 标题, 按最深层级切分 (适合规范文档)
  2. char_split: 固定字符窗口 + 重叠 (降级策略, 兼容 Phase 1)

pandoc 结构感知:
  - Grid Table (+---+---+) → 原子保护，永不切割
  - Math Block ($$...$$) → 原子保护
  - Pipe Table (|...|) → 原子保护
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from src.retriever.vector_store import Chunk

logger = logging.getLogger(__name__)

# Markdown 标题正则 — 匹配 1-8 级 (pandoc Annex 用 ######## 级别)
HEADER_RE = re.compile(r"^(#{1,8})\s+(.+)$", re.MULTILINE)
# 规范章节编号 (如 "6.1.2  PDU Session Establishment"; 支持字母后缀 "5.3.5.13b")
SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*[a-z]?)\s+(.{3,})$", re.MULTILINE)
# 纯文本 TOC 章节行 (如 "1 Scope   7") — mammoth 输出降级
PLAIN_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z]\w.{2,}?)\s+\d+\s*$", re.MULTILINE)

# ── 原子块检测 (不可分割的结构) ──
# Grid Table: pandoc +---+---+ 格式
GRID_TABLE_BOUNDARY = re.compile(r'^\+[-=+]+\+$')
# Math display block: $$ ... $$
MATH_BLOCK_DELIM = re.compile(r'^\$\$$')
# Pipe Table: | a | b |
# 分隔行需匹配任意列数 (3GPP 规范多为 3+ 列表格, 原 ^\|[-: ]+\|$ 只匹配 2 段,
# 导致多列表格被当 prose 切碎 — SSB 表 7.4.3.1-1 等核心内容丢失)
PIPE_TABLE_LINE = re.compile(r'^\|.+\|$')
PIPE_TABLE_SEP = re.compile(r'^\|[\s\-:|]+\|$')

# ── Chunk 元数据规则分类 ──

AUTHORITATIVE_SPECS = {
    "38.211", "38.212", "38.213", "38.214",  # NR PHY
    "38.321", "38.322", "38.323",               # NR MAC
    "38.331", "38.304", "38.305",               # NR RRC
    "38.413", "38.423",                         # NGAP/XnAP
    "36.211", "36.212", "36.213", "36.214",  # LTE PHY
    "36.321", "36.322", "36.323",               # LTE MAC
    "36.331", "36.304", "36.305",               # LTE RRC
}

# Non-38 系列 → topic_domain 映射
_SPEC_SERIES_DOMAIN: dict[str, str] = {
    "23": "core_network",    # 5GS Architecture / Procedures
    "24": "ue_procedures",   # UE-Network signalling
    "36": "lte_ran",         # LTE RAN (E-UTRA)
    "37": "multi_rat",       # Multi-RAT
    "25": "utran",           # UTRAN
    "21": "requirements",    # Service/System requirements
    "22": "requirements",    # Service requirements
    "33": "security",        # Security
    "26": "codec",           # Codec
    "28": "oam",             # OAM/Charging
    "29": "core_network",    # Core network protocols
    "32": "oam",             # OAM
}
TABLE_KEYWORDS = ["table", "Table"]
DEFINITION_KEYWORDS = ["definition", "general", "principles", "overview", "architecture"]
PROCEDURE_KEYWORDS = ["procedure", "procedures", "call flow", "message sequence"]

# 3GPP 表格编号引用: "Table 6.3.3.2-1" / "Table 4.1-1" (支持 - – — 分隔符)
TABLE_REFERENCE_RE = re.compile(r"Table\s+\d+(?:\.\d+)+\s*[-–—]\s*\d+", re.IGNORECASE)


def classify_chunk(
    text: str,
    spec_number: str,
    parent_title: str,
) -> dict[str, str]:
    """基于规则的 chunk 元数据分类.

    Returns:
        {"content_type": ..., "spec_role": ..., "topic_domain": ...}
    """
    # 1. content_type — 表格判定只认真实表结构 (_contains_table) 或父章节标题含 table.
    #    移除 _contains_table_reference: "see Table 6.3.3.2-1" 只是正文引用, 并非
    #    表格本身, 曾导致 48% chunk 被误判为 parameter_table (元数据加权形同虚设).
    if _contains_table(text) or any(
        kw in parent_title for kw in TABLE_KEYWORDS
    ):
        content_type = "parameter_table"
    elif any(kw in parent_title.lower() for kw in DEFINITION_KEYWORDS):
        content_type = "definition"
    elif any(kw in parent_title.lower() for kw in PROCEDURE_KEYWORDS):
        content_type = "procedure"
    else:
        content_type = "overview"

    # 2. spec_role
    if spec_number in AUTHORITATIVE_SPECS:
        spec_role = "authoritative"
    elif spec_number == "38.300":
        spec_role = "overview"
    else:
        spec_role = "supporting"

    # 3. topic_domain — 优先查非38系映射表，38系按子系列推断
    parts = spec_number.split(".")
    major = parts[0] if parts else ""
    if major == "38":
        sub = parts[1][:1] if len(parts) > 1 and parts[1] else ""
        if sub == "2":
            topic_domain = "phy_layer"
        elif sub == "3":
            topic_domain = "mac_layer" if "321" in spec_number or "322" in spec_number or "323" in spec_number else "rrc_layer"
        elif sub == "4":
            topic_domain = "ran_arch"
        else:
            topic_domain = ""
    else:
        topic_domain = _SPEC_SERIES_DOMAIN.get(major, "")

    return {
        "content_type": content_type,
        "spec_role": spec_role,
        "topic_domain": topic_domain,
    }


def _contains_table(text: str) -> bool:
    """检测文本是否包含 Markdown 表格 (pipe 或 grid)."""
    lines = text.split("\n")
    pipe_sep_count = sum(1 for line in lines if PIPE_TABLE_SEP.match(line.strip()))
    plus_count = sum(1 for line in lines if line.strip().startswith("+"))
    return pipe_sep_count >= 1 or plus_count >= 2


def _contains_table_reference(text: str) -> bool:
    """检测文本是否引用 3GPP 表格编号 (如 "Table 6.3.3.2-1").

    参数表 section 常以表号标题开头 (Table X.Y.Z-W: caption) 或正文引用
    (see Table X.Y.Z-W); 这类 chunk 即使表格结构被截断也应归为参数表.
    """
    return TABLE_REFERENCE_RE.search(text) is not None


# ── 章节树节点 ──

@dataclass
class SectionNode:
    """章节树节点."""
    level: int                # 标题层级 1-4
    title: str                # 标题全文 (含编号)
    sec_id: str               # 纯编号 (如 "6.1.2"), 无编号则为 ""
    start: int                # 在原文中的起始位置
    end: int = 0              # 在原文中的结束位置 (子节点末或文件末)
    parent: SectionNode | None = None
    children: list[SectionNode] = field(default_factory=list)


class HeaderAwareSplitter:
    """标题感知分块器 — 按 Markdown 标题层级智能切分，表格/公式原子化.

    原则:
    - 优先在最深层级拆分 (叶子节点)
    - Grid Table / Math Block / Pipe Table 永不切割
    - dynamic 模式：表格/正文分离为独立 chunk，各自有独立的 size 上限
    - 超长节点在段落边界二次切分（原子块内部分隔符被保护）
    - 每个 chunk 自动附加 parent_section_id/parent_title
    """

    def __init__(
        self,
        max_chunk_chars: int = 2500,
        chunk_overlap_chars: int = 100,
        max_chunk_bytes: int = 55000,
        chunk_mode: Literal["fixed", "dynamic"] = "dynamic",
        table_max_chars: int = 5000,
        prose_max_chars: int = 1500,
        max_chunk_hard_chars: int = 8000,  # BGE-M3 8192 token 安全上限
        min_chunk_chars: int = 300,  # 小于此值的正文碎片并入相邻 chunk
    ):
        self.max_chunk = max_chunk_chars
        self.overlap = chunk_overlap_chars
        # Milvus VARCHAR 65535 bytes 硬限制, 留 ~10KB 安全边距
        self.max_chunk_bytes = max_chunk_bytes
        self.chunk_mode = chunk_mode
        self.table_max_chars = table_max_chars
        self.prose_max_chars = prose_max_chars
        # BGE-M3 8192 tokens → 最坏情况 (密集 ASN.1/数字表) ~1 char/token
        # 8000 chars 确保即使最极端密度也不超 token 限制
        self.max_chunk_hard = max_chunk_hard_chars
        self.min_chunk_chars = min_chunk_chars

    def split_document(
        self,
        text: str,
        doc_meta: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        """对单篇文档分块, 返回带完整元数据的 Chunk 列表."""
        if doc_meta is None:
            doc_meta = {}

        doc_id = doc_meta.get("doc_id", "")
        series = doc_meta.get("series", 0)
        spec_number = doc_meta.get("spec_number", "")
        release = doc_meta.get("release", "")
        doc_type = doc_meta.get("doc_type", "3gpp")

        # 构建章节树
        root = self._build_section_tree(text)
        if root.children:
            chunks = self._split_by_tree(text, root, doc_id, series, spec_number, release, doc_type)
        else:
            # 无标题 → 降级为字符分块
            chunks = self._split_by_chars(text, doc_id, series, spec_number, release, doc_type)

        # 小碎片合并 pass — 向前并入优先, 表格/公式原子块保护; 合并后重编号
        merged = self._merge_small_chunks(chunks)
        for i, c in enumerate(merged):
            c.chunk_index = i
        self._finalize_parent_ids(merged)
        return merged

    @staticmethod
    def _finalize_parent_ids(chunks: list[Chunk]) -> None:
        """合并重编号后修正 parent_chunk_id — 同 section 连续 chunk 共享首个子 chunk 索引.

        parent_text 非空表示该 chunk 属于被切分的 section; 连续且 parent_text
        相同的 chunk 属于同一 section, 共享该 section 首个子 chunk 的索引.
        """
        run_start: int | None = None
        for i, c in enumerate(chunks):
            if not c.parent_text:
                c.parent_chunk_id = 0
                run_start = None
                continue
            if run_start is None or c.parent_text != chunks[i - 1].parent_text:
                run_start = i
            c.parent_chunk_id = run_start

    # ── 章节树构建 ──

    def _build_section_tree(self, text: str) -> SectionNode:
        """解析标题构建嵌套章节树 — Markdown # 标题 + 纯文本 TOC 降级."""
        root = SectionNode(level=0, title="", sec_id="", start=0, end=len(text))

        # 收集所有标题位置
        headers: list[tuple[int, int, str, str]] = []  # (start, level, sec_id, title)
        for m in HEADER_RE.finditer(text):
            md_level = len(m.group(1))
            # pandoc Annex 级标题 (########=8) → top-level
            if md_level >= 8:
                md_level = 1   # Annex: top-level
            title = m.group(2).strip()
            sec_id = ""
            num_match = SECTION_NUM_RE.match(title)
            if num_match:
                sec_id = num_match.group(1)
                title = num_match.group(2).strip()
                # 以 section number 真实层级修正 Markdown 层级
                # pandoc 受 DOCX heading 样式数限制 (最多6级), 深层嵌套可能输出相同 # 级别
                num_level = sec_id.count(".") + 2  # "6.2.3.7.4.1" → 5 dots → level 7
                md_level = max(md_level, num_level)
            headers.append((m.start(), md_level, sec_id, title))

        # 降级: 无 Markdown 标题 → 尝试 mammoth 纯文本 TOC 章节行
        if not headers:
            for m in PLAIN_SECTION_RE.finditer(text):
                sec_id = m.group(1)
                title = m.group(2).strip()
                level = sec_id.count(".") + 1
                headers.append((m.start(), min(level, 4), sec_id, title))

        if not headers:
            return root

        # 栈式构建嵌套结构
        stack: list[SectionNode] = [root]
        for i, (pos, level, sec_id, title) in enumerate(headers):
            node = SectionNode(level=level, title=title, sec_id=sec_id, start=pos)
            next_pos = headers[i + 1][0] if i + 1 < len(headers) else len(text)
            node.end = next_pos

            while len(stack) > 1 and stack[-1].level >= level:
                stack.pop()

            node.parent = stack[-1]
            stack[-1].children.append(node)
            stack.append(node)

        if headers:
            root.end = headers[-1][0]

        return root

    # ── 树形分块 ──

    def _split_by_tree(
        self,
        text: str,
        root: SectionNode,
        doc_id: str,
        series: int,
        spec_number: str,
        release: str,
        doc_type: str = "3gpp",
    ) -> list[Chunk]:
        """从章节树叶子节点切分 chunk.

        父章节开头正文 (第一个子标题之前的内容) 单独成 chunk — 否则如
        TS 38.211 §7.4.3.1 的 "4 OFDM symbols" 正文与表 7.4.3.1-1 会随
        子章节切分而丢失 (叶子收集只覆盖子节点自身区间).
        """
        chunks: list[Chunk] = []
        intro_nodes = self._collect_parent_intros(root)
        leaf_nodes = self._collect_leaves(root)

        chunk_idx = 0
        for node in intro_nodes + leaf_nodes:
            content = text[node.start:node.end].strip()
            if not content:
                continue

            parent_id, parent_title = self._get_parent_context(node)
            section_number = node.sec_id
            section_title = node.title
            section_path = self._build_section_path(node)

            if len(content) <= self.max_chunk:
                chunks.append(Chunk(
                    text=content,
                    embedding=None,
                    doc_id=doc_id,
                    series=series,
                    spec_number=spec_number,
                    release=release,
                    parent_section_id=parent_id,
                    parent_title=parent_title,
                    chunk_index=chunk_idx,
                    section_number=section_number,
                    section_title=section_title,
                    section_path=section_path,
                    doc_type=doc_type,
                ))
                chunk_idx += 1
            else:
                sub_chunks = self._split_long_section(
                    content, doc_id, series, spec_number,
                    release, parent_id, parent_title,
                    section_number, section_title, section_path,
                    chunk_idx, doc_type,
                )
                for sc in sub_chunks:
                    sc.parent_text = content[:4096]
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)

        return chunks

    def _collect_parent_intros(self, node: SectionNode) -> list[SectionNode]:
        """收集所有有子节点且开头有正文的父章节 — 开头正文区间 [start, 首子节点 start).

        避免父章节介绍段 (如 SSB 时频结构定义) 在叶子切分时丢失.
        """
        intros: list[SectionNode] = []
        if node.children:
            first_child_start = min(c.start for c in node.children)
            if first_child_start > node.start and node.level > 0:
                intro = SectionNode(
                    level=node.level, title=node.title, sec_id=node.sec_id,
                    start=node.start, end=first_child_start, parent=node.parent,
                )
                intros.append(intro)
            for child in node.children:
                intros.extend(self._collect_parent_intros(child))
        return intros

    def _collect_leaves(self, node: SectionNode) -> list[SectionNode]:
        """收集所有叶子节点."""
        if not node.children:
            return [node] if node.level > 0 else []
        leaves: list[SectionNode] = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves

    def _get_parent_context(self, node: SectionNode) -> tuple[str, str]:
        """获取节点的父章节编号和标题.

        向上查找最近的有编号 (sec_id 非空) 祖先 — RRC 规范的 ASN.1 定义标题无编号
        (如 "#### NR-RRC-Definitions"), 直接父节点 sec_id 为空, 需继承最近的编号祖先
        (如 §6.2), 否则 parent_section_id 为空导致检索后处理/章节级召回失效。
        """
        parent = node.parent
        while parent and parent.level > 0:
            if parent.sec_id:
                return parent.sec_id, parent.title
            parent = parent.parent
        return "", ""

    @staticmethod
    def _build_section_path(node: SectionNode) -> str:
        """构建从根到当前节点的完整层级路径.

        如 "7 Uplink Power control > 7.1 PUSCH > 7.1.1 UE behaviour"
        """
        parts: list[str] = []
        current: SectionNode | None = node
        while current and current.level > 0:
            label = f"{current.sec_id} {current.title}".strip() if current.sec_id else current.title
            parts.append(label)
            current = current.parent
        parts.reverse()
        return " > ".join(parts)

    # ── 长章节二次切分（结构感知） ──

    def _segment_by_atomic_blocks(self, text: str) -> list[tuple[str, str]]:
        """按原子块边界将文本拆分为 (type, text) 列表.

        type 值: "prose" | "grid_table" | "pipe_table" | "math_block"
        复用 _protect_atomic_blocks 的检测逻辑, 但改为分割而非替换.
        """
        lines = text.split('\n')
        segments: list[tuple[str, str]] = []
        i = 0
        n = len(lines)
        prose_buf: list[str] = []

        while i < n:
            stripped = lines[i].strip()

            # Grid Table
            if GRID_TABLE_BOUNDARY.match(stripped):
                if prose_buf:
                    segments.append(("prose", '\n'.join(prose_buf)))
                    prose_buf = []
                start_i = i
                i += 1
                while i < n:
                    s = lines[i].strip()
                    if GRID_TABLE_BOUNDARY.match(s):
                        next_is_data = (i + 1 < n and
                                       lines[i + 1].strip().startswith('|'))
                        if not next_is_data:
                            i += 1
                            break
                    i += 1
                segments.append(("grid_table", '\n'.join(lines[start_i:i])))
                continue

            # Math Block
            if MATH_BLOCK_DELIM.match(stripped):
                if prose_buf:
                    segments.append(("prose", '\n'.join(prose_buf)))
                    prose_buf = []
                start_i = i
                i += 1
                while i < n:
                    if MATH_BLOCK_DELIM.match(lines[i].strip()):
                        i += 1
                        break
                    i += 1
                segments.append(("math_block", '\n'.join(lines[start_i:i])))
                continue

            # Pipe Table
            if PIPE_TABLE_LINE.match(stripped):
                if i + 1 < n and PIPE_TABLE_SEP.match(lines[i + 1].strip()):
                    if prose_buf:
                        segments.append(("prose", '\n'.join(prose_buf)))
                        prose_buf = []
                    start_i = i
                    i += 2
                    while i < n:
                        if not PIPE_TABLE_LINE.match(lines[i].strip()):
                            break
                        i += 1
                    segments.append(("pipe_table", '\n'.join(lines[start_i:i])))
                    continue

            prose_buf.append(lines[i])
            i += 1

        if prose_buf:
            segments.append(("prose", '\n'.join(prose_buf)))

        return segments

    def _split_long_section(
        self,
        text: str,
        doc_id: str,
        series: int,
        spec_number: str,
        release: str,
        parent_id: str,
        parent_title: str,
        section_number: str,
        section_title: str,
        section_path: str,
        start_idx: int,
        doc_type: str = "3gpp",
    ) -> list[Chunk]:
        """对超长文本在段落边界二次切分 — 表格/公式原子化保护 + 字节上限自适应.

        策略:
        1. 找到所有 Grid Table / Math Block / Pipe Table / HTML Table
        2. 用占位符替换 → 安全切分 (\\n\\n 不会误伤表格)
        3. 累计段落至 max_chunk_chars → 还原占位符
        4. 超长原子块保持完整, 但最终检查字节上限:
           超过 max_chunk_bytes → 按内容类型自适应拆分 (行级)
        """
        # dynamic 模式：按原子块边界分离表格/正文
        if self.chunk_mode == "dynamic":
            return self._split_long_section_dynamic(
                text, doc_id, series, spec_number, release,
                parent_id, parent_title, section_number, section_title,
                section_path, start_idx, doc_type,
            )

        # ── fixed 模式：原有段落累计逻辑 ──
        # Step 1: 保护原子块
        protected_text, placeholder_map = self._protect_atomic_blocks(text)

        # Step 2: 安全切分（原子块被占位符隐藏，内部 \\n\\n 不会触发切分）
        paragraphs = protected_text.split("\n\n")

        # Step 3: 累计组装 + 还原 + 字节检查
        chunks: list[Chunk] = []
        buffer = ""
        idx = start_idx

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(buffer) + len(para) <= self.max_chunk:
                buffer += ("\n\n" if buffer else "") + para
            else:
                if buffer.strip():
                    chunk_text = self._restore_atomic_blocks(buffer.strip(), placeholder_map)
                    for sub in self._fit_byte_limit(
                        chunk_text, doc_id, series, spec_number,
                        release, parent_id, parent_title,
                        section_number, section_title, section_path,
                        idx, doc_type,
                    ):
                        chunks.append(sub)
                        idx += 1
                buffer = para

        if buffer.strip():
            chunk_text = self._restore_atomic_blocks(buffer.strip(), placeholder_map)
            for sub in self._fit_byte_limit(
                chunk_text, doc_id, series, spec_number,
                release, parent_id, parent_title,
                section_number, section_title, section_path,
                idx, doc_type,
            ):
                chunks.append(sub)

        return chunks

    def _split_long_section_dynamic(
        self,
        text: str,
        doc_id: str,
        series: int,
        spec_number: str,
        release: str,
        parent_id: str,
        parent_title: str,
        section_number: str,
        section_title: str,
        section_path: str,
        start_idx: int,
        doc_type: str = "3gpp",
    ) -> list[Chunk]:
        """Dynamic 模式：按原子块边界分离表格/正文为独立 chunk.

        1. 用 _segment_by_atomic_blocks 拆分为 (type, text) 列表
        2. 表格/公式 → 独立 chunk，上限 table_max_chars
        3. 纯文本 → 段落切分，上限 prose_max_chars
        """
        segments = self._segment_by_atomic_blocks(text)
        if not segments:
            return []

        chunks: list[Chunk] = []
        idx = start_idx

        for seg_type, seg_text in segments:
            seg_text = seg_text.strip()
            if not seg_text:
                continue

            seg_len = len(seg_text)
            is_table = seg_type in ("grid_table", "pipe_table", "math_block")
            max_limit = self.table_max_chars if is_table else self.prose_max_chars

            if seg_len <= max_limit:
                for sub in self._fit_byte_limit(
                    seg_text, doc_id, series, spec_number,
                    release, parent_id, parent_title,
                    section_number, section_title, section_path,
                    idx, doc_type,
                ):
                    chunks.append(sub)
                    idx += 1
            elif is_table:
                # 超大表格：按表格类型走对应行组拆分 (保留表头 + caption)
                if seg_type == "pipe_table":
                    sub_texts = self._split_pipe_table_rows(seg_text)
                elif seg_type == "grid_table":
                    sub_texts = self._split_grid_table_rows(seg_text)
                else:  # math_block 等
                    sub_texts = self._split_oversized(seg_text)
                for sub_text in sub_texts:
                    for sub in self._fit_byte_limit(
                        sub_text, doc_id, series, spec_number,
                        release, parent_id, parent_title,
                        section_number, section_title, section_path,
                        idx, doc_type,
                    ):
                        chunks.append(sub)
                        idx += 1
            else:
                # 超长纯文本：段落边界切分 + overlap (相邻 chunk 共享句尾上下文)
                paragraphs = [p.strip() for p in seg_text.split("\n\n") if p.strip()]
                buf = ""
                for i, para in enumerate(paragraphs):
                    if len(buf) + len(para) <= max_limit:
                        buf += ("\n\n" if buf else "") + para
                        continue
                    # 落盘当前 buffer — 追加下一段首部作为 overlap (句界切断)
                    overlap_limit = min(self.overlap, max_limit - len(buf))
                    overlap_tail = self._take_overlap(para, overlap_limit)
                    if buf.strip():
                        chunk_text = buf.strip()
                        if overlap_tail:
                            chunk_text = f"{chunk_text}\n\n{overlap_tail}"
                        for sub in self._fit_byte_limit(
                            chunk_text, doc_id, series, spec_number,
                            release, parent_id, parent_title,
                            section_number, section_title, section_path,
                            idx, doc_type,
                        ):
                            chunks.append(sub)
                            idx += 1
                    buf = para
                if buf.strip():
                    for sub in self._fit_byte_limit(
                        buf.strip(), doc_id, series, spec_number,
                        release, parent_id, parent_title,
                        section_number, section_title, section_path,
                        idx, doc_type,
                    ):
                        chunks.append(sub)

        return chunks

    # ── 小碎片合并 + overlap 工具 ──

    @staticmethod
    def _take_overlap(text: str, limit: int) -> str:
        """取文本前 limit 字符作为 chunk 尾部 overlap — 按句界切断.

        边界优先级: 句末 (.  。) → 分句 (; ) → 换行 → 词 (空格) → 硬切.
        返回的 overlap 与下一 chunk 首部内容重叠, 提供共享语义上下文.
        """
        if limit <= 0 or not text:
            return ""
        window = text[:limit]
        for sep in (". ", "。", "; ", "\n", " "):
            idx = window.rfind(sep)
            if idx >= max(limit // 2, 1):
                return window[: idx + len(sep)].rstrip()
        return window

    @staticmethod
    def _is_atomic_chunk(text: str) -> bool:
        """检测 chunk 是否为表格/公式原子块 — 合并时必须保护, 不参与合并."""
        if _contains_table(text):
            return True
        stripped = text.strip()
        return stripped.startswith("$$") and "$$" in stripped[2:]

    def _merge_small_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """合并过小 chunk — 向前并入优先, 原子块 (表格/公式) 永不参与.

        规则:
        - 正文 chunk 长度 < min_chunk_chars → 视为碎片
        - 碎片优先并入前一个非原子 chunk (向前); 否则并入下一个非原子 chunk
        - 原子块 chunk 既不被并入也不作为合并目标 (表格完整性保护)
        - 合并后长度不超 max_chunk_hard (BGE-M3 8192 token 安全上限)
        """
        if self.min_chunk_chars <= 0 or len(chunks) < 2:
            return chunks

        def _is_fragment(c: Chunk) -> bool:
            return len(c.text) < self.min_chunk_chars and not self._is_atomic_chunk(c.text)

        result: list[Chunk] = []
        for c in chunks:
            if (
                _is_fragment(c)
                and result
                and not self._is_atomic_chunk(result[-1].text)
                and len(result[-1].text) + len(c.text) <= self.max_chunk_hard
            ):
                result[-1].text = f"{result[-1].text}\n\n{c.text}"
                continue
            result.append(c)

        # 前向兜底: 剩余碎片并入下一个非原子 chunk
        i = 0
        while i < len(result) - 1:
            if (
                _is_fragment(result[i])
                and not self._is_atomic_chunk(result[i + 1].text)
                and len(result[i].text) + len(result[i + 1].text) <= self.max_chunk_hard
            ):
                result[i + 1].text = f"{result[i].text}\n\n{result[i + 1].text}"
                del result[i]
                continue
            i += 1

        return result

    # ── 字节上限自适应拆分 ──

    def _fit_byte_limit(
        self,
        text: str,
        doc_id: str,
        series: int,
        spec_number: str,
        release: str,
        parent_id: str,
        parent_title: str,
        section_number: str,
        section_title: str,
        section_path: str,
        start_idx: int,
        doc_type: str = "3gpp",
    ) -> list[Chunk]:
        """确保 chunk 不超 max_chunk_bytes (Milvus) 且不超 max_chunk_hard (BGE-M3).

        两个硬限制:
        1. max_chunk_bytes=55000  — Milvus VARCHAR(65535) 安全边距
        2. max_chunk_hard=8000   — BGE-M3 8192 tokens, 超限静默截断会丢尾部

        超限时按行→句→词边界自适应拆分, 并在子 chunk 间保留 overlap.
        """
        text_bytes = len(text.encode("utf-8"))
        text_chars = len(text)

        # 两个限制都满足 — 直接返回单 chunk
        if text_bytes <= self.max_chunk_bytes and text_chars <= self.max_chunk_hard:
            return [Chunk(
                text=text, embedding=None,
                doc_id=doc_id, series=series,
                spec_number=spec_number, release=release,
                parent_section_id=parent_id, parent_title=parent_title,
                chunk_index=start_idx,
                section_number=section_number,
                section_title=section_title,
                section_path=section_path,
                doc_type=doc_type,
            )]

        # 确定触发原因并选择拆分策略
        if text_chars > self.max_chunk_hard:
            logger.debug(
                "chunk 超 BGE-M3 限制 (%d chars > %d), 强制切分: %s/%s",
                text_chars, self.max_chunk_hard, spec_number, parent_title[:60],
            )
            sub_texts = self._split_by_char_limit(text, self.max_chunk_hard)
        else:
            logger.info(
                "chunk 超字节上限 (%dB > %dB), 按内容类型拆分: %s/%s",
                text_bytes, self.max_chunk_bytes, spec_number, parent_title[:60],
            )
            sub_texts = self._split_oversized(text)

        return [
            Chunk(
                text=sub, embedding=None,
                doc_id=doc_id, series=series,
                spec_number=spec_number, release=release,
                parent_section_id=parent_id, parent_title=parent_title,
                chunk_index=start_idx + i,
                section_number=section_number,
                section_title=section_title,
                section_path=section_path,
                doc_type=doc_type,
            )
            for i, sub in enumerate(sub_texts)
        ]

    def _split_oversized(self, text: str) -> list[str]:
        """对超字节上限的内容按类型自适应拆分.

        检测优先级:
        1. Grid Table (Pandoc +---+) → 行组拆分, 每组子表保留表头
        2. HTML Table (<table>)        → <tr> 行拆分
        3. 其他                        → 换行边界拆分 (兜底)
        """
        lines = text.split("\n")
        # 检测窗口: 规范文档中表格前常有章节标题 + 一段说明,
        # 50 行覆盖了最极端的情况 (实测最长 prose 前缀 ~40 行)
        head = lines[: min(50, len(lines))]
        plus_count = sum(1 for line in head if line.strip().startswith("+"))
        if plus_count >= 2:
            return self._split_grid_table_rows(text)

        if "<table" in text[:500].lower():
            return self._split_html_table_rows(text)

        return self._split_text_by_lines(text)

    def _split_grid_table_rows(self, text: str) -> list[str]:
        """拆分巨型 Pandoc Grid Table — 在 +---+ 行组边界切断, 零信息丢失.

        表头策略:
        1. 表格前的文档上下文 (prose + heading) → 完整保留在每个子表
           (上下文大小由上游 max_chunk_chars buffer 自然限制, 无需截断)
        2. 表格列头 (第一个 + 到 +===+) → 始终保留

        极端情况:
        - 合并单元格表格 (单组 > 600KB)
          → 行组内按换行拆分, 子表仍保留完整表头上下文
        """
        lines = text.split("\n")

        # ── 1. 找表格起始 (第一个 + 行) ──
        table_start = 0
        for i, line in enumerate(lines):
            if line.strip().startswith("+"):
                table_start = i
                break

        # ── 2. 文档上下文 = 表格前的所有行 (零丢失: 不截断, 完整保留) ──
        # 上下文大小由上游 _split_long_section 的 max_chunk_chars buffer 自然限制,
        # 通常 ≤ 10KB, 不会造成子表 header 膨胀。
        doc_context = lines[:table_start]

        # ── 3. 表格列头: table_start → 第一个 +===+ (限表格前 40 行搜索) ──
        col_header_end = table_start
        search_end = min(table_start + 40, len(lines))
        for i in range(table_start, search_end):
            if "===" in lines[i]:
                col_header_end = i + 1
                break
        if col_header_end == table_start:
            col_header_end = table_start + 1  # 至少取第一行边框

        # ── 4. 合成表头: 文档上下文 + 表格列头 ──
        header_lines = doc_context + lines[table_start:col_header_end]

        # ── 5. 收集数据行组 (所有 + 线为边界, 中间内容自动归组) ──
        # 规范文档表格的 Pandoc 输出极其异构:
        #   - 标准表:    +---+  content  +---+  +---+  content  +---+
        #   - 合并单元格: +---+  content  +---+  content  +---+
        #                 (一个 +---+ 既是关闭也是开启, 无连续双边框)
        #   - 子表头:     +===+ (语义同 +---+)
        # 此处的无状态收集方案统一处理以上所有情况, 零行丢失.
        data_lines = lines[col_header_end:]
        groups: list[list[str]] = []
        current: list[str] = []

        for line in data_lines:
            is_sep = line.strip().startswith("+")
            if is_sep:
                # flush current group if it has actual content (not just borders)
                if current and any(not ln.strip().startswith("+") for ln in current):
                    groups.append(current)
                current = [line]
            else:
                current.append(line)

        # 最后一段: 仅含实际内容才保留 (尾部孤立的 +---+ 不生成无效组)
        if current and any(not ln.strip().startswith("+") for ln in current):
            groups.append(current)

        if not groups:
            return [text]

        # ── 6. 行组合并到 table_max_chars (字符, 语义单元化) ──
        def _build_table(row_groups: list[list[str]]) -> str:
            """用表头 + 行组构建子表文本."""
            all_lines = list(header_lines)
            for bg in row_groups:
                all_lines.extend(bg)
            return "\n".join(all_lines)

        # 用字符数而非字节数, 目标 = table_max_chars (大表拆成更小的自描述行组)
        split_limit = self.table_max_chars
        header_chars = sum(len(line) + 1 for line in header_lines)

        result: list[str] = []
        buf_groups: list[list[str]] = []
        buf_chars = header_chars

        for g in groups:
            g_chars = sum(len(ln) + 1 for ln in g)

            if g_chars > split_limit:
                # 合并单元格: 单行组超大 → 先 flush 已缓存行组
                if buf_groups:
                    result.append(_build_table(buf_groups))
                    buf_groups = []
                    buf_chars = header_chars

                # 行组内按换行拆分 (每段仍带表头)
                inner_buf: list[str] = []
                inner_chars = header_chars
                for line in g:
                    lc = len(line) + 1
                    if inner_chars + lc > split_limit and inner_buf:
                        result.append("\n".join(header_lines + inner_buf))
                        inner_buf = [line]
                        inner_chars = header_chars + lc
                    else:
                        inner_buf.append(line)
                        inner_chars += lc
                if inner_buf:
                    result.append("\n".join(header_lines + inner_buf))
            elif buf_chars + g_chars > split_limit and buf_groups:
                result.append(_build_table(buf_groups))
                buf_groups = [g]
                buf_chars = header_chars + g_chars
            else:
                buf_groups.append(g)
                buf_chars += g_chars

        if buf_groups:
            result.append(_build_table(buf_groups))

        logger.debug(
            "Grid Table 拆分: %d chars → %d 子表 (上限 %d)",
            len(text), len(result), split_limit,
        )
        return result if result else [text]

    def _split_pipe_table_rows(self, text: str) -> list[str]:
        """拆分超大 pipe 表 (marked 数据集) — 每片保留表头(列名+分隔行)+caption 上下文.

        3GPP marked 数据集的表格是 pipe 表 (|---| 分隔), 此前 >table_max_chars 的
        pipe 表走 _split_text_by_lines 兜底 (换行拆分, 破坏表结构)。此函数按行组
        语义拆分, 目标 table_max_chars, 每片自描述。
        """
        lines = text.split("\n")

        # 找表头起始 (第一个 pipe 行)
        table_start = 0
        for i, line in enumerate(lines):
            if PIPE_TABLE_LINE.match(line.strip()):
                table_start = i
                break

        doc_context = lines[:table_start]  # caption / section 说明
        # 表头 = 列名行 + 分隔行 (至少 2 行)
        header_end = table_start + 2
        header_lines = doc_context + lines[table_start:header_end]
        header_chars = sum(len(line) + 1 for line in header_lines)

        data_lines = lines[header_end:]
        split_limit = self.table_max_chars

        result: list[str] = []
        buf: list[str] = []
        buf_chars = header_chars
        for line in data_lines:
            lc = len(line) + 1
            if buf_chars + lc > split_limit and buf:
                result.append("\n".join(header_lines + buf))
                buf = [line]
                buf_chars = header_chars + lc
            else:
                buf.append(line)
                buf_chars += lc
        if buf:
            result.append("\n".join(header_lines + buf))

        logger.debug(
            "Pipe Table 拆分: %d chars → %d 子表 (上限 %d)",
            len(text), len(result), split_limit,
        )
        return result if result else [text]

    def _split_html_table_rows(self, text: str) -> list[str]:
        """拆分 HTML <table> — 在 <tr> 行边界切断, 每组子表保留 <thead>.

        极端情况: 单个 <tr> 超过 max_chunk_bytes → 行内按换行拆分,
        每段仍带 <thead> + <table> 包裹以保证 HTML 结构完整。
        """
        import re

        thead_match = re.search(
            r"(<table[^>]*>.*?</thead>)", text, re.DOTALL | re.IGNORECASE,
        )
        header = thead_match.group(1) if thead_match else ""
        if not header:
            table_open = re.match(r"(<table[^>]*>)", text, re.IGNORECASE)
            header = table_open.group(1) if table_open else "<table>"

        close_tag = "</table>"
        header_size = len(header.encode("utf-8")) + len(close_tag.encode("utf-8"))

        rows = re.findall(r"(<tr[^>]*>.*?</tr>)", text, re.DOTALL | re.IGNORECASE)
        if not rows:
            return self._split_text_by_lines(text)

        result: list[str] = []
        buf_rows: list[str] = []
        buf_size = header_size

        for row in rows:
            row_size = len(row.encode("utf-8"))

            if row_size > self.max_chunk_bytes:
                # 单个 <tr> 超过上限: flush 已缓存, 行内按换行拆分
                if buf_rows:
                    result.append(header + "\n" + "\n".join(buf_rows) + "\n" + close_tag)
                    buf_rows = []
                    buf_size = header_size

                # 行内拆分 (保留 HTML 结构包裹)
                inner_lines = row.split("\n")
                inner_buf: list[str] = []
                inner_bytes = header_size
                for line in inner_lines:
                    lb = len(line.encode("utf-8")) + 1
                    if inner_bytes + lb > self.max_chunk_bytes and inner_buf:
                        result.append(header + "\n" + "\n".join(inner_buf) + "\n" + close_tag)
                        inner_buf = [line]
                        inner_bytes = header_size + lb
                    else:
                        inner_buf.append(line)
                        inner_bytes += lb
                if inner_buf:
                    result.append(header + "\n" + "\n".join(inner_buf) + "\n" + close_tag)
            elif buf_size + row_size > self.max_chunk_bytes and buf_rows:
                result.append(header + "\n" + "\n".join(buf_rows) + "\n" + close_tag)
                buf_rows = [row]
                buf_size = header_size + row_size
            else:
                buf_rows.append(row)
                buf_size += row_size

        if buf_rows:
            result.append(header + "\n" + "\n".join(buf_rows) + "\n" + close_tag)

        return result if result else [text]

    def _split_text_by_lines(self, text: str) -> list[str]:
        """通用行级拆分 — 在换行边界切断 (兜底策略, 极少触发)."""
        lines = text.split("\n")
        result: list[str] = []
        buf: list[str] = []
        buf_bytes = 0

        for line in lines:
            line_bytes = len(line.encode("utf-8")) + 1  # +1 for \n
            if buf_bytes + line_bytes > self.max_chunk_bytes and buf:
                result.append("\n".join(buf))
                buf = [line]
                buf_bytes = line_bytes
            else:
                buf.append(line)
                buf_bytes += line_bytes

        if buf:
            result.append("\n".join(buf))

        return result if result else [text]

    def _split_by_char_limit(self, text: str, max_chars: int) -> list[str]:
        """按字符上限强制切分 — 在最优语义边界切断，保留子 chunk 重叠。

        拆分优先级: 换行 → 句末 (. ) → 分号 (; ) → 空格 → 硬切
        每个子 chunk 尾部附加 overlap (前一个子 chunk 的后 ~15% 内容)，
        确保 BGE-M3 嵌入时相邻 chunk 共享上下文信号。
        """
        if len(text) <= max_chars:
            return [text]

        # overlap = max_chars 的 15%，最少 100 chars
        overlap = max(int(max_chars * 0.15), 100)
        result: list[str] = []
        pos = 0

        while pos < len(text):
            end = min(pos + max_chars, len(text))
            if end >= len(text):
                result.append(text[pos:])
                break

            # 在 (end - overlap, end] 区间找最佳切点
            window = text[max(pos, end - max_chars // 2):end]
            # 优先级: 换行 → 句末 → 分句 → 空格
            cut = -1
            for sep in ("\n\n", "\n", ". ", "。", "; ", " "):
                idx = window.rfind(sep)
                if idx >= 0:
                    cut = max(pos, end - max_chars // 2) + idx + len(sep)
                    break
            if cut < 0:
                cut = end  # 硬切

            result.append(text[pos:cut])
            # 下一段起始 = cut - overlap (保留重叠上下文)
            pos = max(cut - overlap, pos + 1)

        return result

    # ── 原子块保护 ──

    @staticmethod
    def _protect_atomic_blocks(text: str) -> tuple[str, dict[str, str]]:
        """用占位符替换 Grid Table / Math Block / Pipe Table.

        换行扫描，识别以下原子块并替换为 __TYPE_N__ 占位符：
          - Grid Table: +---+ 开始，最后一个 +---+ 结束
          - Math Block: $$ 开始，下一个 $$ 结束
          - Pipe Table: |...| + |---| + |...| 连续行

        Returns:
            (protected_text, {placeholder: original_content})
        """
        lines = text.split('\n')
        result: list[str] = []
        pmap: dict[str, str] = {}
        pidx = 0
        i = 0
        n = len(lines)

        while i < n:
            stripped = lines[i].strip()

            # ── Grid Table ──
            if GRID_TABLE_BOUNDARY.match(stripped):
                start_i = i
                i += 1
                while i < n:
                    s = lines[i].strip()
                    if GRID_TABLE_BOUNDARY.match(s):
                        # 下一行如果是 |...| → 表格继续；否则这是结束行
                        next_is_data = (i + 1 < n and
                                       lines[i + 1].strip().startswith('|'))
                        if not next_is_data:
                            i += 1  # 包含结束行
                            break
                    i += 1

                table_text = '\n'.join(lines[start_i:i])
                placeholder = f'__GRIDTABLE_{pidx}__'
                pmap[placeholder] = table_text
                result.append(placeholder)
                pidx += 1
                continue

            # ── Math Block ──
            if MATH_BLOCK_DELIM.match(stripped):
                start_i = i
                i += 1
                while i < n:
                    if MATH_BLOCK_DELIM.match(lines[i].strip()):
                        i += 1  # 包含结束 $$
                        break
                    i += 1

                math_text = '\n'.join(lines[start_i:i])
                placeholder = f'__MATHBLOCK_{pidx}__'
                pmap[placeholder] = math_text
                result.append(placeholder)
                pidx += 1
                continue

            # ── Pipe Table ──
            if PIPE_TABLE_LINE.match(stripped):
                if i + 1 < n and PIPE_TABLE_SEP.match(lines[i + 1].strip()):
                    start_i = i
                    i += 2  # 跳过表头 + 分隔行
                    while i < n:
                        if not PIPE_TABLE_LINE.match(lines[i].strip()):
                            break
                        i += 1

                    table_text = '\n'.join(lines[start_i:i])
                    placeholder = f'__PIPETABLE_{pidx}__'
                    pmap[placeholder] = table_text
                    result.append(placeholder)
                    pidx += 1
                    continue

            result.append(lines[i])
            i += 1

        return '\n'.join(result), pmap

    @staticmethod
    def _restore_atomic_blocks(text: str, placeholder_map: dict[str, str]) -> str:
        """将占位符还原为原始原子块内容."""
        for placeholder, original in placeholder_map.items():
            text = text.replace(placeholder, original)
        return text

    # ── 降级：字符分块 ──

    def _split_by_chars(
        self,
        text: str,
        doc_id: str,
        series: int,
        spec_number: str,
        release: str,
        doc_type: str = "3gpp",
    ) -> list[Chunk]:
        """无标题时的固定窗口字符分块 (兼容 Phase 1).

        使用配置的 max_chunk/overlap 而非硬编码值.
        """
        chunk_size = max(self.max_chunk, 500)
        overlap = min(self.overlap, chunk_size // 4)
        separators = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')

        chunks: list[Chunk] = []
        start = 0
        idx = 0
        while start < len(text) - overlap:
            end = min(start + chunk_size, len(text))
            match = separators.search(text, end)
            if match:
                end = match.end()
            if end == start:
                end = start + 1
            chunk_text = text[start:end]
            chunks.append(Chunk(
                text=chunk_text,
                embedding=None,
                doc_id=doc_id, series=series,
                spec_number=spec_number, release=release,
                parent_section_id="", parent_title="",
                chunk_index=idx,
                doc_type=doc_type,
            ))
            idx += 1
            start = end - overlap
            match = separators.search(text, start - 1)
            if match:
                start = match.start() + 1
            if start < 0:
                start = 0
        return chunks


def build_splitter() -> HeaderAwareSplitter:
    """按 IngestionConfig 统一构造分块器 — 全量/增量摄入共用的唯一工厂.

    背景: 此前 orchestrator / bulk_ingest 用 IngestionConfig 构造, 而 incremental
    用 HeaderAwareSplitter() 硬编码默认值 (2500/100), 导致两条路径 chunk 边界 /
    overlap / 大小不一致, 向量空间分叉 (违背 embedding_text 的单一真源原则)。
    所有摄入路径必须经由此工厂构造, 不得自行 new HeaderAwareSplitter()。
    """
    from src.config import ingestion_config

    return HeaderAwareSplitter(
        max_chunk_chars=ingestion_config.chunk_size,
        chunk_overlap_chars=ingestion_config.chunk_overlap,
        max_chunk_bytes=55000,
        chunk_mode=ingestion_config.chunk_mode,  # type: ignore[arg-type]
        table_max_chars=ingestion_config.table_max_chars,
        prose_max_chars=ingestion_config.prose_max_chars,
        max_chunk_hard_chars=ingestion_config.max_chunk_chars,
        min_chunk_chars=ingestion_config.min_chunk_chars,
    )
