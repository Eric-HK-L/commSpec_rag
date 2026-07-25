"""标题感知文档分块器 — Markdown / 纯文本通用，结构感知（表格/公式原子化保护）.

支持两种策略:
  1. header_split: 解析 #~#### 标题, 按最深层级切分 (适合 3GPP 规范)
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
from typing import Any

from src.retriever.vector_store import Chunk

logger = logging.getLogger(__name__)

# Markdown 标题正则 — 匹配 1-8 级 (pandoc 3GPP Annex 用 ######## 级别)
HEADER_RE = re.compile(r"^(#{1,8})\s+(.+)$", re.MULTILINE)
# 3GPP 规范章节编号 (如 "6.1.2  PDU Session Establishment")
SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.{3,})$", re.MULTILINE)
# 纯文本 TOC 章节行 (如 "1 Scope   7") — mammoth 输出降级
PLAIN_SECTION_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+([A-Z]\w.{2,}?)\s+\d+\s*$", re.MULTILINE)

# ── 原子块检测 (不可分割的结构) ──
# Grid Table: pandoc +---+---+ 格式
GRID_TABLE_BOUNDARY = re.compile(r'^\+[-=+]+\+$')
# Math display block: $$ ... $$
MATH_BLOCK_DELIM = re.compile(r'^\$\$$')
# Pipe Table: | a | b |
PIPE_TABLE_LINE = re.compile(r'^\|.+\|$')
PIPE_TABLE_SEP = re.compile(r'^\|[-: ]+\|$')


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
    - 超长节点在段落边界二次切分（原子块内部分隔符被保护）
    - 每个 chunk 自动附加 parent_section_id/parent_title
    """

    def __init__(
        self,
        max_chunk_chars: int = 2500,
        chunk_overlap_chars: int = 100,
        max_chunk_bytes: int = 55000,
    ):
        self.max_chunk = max_chunk_chars
        self.overlap = chunk_overlap_chars
        # Milvus VARCHAR 65535 bytes 硬限制, 留 ~10KB 安全边距
        self.max_chunk_bytes = max_chunk_bytes

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

        return chunks

    # ── 章节树构建 ──

    def _build_section_tree(self, text: str) -> SectionNode:
        """解析标题构建嵌套章节树 — Markdown # 标题 + 纯文本 TOC 降级."""
        root = SectionNode(level=0, title="", sec_id="", start=0, end=len(text))

        # 收集所有标题位置
        headers: list[tuple[int, int, str, str]] = []  # (start, level, sec_id, title)
        for m in HEADER_RE.finditer(text):
            level = len(m.group(1))
            # Cap pandoc deep levels: ######## Annex → treat as level 1 (top-level)
            if level > 4:
                level = 1
            title = m.group(2).strip()
            sec_id = ""
            num_match = SECTION_NUM_RE.match(title)
            if num_match:
                sec_id = num_match.group(1)
                title = num_match.group(2).strip()
            headers.append((m.start(), level, sec_id, title))

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
        """从章节树叶子节点切分 chunk."""
        chunks: list[Chunk] = []
        leaf_nodes = self._collect_leaves(root)

        chunk_idx = 0
        for node in leaf_nodes:
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
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)

        return chunks

    def _collect_leaves(self, node: SectionNode) -> list[SectionNode]:
        """收集所有叶子节点."""
        if not node.children:
            return [node] if node.level > 0 else []
        leaves: list[SectionNode] = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves

    def _get_parent_context(self, node: SectionNode) -> tuple[str, str]:
        """获取节点的父章节编号和标题."""
        ids: list[str] = []
        titles: list[str] = []
        current = node.parent
        while current and current.level > 0:
            if current.sec_id:
                ids.append(current.sec_id)
            if current.title:
                titles.append(current.title)
            current = current.parent

        parent_id = ".".join(reversed(ids)) if ids else ""
        parent_title = " > ".join(reversed(titles)) if titles else ""
        return parent_id, parent_title

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
        """确保 chunk 字节数不超过 max_chunk_bytes, 超限则按内容类型自适应拆分."""
        text_bytes = len(text.encode("utf-8"))
        if text_bytes <= self.max_chunk_bytes:
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
        # 检测窗口: 3GPP 规范中表格前常有章节标题 + 一段说明,
        # 50 行覆盖了最极端的情况 (实测最长 prose 前缀 ~40 行)
        head = lines[: min(50, len(lines))]
        plus_count = sum(1 for l in head if l.strip().startswith("+"))
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
        header_bytes = sum(len(l.encode("utf-8")) + 1 for l in header_lines)

        # ── 5. 收集数据行组 (所有 + 线为边界, 中间内容自动归组) ──
        # 3GPP 表格的 Pandoc 输出极其异构:
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
                if current and any(not l.strip().startswith("+") for l in current):
                    groups.append(current)
                current = [line]
            else:
                current.append(line)

        # 最后一段: 仅含实际内容才保留 (尾部孤立的 +---+ 不生成无效组)
        if current and any(not l.strip().startswith("+") for l in current):
            groups.append(current)

        if not groups:
            return [text]

        # ── 6. 行组合并到 max_chunk_bytes ──
        def _build_table(row_groups: list[list[str]]) -> str:
            """用表头 + 行组构建子表文本."""
            all_lines = list(header_lines)
            for bg in row_groups:
                all_lines.extend(bg)
            return "\n".join(all_lines)

        result: list[str] = []
        buf_groups: list[list[str]] = []
        buf_bytes = header_bytes

        for g in groups:
            g_bytes = sum(len(l.encode("utf-8")) + 1 for l in g)

            if g_bytes > self.max_chunk_bytes:
                # 合并单元格: 单行组超大 → 先 flush 已缓存行组
                if buf_groups:
                    result.append(_build_table(buf_groups))
                    buf_groups = []
                    buf_bytes = header_bytes

                # 行组内按换行拆分 (每段仍带表头)
                inner_buf: list[str] = []
                inner_bytes = header_bytes
                for line in g:
                    lb = len(line.encode("utf-8")) + 1
                    if inner_bytes + lb > self.max_chunk_bytes and inner_buf:
                        result.append("\n".join(header_lines + inner_buf))
                        inner_buf = [line]
                        inner_bytes = header_bytes + lb
                    else:
                        inner_buf.append(line)
                        inner_bytes += lb
                if inner_buf:
                    result.append("\n".join(header_lines + inner_buf))
            elif buf_bytes + g_bytes > self.max_chunk_bytes and buf_groups:
                result.append(_build_table(buf_groups))
                buf_groups = [g]
                buf_bytes = header_bytes + g_bytes
            else:
                buf_groups.append(g)
                buf_bytes += g_bytes

        if buf_groups:
            result.append(_build_table(buf_groups))

        logger.debug(
            "Grid Table 拆分: %dB → %d 子表 (均 ≤ %dB)",
            len(text.encode("utf-8")),
            len(result),
            max(len(r.encode("utf-8")) for r in result) if result else 0,
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
