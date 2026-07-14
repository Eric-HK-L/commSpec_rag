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
    ):
        self.max_chunk = max_chunk_chars
        self.overlap = chunk_overlap_chars

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

        # 构建章节树
        root = self._build_section_tree(text)
        if root.children:
            chunks = self._split_by_tree(text, root, doc_id, series, spec_number, release)
        else:
            # 无标题 → 降级为字符分块
            chunks = self._split_by_chars(text, doc_id, series, spec_number, release)

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
                ))
                chunk_idx += 1
            else:
                sub_chunks = self._split_long_section(
                    content, doc_id, series, spec_number,
                    release, parent_id, parent_title, chunk_idx,
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
        start_idx: int,
    ) -> list[Chunk]:
        """对超长文本在段落边界二次切分 — 表格/公式原子化保护.

        策略:
        1. 找到所有 Grid Table / Math Block / Pipe Table
        2. 用占位符替换 → 安全切分 (\\n\\n 不会误伤表格)
        3. 累计段落至 size 限制 → 还原占位符 → 输出 chunk
        4. 超长原子块 (单表 > max_chunk) 保持完整，宁可大块不断裂
        """
        # Step 1: 保护原子块
        protected_text, placeholder_map = self._protect_atomic_blocks(text)

        # Step 2: 安全切分（原子块被占位符隐藏，内部 \\n\\n 不会触发切分）
        paragraphs = protected_text.split("\n\n")

        # Step 3: 累计组装 + 还原
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
                    chunks.append(Chunk(
                        text=chunk_text,
                        embedding=None,
                        doc_id=doc_id, series=series,
                        spec_number=spec_number, release=release,
                        parent_section_id=parent_id,
                        parent_title=parent_title,
                        chunk_index=idx,
                    ))
                    idx += 1
                buffer = para

        if buffer.strip():
            chunk_text = self._restore_atomic_blocks(buffer.strip(), placeholder_map)
            chunks.append(Chunk(
                text=chunk_text,
                embedding=None,
                doc_id=doc_id, series=series,
                spec_number=spec_number, release=release,
                parent_section_id=parent_id,
                parent_title=parent_title,
                chunk_index=idx,
            ))

        return chunks

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
            ))
            idx += 1
            start = end - overlap
            match = separators.search(text, start - 1)
            if match:
                start = match.start() + 1
            if start < 0:
                start = 0
        return chunks
