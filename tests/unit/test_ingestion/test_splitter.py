"""splitter.py 单元测试 — 原子块保护与章节解析（纯函数）."""

from src.ingestion.splitter import (
    GRID_TABLE_BOUNDARY,
    HEADER_RE,
    MATH_BLOCK_DELIM,
    SECTION_NUM_RE,
    HeaderAwareSplitter,
)
from src.retriever.vector_store import Chunk


def _para(sentence: str, target_len: int) -> str:
    """构建单段文本 (无内部空行) — 重复句子至接近 target_len 字符."""
    out = ""
    while len(out) + len(sentence) <= target_len:
        out += sentence
    return out + sentence[: target_len - len(out)]


class TestSectionNumRegex:

    def test_simple(self):
        m = SECTION_NUM_RE.match("6.1.2  PDU Session Establishment")
        assert m is not None
        assert m.group(1) == "6.1.2"
        assert m.group(2) == "PDU Session Establishment"

    def test_two_level(self):
        m = SECTION_NUM_RE.match("8.3  NGAP Procedures")
        assert m is not None
        assert m.group(1) == "8.3"

    def test_no_match(self):
        m = SECTION_NUM_RE.match("Introduction")
        assert m is None

    def test_single_digit(self):
        m = SECTION_NUM_RE.match("5  General")
        assert m is not None
        assert m.group(1) == "5"


class TestHeaderRegex:

    def test_h1(self):
        m = HEADER_RE.match("# Scope")
        assert m is not None
        assert m.group(2).strip() == "Scope"

    def test_h3(self):
        m = HEADER_RE.match("### 8.3.1  PDU Session Resource Setup")
        assert m is not None

    def test_h8_annex(self):
        # pandoc 3GPP Annex: ######## Annex A
        m = HEADER_RE.match("######## Annex A (normative)")
        assert m is not None
        assert len(m.group(1)) == 8

    def test_no_header(self):
        m = HEADER_RE.match("This is plain text")
        assert m is None


class TestGridTableDetection:

    def test_boundary_match(self):
        assert GRID_TABLE_BOUNDARY.match("+-----+------+") is not None
        assert GRID_TABLE_BOUNDARY.match("+===+===+") is not None

    def test_no_false_positive(self):
        assert GRID_TABLE_BOUNDARY.match("+ not a table") is None
        assert GRID_TABLE_BOUNDARY.match("| table |") is None


class TestMathBlockDetection:

    def test_delim_match(self):
        assert MATH_BLOCK_DELIM.match("$$") is not None

    def test_no_match(self):
        assert MATH_BLOCK_DELIM.match("$inline$") is None


class TestAtomicBlockProtection:

    def test_protect_grid_table(self):
        text = "Before\n+-----+------+\n| a   | b    |\n+-----+------+\nAfter"
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(text)
        assert "__GRIDTABLE_0__" in protected
        assert "__GRIDTABLE_0__" in pmap
        assert "+-----+" in pmap["__GRIDTABLE_0__"]

    def test_protect_math_block(self):
        text = "Text\n$$\nE = mc^2\n$$\nMore text"
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(text)
        assert "__MATHBLOCK_0__" in protected
        assert "E = mc^2" in pmap["__MATHBLOCK_0__"]

    def test_protect_pipe_table(self):
        # 单列管道表（regex: ^\|[-: ]+\|$ 不支持多列中的 |）
        text = "Before\n| a |\n|----|\n| 1 |\nAfter"
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(text)
        assert "__PIPETABLE_0__" in protected
        assert "| a |" in pmap["__PIPETABLE_0__"]

    def test_protect_pipe_table_multi_column(self):
        # 3GPP 规范多为 3+ 列表格 — 分隔行 |---|----|----| 必须被识别为 pipe table
        # 修复前 PIPE_TABLE_SEP ^\|[-: ]+\|$ 只匹配 2 段, 3 列表格被当 prose 切碎
        text = (
            "Before\n"
            "| Channel | Symbol | Subcarrier |\n"
            "|---------|--------|------------|\n"
            "| PSS     | 0      | 56..182    |\n"
            "After"
        )
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(text)
        assert "__PIPETABLE_0__" in protected
        assert "| Channel | Symbol | Subcarrier |" in pmap["__PIPETABLE_0__"]
        assert "| PSS     | 0      | 56..182    |" in pmap["__PIPETABLE_0__"]

    def test_segment_pipe_table_multi_column(self):
        # 多列表格应被 _segment_by_atomic_blocks 识别为独立 pipe_table 段
        text = (
            "Intro paragraph.\n\n"
            "| Channel | Symbol |\n"
            "|---------|--------|\n"
            "| PSS     | 0      |\n"
            "| SSS     | 2      |\n"
        )
        segments = HeaderAwareSplitter()._segment_by_atomic_blocks(text)
        types = [t for t, _ in segments]
        assert "pipe_table" in types
        table_seg = next(tx for t, tx in segments if t == "pipe_table")
        assert "| PSS     | 0      |" in table_seg

    def test_restore_roundtrip(self):
        original = "Before\n+---+---+\n| x | y |\n+---+---+\nAfter"
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(original)
        restored = HeaderAwareSplitter._restore_atomic_blocks(protected, pmap)
        assert restored == original

    def test_no_atomic_blocks(self):
        text = "Plain text without any tables or math."
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(text)
        assert protected == text
        assert pmap == {}

    def test_multiple_atomic_blocks(self):
        text = "Start\n+---+---+\n| a | b |\n+---+---+\nMid\n$$\nx=y\n$$\nEnd"
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(text)
        assert "__GRIDTABLE_0__" in protected
        assert "__MATHBLOCK_1__" in protected
        assert len(pmap) == 2

    def test_unclosed_table_at_end(self):
        text = "+---+---+\n| a | b |\n"
        protected, pmap = HeaderAwareSplitter._protect_atomic_blocks(text)
        assert len(pmap) >= 1  # 至少第一行被保护


class TestTakeOverlap:
    """_take_overlap — 追加下一段首部作为 overlap 尾部, 按句界切断."""

    def test_cuts_at_sentence_boundary(self):
        out = HeaderAwareSplitter._take_overlap("Alpha sentence. Beta sentence. Gamma", 20)
        assert out == "Alpha sentence."

    def test_hard_cut_when_no_boundary(self):
        out = HeaderAwareSplitter._take_overlap("aaaaabbbbbcccccddddd", 10)
        assert out == "aaaaabbbbb"

    def test_word_boundary_fallback(self):
        # 窗口内无句界 → 词边界切断 (不硬切中词)
        out = HeaderAwareSplitter._take_overlap("one two three four five", 14)
        assert len(out) <= 14
        assert out.endswith("e")  # "one two three"

    def test_empty_and_zero_limit(self):
        assert HeaderAwareSplitter._take_overlap("", 10) == ""
        assert HeaderAwareSplitter._take_overlap("abc", 0) == ""


class TestDynamicOverlap:
    """dynamic 模式段落切分 overlap 生效 — 相邻 chunk 共享句尾上下文.

    缺陷: 旧实现 _split_long_section_dynamic 段落累计切分无 overlap,
    chunk_overlap 只在 char_split 降级路径生效 (实测 42% chunks < 500 chars).
    """

    def _splitter(self, **kw):
        defaults = dict(
            max_chunk_chars=600,
            chunk_overlap_chars=200,
            prose_max_chars=600,
            min_chunk_chars=0,  # 隔离 overlap, 关闭合并
        )
        defaults.update(kw)
        return HeaderAwareSplitter(**defaults)

    def test_consecutive_chunks_share_overlap(self):
        p1 = _para("Paragraph one content. ", 500)
        p2 = _para("Paragraph two content. ", 500)
        p3 = _para("Paragraph three content. ", 500)
        doc = "# 5.1 Test Section\n\n" + "\n\n".join([p1, p2, p3])

        chunks = self._splitter().split_document(doc)
        assert len(chunks) == 3

        # chunk2 从 p2 开始 (overlap 之后的内容连续; chunk 文本经 strip)
        assert chunks[1].text.startswith(p2.strip())
        assert chunks[2].text.startswith(p3.strip())
        # chunk1 尾部与 chunk2 首部共享的文本 = p2 的非平凡前缀 (overlap ≥ 50 chars)
        shared_len = 0
        for i in range(min(len(p2), len(chunks[0].text)), 0, -1):
            if chunks[0].text.endswith(p2[:i]):
                shared_len = i
                break
        assert shared_len >= 50, f"chunk1 尾部应带 p2 overlap, 实际共享 {shared_len} chars"

    def test_zero_overlap_no_shared_tail(self):
        p1 = _para("Paragraph one content. ", 500)
        p2 = _para("Paragraph two content. ", 500)
        doc = "# 5.1 Test Section\n\n" + "\n\n".join([p1, p2])
        chunks = self._splitter(chunk_overlap_chars=0).split_document(doc)
        assert len(chunks) == 2
        # 无 overlap: chunk1 尾部即 p1 尾部, 不含 p2 内容
        assert not chunks[0].text.endswith("Paragraph two")


class TestMergeSmallChunks:
    """min_chunk_chars 小碎片合并 — 向前并入优先, 表格/公式原子块保护."""

    def _splitter(self, min_chunk: int = 300):
        return HeaderAwareSplitter(min_chunk_chars=min_chunk, chunk_overlap_chars=0)

    @staticmethod
    def _chunk(text: str, idx: int = 0) -> Chunk:
        return Chunk(text=text, doc_id="d", spec_number="38.413", chunk_index=idx)

    def test_small_merges_into_previous(self):
        s = self._splitter()
        out = s._merge_small_chunks([self._chunk("A" * 1000, 0), self._chunk("B" * 200, 1)])
        assert len(out) == 1
        assert out[0].text == "A" * 1000 + "\n\n" + "B" * 200

    def test_first_small_merges_into_next(self):
        s = self._splitter()
        out = s._merge_small_chunks([self._chunk("A" * 200, 0), self._chunk("B" * 1000, 1)])
        assert len(out) == 1
        assert out[0].text == "A" * 200 + "\n\n" + "B" * 1000

    def test_atomic_table_chunk_protected(self):
        """小表格 chunk 不参与合并 — 既不并入邻居也不被邻居吸收."""
        s = self._splitter()
        table = self._chunk("| Parameter |\n|-----------|\n| A         |", 1)
        out = s._merge_small_chunks([
            self._chunk("A" * 1000, 0), table, self._chunk("B" * 1000, 2),
        ])
        assert len(out) == 3  # 表格保持独立 chunk
        assert "| Parameter |" in out[1].text

    def test_consecutive_smalls_all_merged(self):
        s = self._splitter()
        out = s._merge_small_chunks([
            self._chunk("A" * 1000, 0),
            self._chunk("B" * 150, 1),
            self._chunk("C" * 150, 2),
            self._chunk("D" * 1000, 3),
        ])
        assert len(out) == 2
        assert len(out[0].text) == 1000 + 2 + 150 + 2 + 150  # 两次 "\n\n" 分隔

    def test_big_chunks_left_alone(self):
        s = self._splitter()
        out = s._merge_small_chunks([self._chunk("A" * 1000, 0), self._chunk("B" * 1000, 1)])
        assert len(out) == 2


class TestMergeSmallChunksEndToEnd:
    """split_document 全链路 — 段落切分产生的小碎片被合并, chunk_index 重编号."""

    def test_fragment_merged_and_renumbered(self):
        splitter = HeaderAwareSplitter(
            max_chunk_chars=1000, prose_max_chars=1000,
            chunk_overlap_chars=0, min_chunk_chars=300,
        )
        p1 = _para("First paragraph fills the buffer. ", 900)
        p2 = _para("Tiny residual paragraph. ", 200)
        p3 = _para("Third paragraph. ", 900)
        doc = "# 5.1 Test Section\n\n" + "\n\n".join([p1, p2, p3])

        chunks = splitter.split_document(doc)
        # 旧行为: [900, 200, 900] — 200 字碎片; 合并后应消除
        assert len(chunks) == 2
        assert all(len(c.text) >= 300 for c in chunks), "合并后不应残留 <300 字碎片"
        assert [c.chunk_index for c in chunks] == [0, 1], "chunk_index 应连续重编号"
        assert "Tiny residual paragraph" in chunks[0].text  # 并入前一个 chunk

    def test_small_table_survives_merge(self):
        """全链路: 小表格 chunk 即使 < min_chunk_chars 也保持独立."""
        splitter = HeaderAwareSplitter(
            max_chunk_chars=1000, prose_max_chars=1000,
            chunk_overlap_chars=0, min_chunk_chars=300,
        )
        p1 = _para("First paragraph fills the buffer. ", 900)
        table = "| Parameter |\n|-----------|\n| A         |"
        p3 = _para("Third paragraph. ", 900)
        doc = "# 5.1 Test Section\n\n" + "\n\n".join([p1, table, p3])

        chunks = splitter.split_document(doc)
        assert len(chunks) == 3
        table_chunks = [c for c in chunks if "Parameter" in c.text]
        assert len(table_chunks) == 1, "表格 chunk 必须保持独立"
        assert "|-----------|" in table_chunks[0].text


class TestMinChunkConfig:
    def test_default_min_chunk_chars(self):
        assert HeaderAwareSplitter().min_chunk_chars == 300


class TestParentTextOnSubChunks:
    """small-to-big 摄入侧数据 — 切分的子 chunk 携带所属 section 完整文本."""

    def _split_long_section(self, content: str) -> list[Chunk]:
        splitter = HeaderAwareSplitter(
            max_chunk_chars=400, prose_max_chars=400,
            chunk_overlap_chars=0, min_chunk_chars=0,
        )
        return splitter._split_long_section(
            content, "doc", 38, "38.331", "R18",
            "5", "RRC Procedures", "5.3", "Setup", "5 > 5.3 Setup",
            0, "3gpp",
        )

    def test_sub_chunks_carry_section_text(self):
        p1 = _para("Procedure description one. ", 500)
        p2 = _para("Procedure description two. ", 500)
        doc = "# 5.1 Test Section\n\n" + p1 + "\n\n" + p2
        splitter = HeaderAwareSplitter(
            max_chunk_chars=400, prose_max_chars=400,
            chunk_overlap_chars=0, min_chunk_chars=0,
        )
        chunks = splitter.split_document(doc)
        assert len(chunks) >= 2
        for c in chunks:
            assert c.parent_text == doc[:4096]
            assert "Procedure description one" in c.parent_text

    def test_short_section_has_no_parent(self):
        """整段即 section (未切分) → parent_text 留空, 避免自我复制."""
        splitter = HeaderAwareSplitter(min_chunk_chars=0)
        doc = "# 5.1 Short Section\n\nShort content that fits in one chunk."
        chunks = splitter.split_document(doc)
        assert len(chunks) == 1
        assert chunks[0].parent_text == ""
        assert chunks[0].parent_chunk_id == 0

    def test_parent_chunk_id_points_to_section_first_subchunk(self):
        p1 = _para("Procedure description one. ", 500)
        p2 = _para("Procedure description two. ", 500)
        doc = "# 5.1 Test Section\n\n" + p1 + "\n\n" + p2
        splitter = HeaderAwareSplitter(
            max_chunk_chars=400, prose_max_chars=400,
            chunk_overlap_chars=0, min_chunk_chars=0,
        )
        chunks = splitter.split_document(doc)
        assert len(chunks) >= 2
        # 同 section 所有子 chunk 共享首个子 chunk 索引 (重编号后)
        assert all(c.parent_chunk_id == chunks[0].chunk_index for c in chunks)
