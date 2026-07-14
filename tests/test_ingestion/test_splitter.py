"""splitter.py 单元测试 — 原子块保护与章节解析（纯函数）."""

from src.ingestion.splitter import (
    GRID_TABLE_BOUNDARY,
    HEADER_RE,
    MATH_BLOCK_DELIM,
    SECTION_NUM_RE,
    HeaderAwareSplitter,
)


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
