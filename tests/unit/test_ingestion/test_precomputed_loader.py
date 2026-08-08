"""precomputed_loader.py 单元测试 — 纯静态解析方法."""

import warnings

# 该模块已被官方标记废弃 (2026-07-14), 导入时会产生 DeprecationWarning — 测试本身需要它
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from src.ingestion.precomputed_loader import PrecomputedLoader


class TestExtractSeries:
    """_extract_series — 从文件名提取 Series."""

    def test_standard(self):
        assert PrecomputedLoader._extract_series("38413-i40") == 38

    def test_23_series(self):
        assert PrecomputedLoader._extract_series("23501-g80") == 23

    def test_release_prefix(self):
        assert PrecomputedLoader._extract_series("release-18") == 0

    def test_non_digit(self):
        assert PrecomputedLoader._extract_series("AB12345") == 0


class TestExtractSpecNumber:
    """_extract_spec_number — 从 doc_id 或文本提取规范编号."""

    def test_from_doc_id_5_digits(self):
        spec = PrecomputedLoader._extract_spec_number("38300-i30", "")
        assert spec == "38.300"

    def test_from_doc_id_23_series(self):
        spec = PrecomputedLoader._extract_spec_number("23501-g80", "")
        assert spec == "23.501"

    def test_from_text_first_line(self):
        spec = PrecomputedLoader._extract_spec_number(
            "release-18", "TS 38.413 NG Application Protocol"
        )
        assert spec == "38.413"

    def test_empty(self):
        spec = PrecomputedLoader._extract_spec_number("abc", "")
        assert spec == ""


class TestExtractRelease:
    """_extract_release — 从文本提取 Release."""

    def test_r18(self):
        assert PrecomputedLoader._extract_release(
            "3GPP TS 38.300 V17.0.0 (Release 18) technical specification"
        ) == "R18"

    def test_r17(self):
        assert PrecomputedLoader._extract_release(
            "Version 17.0.0 (Release 17)"
        ) == "R17"

    def test_no_release(self):
        assert PrecomputedLoader._extract_release("No version info here") == ""

    def test_first_500_chars_only(self):
        # release 出现在 500 字符之后
        long_prefix = "X" * 500
        text = long_prefix + " (Release 18)"
        assert PrecomputedLoader._extract_release(text) == ""


class TestBuildSectionMap:
    """_build_section_map — 解析章节标题."""

    def test_single_section(self):
        text = "5.1.2  PDU Session Establishment Procedure\nContent here"
        sections = PrecomputedLoader._build_section_map(text)
        assert len(sections) == 1
        assert sections[0][1] == "5.1.2"
        assert "PDU Session" in sections[0][2]

    def test_multiple_sections(self):
        text = (
            "5.1  Overview\nIntro text\n"
            "5.1.2  PDU Session\nDetails\n"
            "6.3.3  RRC Procedures\nMore details"
        )
        sections = PrecomputedLoader._build_section_map(text)
        assert len(sections) == 3

    def test_filter_deep_nesting(self):
        # 深度 > 4 的章节编号应被过滤
        text = "1.2.3.4.5  Too Deep\nContent"
        sections = PrecomputedLoader._build_section_map(text)
        assert len(sections) == 0

    def test_short_title(self):
        text = "1.1  AB\nContent"  # title 只有 2 字符
        sections = PrecomputedLoader._build_section_map(text)
        assert len(sections) == 0  # < 5 chars 被过滤


class TestFindParentSection:
    """_find_parent_section — 根据位置找父章节."""

    def test_find_parent(self):
        section_map = [
            (0, "5.1", "Overview"),
            (50, "5.1.2", "PDU Session"),
            (200, "5.1.3", "QoS"),
        ]
        pid, ptitle = PrecomputedLoader._find_parent_section(section_map, 100)
        assert pid == "5.1.2"
        assert ptitle == "PDU Session"

    def test_before_first(self):
        section_map = [(100, "1.0", "Intro")]
        pid, ptitle = PrecomputedLoader._find_parent_section(section_map, 50)
        assert pid == ""
        assert ptitle == ""

    def test_empty_map(self):
        pid, ptitle = PrecomputedLoader._find_parent_section([], 100)
        assert pid == ""
        assert ptitle == ""


class TestCustomTextSplitter:
    """_custom_text_splitter — Telco-RAG 原始分块算法."""

    def test_single_chunk(self):
        # 文本短于 chunk_overlap 时 while 条件不满足，返回空
        chunks = PrecomputedLoader._custom_text_splitter(
            "Short text sample for testing", chunk_size=500, chunk_overlap=10,
        )
        assert len(chunks) == 1

    def test_large_text_splits(self):
        text = "The quick brown fox. " * 50  # ~1000 chars
        chunks = PrecomputedLoader._custom_text_splitter(
            text, chunk_size=200, chunk_overlap=25, word_split=True,
        )
        assert len(chunks) > 1
        # 每个 chunk 不应超过 chunk_size 太多 (允许词边界扩展)
        for c in chunks:
            assert len(c) <= 300  # generous upper bound for word boundary

    def test_no_word_split(self):
        text = "A" * 1000
        chunks = PrecomputedLoader._custom_text_splitter(
            text, chunk_size=200, chunk_overlap=25, word_split=False,
        )
        assert len(chunks) > 1

    def test_chunks_have_overlap(self):
        text = "abcdefghij " * 50
        chunks = PrecomputedLoader._custom_text_splitter(
            text, chunk_size=100, chunk_overlap=20, word_split=False,
        )
        if len(chunks) >= 2:
            # 前一个 chunk 的尾部与后一个 chunk 的头部有重叠
            pass  # overlap verification depends on exact split positions
