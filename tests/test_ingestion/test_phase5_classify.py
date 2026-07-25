"""Phase 5 新功能测试 — classify_chunk + _segment_by_atomic_blocks（纯函数）."""

import pytest
from src.ingestion.splitter import (
    AUTHORITATIVE_SPECS,
    DEFINITION_KEYWORDS,
    HeaderAwareSplitter,
    PROCEDURE_KEYWORDS,
    classify_chunk,
    _contains_table,
)


# ════════════════════════════════════════════════════════════════
# classify_chunk — content_type 检测
# ════════════════════════════════════════════════════════════════

class TestClassifyChunkContentType:

    def test_table_detected_as_parameter_table(self):
        """包含 Markdown pipe table 的 chunk 应识别为 parameter_table."""
        text = """
| Parameter |
|-----------|
| A1        |
| A2        |
"""
        result = classify_chunk(text, "38.211", "Physical random-access channel")
        assert result["content_type"] == "parameter_table"

    def test_grid_table_detected(self):
        """包含 grid table (+---+---+) 的 chunk 应识别为 parameter_table."""
        text = """
+--------+----------------+---------------------+
| Format | Sequence length| Subcarrier spacing  |
+--------+----------------+---------------------+
| A1     | 839            | 1.25 kHz            |
+--------+----------------+---------------------+
"""
        result = classify_chunk(text, "38.211", "Physical channels")
        assert result["content_type"] == "parameter_table"

    def test_definition_by_parent_title(self):
        """parent_title 含 'definition' → content_type=definition."""
        text = "The NR physical layer provides data transport services to higher layers."
        result = classify_chunk(text, "38.201", "5.1 General definition and principles")
        assert result["content_type"] == "definition"

    def test_procedure_by_parent_title(self):
        """parent_title 含 'procedure' → content_type=procedure."""
        text = "The UE shall initiate the random access procedure as follows..."
        result = classify_chunk(text, "38.321", "5.1 Random Access procedure")
        assert result["content_type"] == "procedure"

    def test_fallback_to_overview(self):
        """不含表格且标题无关键词 → content_type=overview."""
        text = "This specification defines the requirements for NR UE radio transmission and reception."
        result = classify_chunk(text, "38.101", "1 Scope")
        assert result["content_type"] == "overview"


# ════════════════════════════════════════════════════════════════
# classify_chunk — spec_role 检测
# ════════════════════════════════════════════════════════════════

class TestClassifyChunkSpecRole:

    @pytest.mark.parametrize("spec", sorted(AUTHORITATIVE_SPECS))
    def test_authoritative_specs(self, spec):
        """白名单中的规范 → spec_role=authoritative."""
        result = classify_chunk("text", spec, "Some title")
        assert result["spec_role"] == "authoritative"

    def test_overview_spec_38300(self):
        """38.300 是概览规范 → spec_role=overview."""
        result = classify_chunk("text", "38.300", "Some title")
        assert result["spec_role"] == "overview"

    def test_supporting_spec(self):
        """非白名单/非38.300 → spec_role=supporting."""
        result = classify_chunk("text", "38.101", "Scope")
        assert result["spec_role"] == "supporting"

    def test_non_38_series_supporting(self):
        """36 系列规范 → supporting."""
        result = classify_chunk("text", "36.331", "RRC protocol specification")
        assert result["spec_role"] == "supporting"


# ════════════════════════════════════════════════════════════════
# classify_chunk — topic_domain 推断
# ════════════════════════════════════════════════════════════════

class TestClassifyChunkTopicDomain:

    def test_phy_layer_38_2xx(self):
        result = classify_chunk("text", "38.211", "title")
        assert result["topic_domain"] == "phy_layer"

    def test_phy_layer_38_213(self):
        result = classify_chunk("text", "38.213", "title")
        assert result["topic_domain"] == "phy_layer"

    def test_mac_layer_38_321(self):
        result = classify_chunk("text", "38.321", "title")
        assert result["topic_domain"] == "mac_layer"

    def test_rrc_layer_38_331(self):
        result = classify_chunk("text", "38.331", "title")
        assert result["topic_domain"] == "rrc_layer"

    def test_ran_arch_38_413(self):
        result = classify_chunk("text", "38.413", "title")
        assert result["topic_domain"] == "ran_arch"

    def test_ran_arch_38_423(self):
        result = classify_chunk("text", "38.423", "title")
        assert result["topic_domain"] == "ran_arch"

    def test_unknown_38_series_sub(self):
        """38.1xx → 未知子类 → 空字符串."""
        result = classify_chunk("text", "38.101", "title")
        assert result["topic_domain"] == ""

    def test_non_38_series(self):
        """非 38 系列 — 按 series 映射到对应 domain."""
        # 36 系列 → lte_ran
        result = classify_chunk("text", "36.331", "title")
        assert result["topic_domain"] == "lte_ran"

    def test_non_38_series_core_network(self):
        """23 系列 → core_network."""
        result = classify_chunk("text", "23.501", "title")
        assert result["topic_domain"] == "core_network"

    def test_non_38_series_security(self):
        """33 系列 → security."""
        result = classify_chunk("text", "33.401", "title")
        assert result["topic_domain"] == "security"

    def test_non_38_series_unknown(self):
        """未映射的系列 → 空字符串."""
        result = classify_chunk("text", "55.001", "title")
        assert result["topic_domain"] == ""


# ════════════════════════════════════════════════════════════════
# classify_chunk — 综合（三字段同时验证）
# ════════════════════════════════════════════════════════════════

class TestClassifyChunkIntegration:

    def test_38211_table_authoritative_phy(self):
        """38.211 表格 chunk → parameter_table + authoritative + phy_layer."""
        text = """
| Preamble format |
|-----------------|
| A1              |
"""
        result = classify_chunk(text, "38.211", "Table 6.3.3.1-1: PRACH preamble formats")
        assert result == {
            "content_type": "parameter_table",
            "spec_role": "authoritative",
            "topic_domain": "phy_layer",
        }

    def test_38321_procedure_mac(self):
        """38.321 procedure chunk → procedure + authoritative + mac_layer."""
        text = "The MAC entity shall for each activated Serving Cell..."
        result = classify_chunk(text, "38.321", "5.4.1 Random Access procedure initialization")
        assert result == {
            "content_type": "procedure",
            "spec_role": "authoritative",
            "topic_domain": "mac_layer",
        }

    def test_38101_overview_supporting(self):
        """38.101 overview chunk → overview + supporting + ''."""
        text = "This TS specifies the minimum RF characteristics..."
        result = classify_chunk(text, "38.101", "1 Scope")
        assert result == {
            "content_type": "overview",
            "spec_role": "supporting",
            "topic_domain": "",
        }


# ════════════════════════════════════════════════════════════════
# _segment_by_atomic_blocks — 表格/正文分离
# ════════════════════════════════════════════════════════════════

class TestSegmentByAtomicBlocks:

    @pytest.fixture
    def splitter(self):
        return HeaderAwareSplitter()

    def test_pure_prose(self, splitter):
        text = "This is a paragraph.\n\nThis is another paragraph."
        segments = splitter._segment_by_atomic_blocks(text)
        assert len(segments) == 1
        assert segments[0][0] == "prose"

    def test_pipe_table_separated(self, splitter):
        text = """Some introductory text.

| Parameter |
|-----------|
| A         |
| B         |

Some concluding text."""
        segments = splitter._segment_by_atomic_blocks(text)
        types = [s[0] for s in segments]
        assert types == ["prose", "pipe_table", "prose"]

    def test_grid_table_separated(self, splitter):
        text = """Grid table below:

+--------+-------+
| Param  | Value |
+--------+-------+
| X      | 10    |
+--------+-------+

After the table."""
        segments = splitter._segment_by_atomic_blocks(text)
        types = [s[0] for s in segments]
        assert "grid_table" in types
        assert types[0] == "prose"
        assert types[-1] == "prose"

    def test_prose_only_no_tables(self, splitter):
        text = "Just normal text without any tables or math blocks."
        segments = splitter._segment_by_atomic_blocks(text)
        assert len(segments) == 1
        assert segments[0] == ("prose", text)

    def test_table_content_preserved(self, splitter):
        text = """| A |
|---|
| 1 |
| 2 |"""
        segments = splitter._segment_by_atomic_blocks(text)
        table_seg = segments[0]
        assert table_seg[0] == "pipe_table"
        assert "| A |" in table_seg[1]
        assert "| 1 |" in table_seg[1]


# ════════════════════════════════════════════════════════════════
# 白名单与关键词常量完整性
# ════════════════════════════════════════════════════════════════

class TestConstants:

    def test_authoritative_specs_not_empty(self):
        assert len(AUTHORITATIVE_SPECS) > 0

    def test_authoritative_only_38_series(self):
        """权威规范应都是 38 系列（当前设计）."""
        for spec in AUTHORITATIVE_SPECS:
            assert spec.startswith("38."), f"{spec} 不是 38 系列"

    def test_definition_keywords_contain_expected(self):
        assert "definition" in DEFINITION_KEYWORDS
        assert "overview" in DEFINITION_KEYWORDS

    def test_procedure_keywords_contain_expected(self):
        assert "procedure" in PROCEDURE_KEYWORDS
        assert "procedures" in PROCEDURE_KEYWORDS
