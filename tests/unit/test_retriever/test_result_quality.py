"""ResultQuality 深模块单元测试 — 4 条规则 + 过滤回填逻辑."""

from __future__ import annotations

from src.retriever.result_quality import filter_low_quality, is_low_quality
from src.retriever.search import RetrievalResult


def _mk(text: str = "", title: str = "", sid: str = "", chunk_id: str = "1",
        score: float = 0.8) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=score, spec_number="38.413",
        parent_section_id=sid, parent_title=title,
    )


# ── 规则 1: 标题关键词 ──


class TestRule1TitleKeywords:
    def test_substring_hit(self):
        assert is_low_quality(_mk(title="3.3 Abbreviations")) is True

    def test_symbols_hit(self):
        assert is_low_quality(_mk(title="Symbols and notation")) is True

    def test_exemption_operation(self):
        assert is_low_quality(_mk(title="Reference operation")) is False

    def test_exemption_procedure(self):
        assert is_low_quality(_mk(title="Handover procedure referencing")) is False

    def test_exemption_function(self):
        assert is_low_quality(_mk(title="Definition function overview")) is False

    def test_exemption_configure(self):
        assert is_low_quality(_mk(title="BWP configuration")) is False

    def test_exemption_establish(self):
        assert is_low_quality(_mk(title="PDU Session establishment")) is False


# ── 规则 2: 结构性前缀 ──


class TestRule2StructuralPrefixes:
    def test_contents_prefix(self):
        assert is_low_quality(_mk(text="#  Contents\n1 Scope")) is True

    def test_foreword_prefix(self):
        assert is_low_quality(_mk(text="#  Foreword\nThis specification...")) is True

    def test_scope_prefix(self):
        assert is_low_quality(_mk(text="# 1 Scope\nThe present document...")) is True

    def test_normal_markdown_heading(self):
        assert is_low_quality(_mk(text="# 6.1.2 PDU Session Setup\nDetailed content...")) is False


# ── 规则 3: 短文本结构关键词 ──


class TestRule3ShortText:
    def test_short_contents_only(self):
        assert is_low_quality(_mk(text="Contents 1 Scope 2 References")) is True

    def test_long_text_with_keyword_passes(self):
        long_text = ("The Contents of the scope section describes detailed "
                     "normative requirements for the NGAP protocol messages "
                     "exchanged over N2 interface between gNB and AMF.")
        assert is_low_quality(_mk(text=long_text)) is False


# ── 规则 4: 顶层章节号 + 标题 ──


class TestRule4LowInfoSections:
    def test_top_level_scope(self):
        assert is_low_quality(_mk(title="Scope", sid="1",
                                  text="The present document defines normative requirements...")) is True

    def test_subsection_not_hit(self):
        assert is_low_quality(_mk(title="Scope", sid="1.2",
                                  text="The present document defines normative requirements...")) is False

    def test_top_level_normal_title(self):
        assert is_low_quality(_mk(title="Introduction of QoS model", sid="2",
                                  text="The QoS model distinguishes flows...")) is False

    def test_no_title_not_hit(self):
        assert is_low_quality(_mk(sid="1", text="Some sufficiently long normative text content here.")) is False


# ── 边界情况 ──


class TestEdgeCases:
    def test_none_text_and_title(self):
        r = RetrievalResult(chunk_id="1", text=None, score=0.5, spec_number="38.413")
        assert is_low_quality(r) is False

    def test_empty_result(self):
        r = RetrievalResult(chunk_id="1", text="", score=0.5, spec_number="38.413")
        assert is_low_quality(r) is False


# ── filter_low_quality 过滤与回填 ──


class TestFilterLowQuality:
    def _results(self):
        good = [
            _mk(text=f"High quality content about topic {i}", title=f"Section 6.{i} Details",
                chunk_id=str(i), score=0.9 - i * 0.1)
            for i in range(3)
        ]
        bad = [
            _mk(text="Abbreviations list", title="Abbreviations", chunk_id="b1", score=0.95),
            _mk(text="#  Contents\n1 Scope", title="", chunk_id="b2", score=0.94),
        ]
        return good, bad

    def test_keeps_only_quality(self):
        good, bad = self._results()
        out = filter_low_quality(bad + good, target_k=3)
        assert len(out) == 3
        assert all(r in good for r in out)

    def test_backfill_when_insufficient(self):
        good, bad = self._results()
        out = filter_low_quality(bad + good[:1], target_k=3)
        # 1 条高质量 + 2 条低质量回填
        assert len(out) == 3
        assert out[0] is good[0]  # 高质量排前

    def test_no_duplicates_in_backfill(self):
        good, bad = self._results()
        out = filter_low_quality(good[:1] + bad, target_k=5)
        assert len(out) == len(set(id(r) for r in out))

    def test_empty_input(self):
        assert filter_low_quality([], target_k=3) == []

    def test_target_k_zero(self):
        """target_k=0 为退化输入: 保持原始逻辑语义 (至少纳入一条后停止)."""
        good, _ = self._results()
        assert filter_low_quality(good, target_k=0) == good[:1]

    def test_preserves_input_order(self):
        good, _ = self._results()
        out = filter_low_quality(list(reversed(good)), target_k=3)
        assert out == list(reversed(good))
