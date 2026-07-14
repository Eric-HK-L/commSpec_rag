"""verifier.py 单元测试 — AnswerVerifier 事实溯源验证."""

from src.generator.verifier import AnswerVerifier
from src.retriever.search import RetrievalResult


def _make_source(spec_number="38.413", text="This is a test source text about PDU session establishment.", chunk_id="1"):
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=0.8,
        doc_id="doc1", series=38, spec_number=spec_number,
        release="R18", parent_section_id="8.3.1",
        parent_title="PDU Session", chunk_index=0,
    )


class TestExtractKeyPhrases:

    def test_single_phrase(self):
        text = "This is a long enough phrase for extraction."
        phrases = AnswerVerifier._extract_key_phrases(text, min_length=10)
        assert len(phrases) >= 1
        assert any("long enough phrase" in p for p in phrases)

    def test_min_length_filter(self):
        text = "Short.\nNow a longer phrase that should pass the min length check."
        phrases = AnswerVerifier._extract_key_phrases(text, min_length=20)
        # "Short." 太短，会被过滤
        assert all(len(p) >= 20 for p in phrases)

    def test_max_5_phrases(self):
        lines = [f"This is line number {i:02d} with sufficient length to pass." for i in range(10)]
        text = ".\n".join(lines)
        phrases = AnswerVerifier._extract_key_phrases(text, min_length=5)
        assert len(phrases) <= 5


class TestVerify:

    def test_uncertainty_acknowledgment(self):
        v = AnswerVerifier()
        result = v.verify(
            "根据提供的规范片段无法确定该参数的具体取值范围。",
            [_make_source()],
        )
        assert result["verified"] is True
        assert len(result["warnings"]) == 0

    def test_hallucination_warning(self):
        v = AnswerVerifier()
        result = v.verify(
            "According to TS 99.999, the PDU session setup uses...",
            [_make_source(spec_number="38.413")],
        )
        assert any("TS 99.999" in w for w in result["warnings"])

    def test_valid_reference_no_warning(self):
        v = AnswerVerifier()
        # 答案引用了源文本中的子串以维持覆盖率
        src = _make_source(
            spec_number="38.413",
            text="The NGAP protocol is used for PDU session resource setup on the N2 interface.",
        )
        result = v.verify(
            "According to TS 38.413, the NGAP protocol is used for PDU session setup.",
            [src],
        )
        # 不应有幻觉告警（TS 编号匹配）
        assert not any("TS 38.413" in w for w in result["warnings"])

    def test_coverage_high(self):
        """当答案中有源文本短语时应高覆盖."""
        # 答案需包含源文本的完整子串（_extract_key_phrases 按 min_length=20 提取连续片段）
        # 注意：答案中不能有包含 . 的 spec number（会被 split(".") 多切）
        source_text = "NGAP protocol over N2 interface for PDU session resource setup and release"
        v = AnswerVerifier()
        result = v.verify(
            "The specification defines the NGAP protocol over N2 interface for PDU session resource setup and release. "
            + "This is additional explanatory text added for completeness.",
            [_make_source(text=source_text)],
        )
        assert result["coverage"] >= 0.5

    def test_empty_answer(self):
        v = AnswerVerifier()
        result = v.verify("", [_make_source()])
        assert "coverage" in result

    def test_no_sources(self):
        v = AnswerVerifier()
        result = v.verify("Some answer text.", [])
        # 无源时 total_sentences=1, cited_count=0 → coverage=0
        assert result["coverage"] <= 0.5  # 低覆盖但不报错
