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


class TestVerifyZhAnswer:
    """中文回答溯源检查 — 英文术语匹配 (中文译文无法与英文检索原文字面重合)."""

    def test_zh_answer_with_matching_terms_no_warning(self):
        """回答保留的英文术语都能在检索文本中找到 → 不误报."""
        v = AnswerVerifier()
        src = _make_source(
            spec_number="O-RAN.WG4.TS.CUS.0",
            text="Section Type 5 shall be used to convey UE scheduling information. "
                 "CIBF and DMRS are used for beamforming scenarios.",
        )
        result = v.verify(
            "ORAN Section Type 5 是 C-Plane 协议中的一种消息类型，用于携带 UE 调度信息，"
            "在 CIBF 与 DMRS-BF 场景中使用 [1]。",
            [src],
        )
        assert result["verified"] is True
        assert not any("重合度" in w for w in result["warnings"])
        assert result["coverage"] >= 0.3

    def test_zh_answer_with_fabricated_terms_warning(self):
        """回答中出现检索文本没有的术语 (如编造的缩写) → 术语重合度低告警."""
        v = AnswerVerifier()
        src = _make_source(
            spec_number="O-RAN.WG4.TS.CUS.0",
            text="Section Type 4 is used for slot configuration control messages.",
        )
        result = v.verify(
            "该消息类型用于 CIBF 波束成形场景，同时支持 XYZQ 协议扩展功能 [1]。",
            [src],
        )
        assert any("重合度" in w for w in result["warnings"])

    def test_zh_answer_no_english_terms_skips_overlap(self):
        """回答无英文术语可匹配 → 不因重合度告警, 仅依赖引用/TS 检查."""
        v = AnswerVerifier()
        result = v.verify(
            "该参数用于配置无线资源控制连接。",
            [_make_source()],
        )
        assert result["coverage"] == 1.0
        assert not any("重合度" in w for w in result["warnings"])

    def test_invalid_ref_number_warning(self):
        """引用编号超出检索结果范围且未见于检索文本 → 幻觉告警."""
        v = AnswerVerifier()
        result = v.verify(
            "根据定义该过程需要三个步骤 [99]。",
            [_make_source()],
        )
        assert any("引用编号" in w for w in result["warnings"])

    def test_spec_internal_ref_not_warned(self):
        """规范内部引用编号 (原文中出现过) 不视为幻觉."""
        v = AnswerVerifier()
        src = _make_source(
            text="The procedure is defined as in TS 23.287 [48] for application layer.",
        )
        result = v.verify(
            "该过程参照 TS 23.287 [48] 的定义实现。",
            [src],
        )
        assert not any("引用编号" in w for w in result["warnings"])
