"""cross_ref.py 单元测试 — 引用模式识别."""

from src.retriever.cross_ref import _deduplicate_refs, extract_references


class TestExtractReferences:

    def test_ts_spec_ref(self):
        refs = extract_references("See TS 38.413 for details.")
        assert len(refs) == 1
        assert refs[0].spec_type == "TS"
        assert refs[0].series == 38
        assert refs[0].spec_number.startswith("38.")  # 实际格式为 38.413

    def test_tr_spec_ref(self):
        refs = extract_references("Refer to TR 23.700 for study.")
        assert len(refs) == 1
        assert refs[0].spec_type == "TR"
        assert refs[0].spec_number.startswith("23.")  # 实际格式为 23.700

    def test_ts_with_clause(self):
        refs = extract_references("As defined in TS 38.413 §8.3.1")
        assert len(refs) == 1
        assert refs[0].spec_number.startswith("38.")
        assert refs[0].clause == "8.3.1"

    def test_ts_with_clause_keyword(self):
        refs = extract_references("See TS 23.501 clause 5.6.7")
        assert len(refs) == 1
        assert refs[0].clause == "5.6.7"

    def test_multiple_refs(self):
        text = "TS 38.413 §8.3.1 and TS 23.501 §5.6.7 both apply."
        refs = extract_references(text)
        assert len(refs) >= 2
        specs = {r.spec_number for r in refs}
        assert any(s.startswith("38.") for s in specs)
        assert any(s.startswith("23.") for s in specs)

    def test_no_ref(self):
        refs = extract_references("This is a general description.")
        assert len(refs) == 0

    def test_empty_text(self):
        refs = extract_references("")
        assert len(refs) == 0

    def test_case_insensitive(self):
        refs = extract_references("ts 38.413 and tr 23.700")
        assert len(refs) == 2

    def test_annex_ref(self):
        # 当前只解析 TS 本体，Annex A 不会被解析为 clause
        refs = extract_references("See TS 38.413 Annex A for details.")
        assert len(refs) >= 1
        assert refs[0].spec_number.startswith("38.")


class TestDeduplicateRefs:

    def test_duplicates_removed(self):
        refs = extract_references("TS 38.413 §8.3.1 and TS 38.413 §8.3.1")
        unique = _deduplicate_refs(refs)
        assert len(unique) == 1

    def test_different_clauses_kept(self):
        refs = extract_references("TS 38.413 §8.3.1 and TS 38.413 §8.3.2")
        unique = _deduplicate_refs(refs)
        assert len(unique) == 2

    def test_empty_list(self):
        assert _deduplicate_refs([]) == []
