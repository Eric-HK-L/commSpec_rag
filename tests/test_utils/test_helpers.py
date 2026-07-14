"""helpers.py 单元测试."""

from src.utils.helpers import ensure_dir, extract_spec_number


class TestExtractSpecNumber:

    def test_from_doc_id_5digit(self):
        assert extract_spec_number("38300_title.docx") == "38.300"

    def test_from_doc_id_5digit_prefix(self):
        assert extract_spec_number("23501-Clause-8.docx") == "23.501"

    def test_from_doc_id_no_digits(self):
        assert extract_spec_number("README.md") == ""

    def test_from_text_header(self):
        result = extract_spec_number("", text="TS 38.413 Introduction\n...")
        assert result == "38.413"

    def test_from_text_multiline(self):
        result = extract_spec_number("12345", text="TS 36.211 Physical Channels\n")
        assert result == "12.345"

    def test_from_text_no_match(self):
        result = extract_spec_number("", text="No spec here")
        assert result == ""

    def test_case_insensitive(self):
        result = extract_spec_number("", text="ts 23.501 Introduction")
        assert result == "23.501"

    def test_short_digits(self):
        # 少于 5 位数字 → fallback 到 text
        result = extract_spec_number("38", text="TS 38.413")
        assert result == "38.413"


class TestEnsureDir:

    def test_creates_dir(self, tmp_path):
        new_dir = tmp_path / "sub" / "nested"
        result = ensure_dir(new_dir)
        assert result.exists()
        assert result.is_dir()

    def test_existing_dir(self, tmp_path):
        existing = tmp_path / "already"
        existing.mkdir()
        result = ensure_dir(existing)
        assert result.exists()  # 不应报错
