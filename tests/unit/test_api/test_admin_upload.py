"""admin_router 文档上传归类逻辑单元测试.

覆盖上传 API 的核心纯函数:
  - _classify_kind: 3GPP / O-RAN / unknown 识别 (文件名 + 内容头双保险)
  - _detect_3gpp_release: Release 检测
  - _spec_and_series: 规范编号与 series 提取
  - _sanitize_filename: 文件名清洗

均为纯函数测试, 不依赖服务器与文件系统.
"""

import pytest

from src.api.rest.admin_router import (
    _classify_kind,
    _detect_3gpp_release,
    _sanitize_filename,
    _spec_and_series,
)


class TestClassifyKind:
    """文档归属识别: 3gpp | oran | unknown."""

    @pytest.mark.parametrize("filename", [
        "38305.md",
        "38300-60.docx",
        "TS_38.300_R18_v17.0.0.docx",
    ])
    def test_3gpp_by_filename(self, filename):
        assert _classify_kind(filename, "") == "3gpp"

    def test_3gpp_by_content_head(self):
        content = "# 3GPP TS 38.300 V18.4.0 (2024-01)"
        assert _classify_kind("unknown_name.pdf", content) == "3gpp"

    def test_3gpp_by_content_tr(self):
        content = "3GPP TR 23.501 V17.0.0 ---"
        assert _classify_kind("raw.md", content) == "3gpp"

    @pytest.mark.parametrize("filename", [
        "O-RAN.WG4.TS.CUS.0-R005-v20.00.md",
        "O-RAN.WG1.CUS.0-R003-v11.00.docx",
    ])
    def test_oran_by_filename(self, filename):
        assert _classify_kind(filename, "") == "oran"

    def test_oran_by_content_head(self):
        content = "O-RAN ALLIANCE Technical Specification"
        assert _classify_kind("file.md", content) == "oran"

    def test_oran_by_content_wg(self):
        content = "O-RAN WORKING GROUP 4"
        assert _classify_kind("file.md", content) == "oran"

    @pytest.mark.parametrize("filename,content", [
        ("meeting_notes.txt", ""),
        ("budget.xlsx", "Quarterly report"),
        ("slides.pdf", "Internal presentation"),
    ])
    def test_unknown(self, filename, content):
        assert _classify_kind(filename, content) == "unknown"


class TestDetect3gppRelease:
    """3GPP Release 检测."""

    def test_from_version_header(self):
        assert _detect_3gpp_release("", "3GPP TS 38.300 V18.4.0") == "R18"

    def test_from_release_parenthesis(self):
        assert _detect_3gpp_release("", "(Release 19)") == "R19"

    def test_fallback_r18(self):
        assert _detect_3gpp_release("38300.docx", "") == "R18"


class TestSpecAndSeries:
    """规范编号与 series 提取."""

    def test_standard_name(self):
        spec, series = _spec_and_series("38300-60.docx")
        assert spec == "38.300"
        assert series == "38"

    def test_simple_number(self):
        spec, series = _spec_and_series("38865.md")
        assert spec == "38.865"
        assert series == "38"

    def test_ts_format(self):
        spec, series = _spec_and_series("TS_23.501_R18_v17.0.0.docx")
        assert spec == "23.501"
        assert series == "23"

    def test_non_3gpp(self):
        spec, series = _spec_and_series("meeting_notes.txt")
        assert spec == ""
        assert series == ""


class TestSanitizeFilename:
    """文件名清洗."""

    def test_removes_path_traversal(self):
        assert _sanitize_filename("../../etc/passwd") == "passwd"

    def test_replaces_dangerous_chars(self):
        out = _sanitize_filename("a/b\\c:d*e?f")
        assert "/" not in out
        assert "\\" not in out
        assert ":" not in out

    def test_empty_fallback(self):
        assert _sanitize_filename("") == "unnamed"
        assert _sanitize_filename(None) == "unnamed"  # type: ignore[arg-type]

    def test_keeps_chinese_and_dots(self):
        """中文与点号保留, 空格替换为下划线 (文件名安全化)."""
        out = _sanitize_filename("测试文档 v1.0.docx")
        assert out == "测试文档_v1.0.docx"
