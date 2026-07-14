"""release_monitor.py 单元测试 — 文件名解析与变更报告."""

from pathlib import Path

from src.ingestion.release_monitor import ChangeReport, DocFile, ReleaseMonitor


class TestParseSpecInfo:

    def test_ts_format(self):
        # 注意: regex \s* 不匹配下划线，TS_38 不命中 TS/TR 模式，回退到数字模式
        # Release 模式需要 [Rr]el 前缀，R18 不命中
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("38300-Rel18.docx")
        assert series == 38
        assert spec == "38300"
        assert release == "R18"

    def test_ts_with_space(self):
        # TS 38.413 匹配 TS/TR 模式 (空格分隔)
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("TS 38.413.docx")
        assert series == 38
        assert spec == "38413"
        # "R18" 不匹配 [Rr]el 模式
        assert release == ""

    def test_tr_format(self):
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("TR 23.700 v1.0.0.docx")
        assert series == 23
        assert spec == "23700"
        assert release == ""  # 无 Release

    def test_digit_only_format(self):
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("38300-rel18.docx")
        assert series == 38
        assert spec == "38300"
        assert release == "R18"

    def test_digit_only_with_release_keyword(self):
        # Rel18 命中 release regex
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("23501-Rel18.docx")
        assert series == 23
        assert spec == "23501"
        assert release == "R18"

    def test_no_release(self):
        # TS_38.413 下划线阻断了 regex → 默认值
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("38413.docx")
        assert series == 38
        assert spec == "38413"
        assert release == ""

    def test_five_digit_spec(self):
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("38101-f60.docx")
        assert series == 38
        assert spec == "38101"
        assert release == ""

    def test_unrecognizable(self):
        m = ReleaseMonitor()
        series, spec, release = m._parse_spec_info("README.md")
        assert series == 0
        assert spec == ""
        assert release == ""


class TestChangeReport:

    def test_has_changes_true(self):
        doc = DocFile(path=Path("test.docx"), sha256="abc", size=100, mtime=0)
        report = ChangeReport(new_files=[doc])
        assert report.has_changes is True

    def test_has_changes_false(self):
        report = ChangeReport()
        assert report.has_changes is False

    def test_total_changes(self):
        doc = DocFile(path=Path("a.docx"), sha256="a", size=1, mtime=0)
        report = ChangeReport(
            new_files=[doc],
            modified_files=[doc],
            deleted_keys=["k1", "k2"],
        )
        assert report.total_changes == 4
