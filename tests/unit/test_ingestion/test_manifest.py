"""manifest.py 单元测试 — 版本解析/比较、清单逻辑（纯函数）."""

from src.ingestion.manifest import (
    IngestionManifest,
    SpecRecord,
    compare_versions,
    parse_3gpp_version,
)


class TestParse3gppVersion:

    def test_i_version(self):
        assert parse_3gpp_version("36322-i10") == "i10"

    def test_f_version(self):
        assert parse_3gpp_version("38101-f60") == "f60"

    def test_id_version(self):
        assert parse_3gpp_version("36101-id0") == "id0"

    def test_no_version(self):
        assert parse_3gpp_version("cover") is None

    def test_no_version_docx(self):
        assert parse_3gpp_version("summary") is None

    def test_dot_separator(self):
        assert parse_3gpp_version("38.413-i20") == "i20"


class TestCompareVersions:

    def test_equal(self):
        assert compare_versions("i10", "i10") == 0

    def test_numeric_gt(self):
        assert compare_versions("i50", "i10") == 1
        assert compare_versions("i10", "i50") == -1

    def test_none_vs_value(self):
        assert compare_versions(None, "i10") == -1
        assert compare_versions("i10", None) == 1

    def test_both_none(self):
        assert compare_versions(None, None) == 0

    def test_same_number_diff_letter(self):
        # "i10a" vs "i10b" — same number, letter sorts
        assert compare_versions("i10a", "i10b") == -1
        assert compare_versions("i10b", "i10a") == 1

    def test_diff_prefix_same_number(self):
        # "i10" vs "f10" — same number (10), sort by letter
        assert compare_versions("f10", "i10") == -1

    def test_dotted_release_version(self):
        # 点号格式发布版本 ("18.4.0") 与字母格式内部版本 ("i30") 混合比较 — 不应抛异常
        # "18.4.0" 数字部分 1840 > "i30" 的 30 → 返回 1
        assert compare_versions("18.4.0", "i30") == 1
        assert compare_versions("i30", "18.4.0") == -1

    def test_dotted_version_comparison(self):
        # 两个点号版本号按数字部分比较
        assert compare_versions("18.4.0", "18.0.0") == 1
        assert compare_versions("18.0.0", "18.1.0") == -1
        assert compare_versions("18.4.0", "18.4.0") == 0


class TestMakeKey:

    def test_standard(self):
        assert IngestionManifest.make_key("38.413", "R18") == "38.413|R18"

    def test_different_release(self):
        k1 = IngestionManifest.make_key("38.413", "R17")
        k2 = IngestionManifest.make_key("38.413", "R18")
        assert k1 != k2


class TestShouldReplace:

    def test_no_existing(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {}
        assert m.should_replace("38.413", "R18", "i10") is False

    def test_newer_version(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {
            "38.413|R18": SpecRecord(
                spec_number="38.413", release="R18",
                latest_version="i10", file_path="", sha256="", chunk_count=0,
                ingested_at="",
            ),
        }
        assert m.should_replace("38.413", "R18", "i50") is True

    def test_older_version(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {
            "38.413|R18": SpecRecord(
                spec_number="38.413", release="R18",
                latest_version="i50", file_path="", sha256="", chunk_count=0,
                ingested_at="",
            ),
        }
        assert m.should_replace("38.413", "R18", "i10") is False


class TestHasSameHash:

    def test_same(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {
            "38.413|R18": SpecRecord(
                spec_number="38.413", release="R18",
                latest_version="i10", file_path="", sha256="abc123", chunk_count=0,
                ingested_at="",
            ),
        }
        assert m.has_same_hash("38.413", "R18", "abc123") is True

    def test_different(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {
            "38.413|R18": SpecRecord(
                spec_number="38.413", release="R18",
                latest_version="i10", file_path="", sha256="abc123", chunk_count=0,
                ingested_at="",
            ),
        }
        assert m.has_same_hash("38.413", "R18", "xyz789") is False

    def test_not_found(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {}
        assert m.has_same_hash("38.413", "R18", "abc") is False


class TestGetOrphanedKeys:

    def test_no_orphans(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {
            "38.413|R18": SpecRecord("38.413", "R18", "i10", "", "", 0, ""),
        }
        existing = {("38.413", "R18")}
        assert m.get_orphaned_keys(existing) == []

    def test_has_orphan(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {
            "38.413|R18": SpecRecord("38.413", "R18", "i10", "", "", 0, ""),
        }
        existing = {("23.501", "R18")}  # 不同 spec
        orphans = m.get_orphaned_keys(existing)
        assert ("38.413", "R18") in orphans

    def test_multiple_orphans(self):
        m = IngestionManifest.__new__(IngestionManifest)
        m._records = {
            "38.413|R18": SpecRecord("38.413", "R18", "i10", "", "", 0, ""),
            "23.501|R18": SpecRecord("23.501", "R18", "i20", "", "", 0, ""),
        }
        existing = set()
        orphans = m.get_orphaned_keys(existing)
        assert len(orphans) == 2
