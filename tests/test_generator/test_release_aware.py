"""release_aware.py 单元测试 — Release 意图检测与结果过滤/分组."""

from src.generator.release_aware import (
    IntentType,
    ReleaseIntent,
    build_release_context,
    build_release_note_for_prompt,
    detect_release_intent,
    filter_by_release,
    group_by_release,
)
from src.retriever.search import RetrievalResult


def _make_result(spec_number="38.300", release="R18", score=0.9):
    return RetrievalResult(
        chunk_id=1, text="test text", score=score,
        spec_number=spec_number, release=release,
        parent_section_id="5.1.2",
    )


class TestDetectReleaseIntent:
    """detect_release_intent — 从查询文本检测 Release 意图."""

    def test_no_release(self):
        intent = detect_release_intent("What is 5QI?")
        assert intent.type == IntentType.NONE
        assert intent.releases == []
        assert intent.is_comparative is False

    def test_single_r18(self):
        intent = detect_release_intent("R18 PDU session setup")
        assert intent.type == IntentType.SINGLE
        assert intent.releases == ["R18"]
        assert intent.primary_release == "R18"

    def test_single_release_18(self):
        intent = detect_release_intent("Release 18 NAS procedures")
        assert intent.type == IntentType.SINGLE
        assert intent.releases == ["R18"]

    def test_single_rel_17(self):
        intent = detect_release_intent("Rel-17 QoS model")
        assert intent.type == IntentType.SINGLE
        assert intent.releases == ["R17"]

    def test_compare_r17_vs_r18(self):
        intent = detect_release_intent("R17 vs R18 PDU session differences")
        assert intent.type == IntentType.COMPARE
        assert intent.releases == ["R17", "R18"]
        assert intent.is_comparative is True

    def test_compare_versus(self):
        intent = detect_release_intent("R16 versus R17 changes")
        assert intent.type == IntentType.COMPARE
        assert intent.releases == ["R16", "R17"]

    def test_compare_chinese(self):
        intent = detect_release_intent("R17 与 R18 的区别")
        assert intent.type == IntentType.COMPARE
        assert intent.releases == ["R17", "R18"]

    def test_comparative_no_release(self):
        intent = detect_release_intent("what changed in 5G QoS?")
        assert intent.type == IntentType.NONE
        assert intent.is_comparative is True

    def test_comparative_keyword_difference(self):
        intent = detect_release_intent("difference between PDU sessions")
        assert intent.type == IntentType.NONE
        assert intent.is_comparative is True


class TestFilterByRelease:
    """filter_by_release — 按 Release 标签过滤."""

    def test_filter_r18(self):
        results = [
            _make_result(release="R18", spec_number="38.300"),
            _make_result(release="R17", spec_number="38.300"),
            _make_result(release="R18", spec_number="38.211"),
        ]
        filtered = filter_by_release(results, "R18")
        assert len(filtered) == 2
        assert all(r.release == "R18" for r in filtered)

    def test_case_insensitive(self):
        results = [_make_result(release="r18")]
        filtered = filter_by_release(results, "R18")
        assert len(filtered) == 1

    def test_no_match(self):
        results = [_make_result(release="R18")]
        filtered = filter_by_release(results, "R17")
        assert len(filtered) == 0


class TestGroupByRelease:
    """group_by_release — 按 Release 分组."""

    def test_two_groups(self):
        results = [
            _make_result(release="R18", spec_number="38.300"),
            _make_result(release="R17", spec_number="38.300"),
            _make_result(release="R18", spec_number="38.211"),
        ]
        groups = group_by_release(results)
        assert "R18" in groups
        assert "R17" in groups
        assert len(groups["R18"]) == 2
        assert len(groups["R17"]) == 1

    def test_unknown_release(self):
        results = [_make_result(release="")]
        groups = group_by_release(results)
        assert "UNKNOWN" in groups

    def test_empty(self):
        groups = group_by_release([])
        assert groups == {}


class TestBuildReleaseContext:
    """build_release_context — 基于意图构建检索上下文."""

    def test_single_release(self):
        results = [
            _make_result(release="R18"),
            _make_result(release="R17"),
        ]
        intent = ReleaseIntent(type=IntentType.SINGLE, releases=["R18"])
        filtered, note = build_release_context(results, intent)
        assert len(filtered) == 1
        assert "R18" in note

    def test_compare(self):
        results = [
            _make_result(release="R18", spec_number="38.300"),
            _make_result(release="R17", spec_number="38.300"),
        ]
        intent = ReleaseIntent(type=IntentType.COMPARE, releases=["R17", "R18"])
        merged, note = build_release_context(results, intent)
        assert len(merged) == 2
        assert "R17" in note
        assert "R18" in note

    def test_none_intent(self):
        results = [_make_result()]
        intent = ReleaseIntent(type=IntentType.NONE)
        filtered, note = build_release_context(results, intent)
        assert len(filtered) == 1
        assert note == ""

    def test_comparative_no_release(self):
        results = [_make_result()]
        intent = ReleaseIntent(type=IntentType.NONE, is_comparative=True)
        filtered, note = build_release_context(results, intent)
        assert "对比" in note


class TestBuildReleaseNote:
    """build_release_note_for_prompt — 提示词增强."""

    def test_empty_note(self):
        intent = ReleaseIntent(type=IntentType.NONE)
        assert build_release_note_for_prompt(intent, "") == ""

    def test_single_release_note(self):
        intent = ReleaseIntent(type=IntentType.SINGLE, releases=["R18"])
        result = build_release_note_for_prompt(intent, "（以下内容限定为 R18）")
        assert "Release 版本说明" in result
        assert "R18" in result
        assert "限定" in result

    def test_compare_note(self):
        intent = ReleaseIntent(type=IntentType.COMPARE, releases=["R17", "R18"])
        result = build_release_note_for_prompt(intent, "版本对比 | R17: 5 条，R18: 3 条")
        assert "对比" in result
        assert "R17" in result
        assert "R18" in result


class TestReleaseIntent:
    """ReleaseIntent dataclass."""

    def test_primary_release(self):
        intent = ReleaseIntent(type=IntentType.SINGLE, releases=["R18"])
        assert intent.primary_release == "R18"

    def test_no_primary(self):
        intent = ReleaseIntent(type=IntentType.NONE)
        assert intent.primary_release is None
