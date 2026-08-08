"""router.py 单元测试 — 常量与数据结构正确性."""

from src.retriever.router import SERIES_DESCRIPTIONS, SERIES_LABELS


class TestSeriesConstants:

    def test_18_series_labels(self):
        assert len(SERIES_LABELS) == 18
        assert SERIES_LABELS[0] == 21
        assert SERIES_LABELS[-1] == 38

    def test_18_descriptions(self):
        assert len(SERIES_DESCRIPTIONS) == 18
        assert "21 series" in SERIES_DESCRIPTIONS[0]
        assert "38 series" in SERIES_DESCRIPTIONS[-1]

    def test_all_labels_in_range(self):
        assert all(21 <= s <= 38 for s in SERIES_LABELS)

    def test_no_duplicates(self):
        assert len(set(SERIES_LABELS)) == 18
