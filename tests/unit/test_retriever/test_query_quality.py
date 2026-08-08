"""query_quality.py 单元测试 — 检索质量评分与诊断."""

import pytest

from src.retriever.query_quality import (
    DEFAULT_THRESHOLDS,
    QualityScore,
    QualityThresholds,
    compute_retrieval_noise_ratio,
    diagnose_quality,
    evaluate_quality,
    filter_noise,
    should_auto_rewrite,
)
from src.retriever.search import RetrievalResult


def _make_result(spec_number="38.300", score=0.9, section_id="5.1.2"):
    return RetrievalResult(
        chunk_id=1, text="test", score=score,
        spec_number=spec_number,
        parent_section_id=section_id,
    )


class TestQualityScore:
    """QualityScore — 多维度质量评分 dataclass."""

    def test_all_good(self):
        qs = QualityScore(density=0.9, diversity=0.8, coverage=5, confidence=2.0)
        assert qs.is_low_density is False
        assert qs.is_low_diversity is False
        assert qs.is_low_coverage is False
        assert qs.overall_ok is True

    def test_three_bad(self):
        qs = QualityScore(density=0.1, diversity=0.1, coverage=1, confidence=0.5)
        assert qs.is_low_density is True
        assert qs.is_low_diversity is True
        assert qs.is_low_coverage is True
        assert qs.overall_ok is False  # 3 severe > 1

    def test_one_bad_still_ok(self):
        qs = QualityScore(density=0.1, diversity=0.8, coverage=5, confidence=2.0)
        assert qs.overall_ok is True  # 1 severe <= 1

    def test_defaults(self):
        qs = QualityScore()
        assert qs.density == 0.0
        assert qs.diversity == 0.0
        assert qs.coverage == 0


class TestQualityThresholds:
    """QualityThresholds — 阈值配置."""

    def test_defaults(self):
        assert DEFAULT_THRESHOLDS.density_min == 0.5
        assert DEFAULT_THRESHOLDS.diversity_min == 0.3
        assert DEFAULT_THRESHOLDS.coverage_min == 3

    def test_custom_thresholds(self):
        qt = QualityThresholds(density_min=0.8, diversity_min=0.5)
        assert qt.density_min == 0.8
        assert qt.diversity_min == 0.5


class TestEvaluateQuality:
    """evaluate_quality — 多维度检索质量评分."""

    def test_empty_results(self):
        qs = evaluate_quality([])
        assert qs.density == 0.0
        assert qs.diversity == 0.0
        assert qs.coverage == 0

    def test_single_result(self):
        results = [_make_result(score=0.95, spec_number="38.300", section_id="5.1")]
        qs = evaluate_quality(results)
        assert qs.density == 0.95
        assert qs.diversity == 1.0  # 1 unique / 1 total
        assert qs.coverage == 1  # "5.1" → 2 parts

    def test_multiple_results(self):
        results = [
            _make_result(score=0.9, spec_number="38.300", section_id="5.1.2"),
            _make_result(score=0.8, spec_number="38.300", section_id="5.1.2"),
            _make_result(score=0.7, spec_number="38.211", section_id="6.3.3.1"),
        ]
        qs = evaluate_quality(results)
        assert qs.density == pytest.approx(0.8)
        assert qs.diversity == pytest.approx(2 / 3)  # 2 unique specs / 3
        assert qs.coverage == 2  # sections with 3 parts and 4 parts

    def test_confidence(self):
        results = [
            _make_result(score=0.9),
            _make_result(score=0.3),
        ]
        qs = evaluate_quality(results)
        # confidence = 0.9 / avg(0.9, 0.3) = 0.9 / 0.6 = 1.5
        assert qs.confidence == pytest.approx(1.5)


class TestDiagnoseQuality:
    """diagnose_quality — 生成推荐动作."""

    def test_all_good(self):
        qs = QualityScore(density=0.9, diversity=0.8, coverage=5, confidence=2.0)
        action = diagnose_quality(qs, results_count=10)
        assert action.should_rewrite is False
        assert action.should_expand is False
        assert action.should_suggest is False
        assert "正常" in action.reason

    def test_low_density_rewrite(self):
        qs = QualityScore(density=0.2, diversity=0.8, coverage=5)
        action = diagnose_quality(qs, results_count=10)
        assert action.should_rewrite is True
        assert "偏低" in action.reason

    def test_low_diversity_expand(self):
        qs = QualityScore(density=0.9, diversity=0.1, coverage=5)
        action = diagnose_quality(qs, results_count=10)
        assert action.should_expand is True
        assert "集中" in action.reason

    def test_low_coverage_suggest(self):
        qs = QualityScore(density=0.9, diversity=0.8, coverage=1)
        action = diagnose_quality(qs, results_count=5)
        assert action.should_suggest is True
        assert "覆盖" in action.reason

    def test_low_coverage_few_results(self):
        # coverage 低但结果数 < 3 → 不触发 suggest
        qs = QualityScore(density=0.9, diversity=0.8, coverage=1)
        action = diagnose_quality(qs, results_count=2)
        assert action.should_suggest is False


class TestShouldAutoRewrite:
    """should_auto_rewrite — 自动改写判断."""

    def test_density_low(self):
        qs = QualityScore(density=0.1)
        assert should_auto_rewrite(qs) is True

    def test_density_ok(self):
        qs = QualityScore(density=0.9)
        assert should_auto_rewrite(qs) is False


class TestNoiseRatio:
    """compute_retrieval_noise_ratio — 噪声比计算."""

    def test_all_clean(self):
        results = [_make_result(score=0.9), _make_result(score=0.8)]
        ratio = compute_retrieval_noise_ratio(results, threshold=0.4)
        assert ratio == 0.0

    def test_half_noisy(self):
        results = [_make_result(score=0.9), _make_result(score=0.2)]
        ratio = compute_retrieval_noise_ratio(results, threshold=0.4)
        assert ratio == 0.5

    def test_empty(self):
        assert compute_retrieval_noise_ratio([]) == 1.0


class TestFilterNoise:
    """filter_noise — 过滤低分噪声."""

    def test_removes_low_score(self):
        results = [_make_result(score=0.9), _make_result(score=0.1)]
        filtered = filter_noise(results, min_score=0.3)
        assert len(filtered) == 1
        assert filtered[0].score == 0.9

    def test_keep_at_least_one(self):
        results = [_make_result(score=0.1)]
        filtered = filter_noise(results, min_score=0.3)
        assert len(filtered) == 1  # 至少保留一条
