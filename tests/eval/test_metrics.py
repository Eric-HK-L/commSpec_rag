"""eval/metrics.py 单元测试 — 检索指标纯函数."""

import pytest

from tests.eval.metrics import (
    EvalResult,
    EvalSample,
    compute_relevant_ranks,
    dcg_at_k,
    evaluate_batch,
    evaluate_one,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestComputeRelevantRanks:
    """compute_relevant_ranks — 预期规范在检索列表中的排名."""

    def test_perfect_match(self):
        ranks = compute_relevant_ranks(
            ["38.300", "38.211", "38.331"], ["38.300", "38.211"],
        )
        assert ranks == [1, 2]

    def test_partial_match(self):
        ranks = compute_relevant_ranks(
            ["38.211", "38.213", "38.300"], ["38.300", "38.331"],
        )
        assert ranks == [3]

    def test_no_match(self):
        ranks = compute_relevant_ranks(["38.211", "38.213"], ["38.300"])
        assert ranks == []

    def test_case_insensitive(self):
        ranks = compute_relevant_ranks(["38.300"], ["38.300"])
        assert ranks == [1]

    def test_duplicate_expected(self):
        ranks = compute_relevant_ranks(
            ["38.211", "38.300"], ["38.300", "38.300"],
        )
        assert ranks == [2]


class TestRecallAtK:
    """recall_at_k — Top-K 命中率."""

    def test_full_recall(self):
        assert recall_at_k(
            ["38.300", "38.211", "38.331"], ["38.300", "38.211"], k=5,
        ) == 1.0

    def test_partial_recall(self):
        assert recall_at_k(
            ["38.300", "38.331"], ["38.300", "38.211", "38.213"], k=5,
        ) == 1.0 / 3

    def test_zero_expected(self):
        assert recall_at_k(["38.300"], [], k=5) == 1.0

    def test_k_limits(self):
        # k 限制了检索列表长度
        assert recall_at_k(
            ["a", "b", "c"], ["c"], k=2,
        ) == 0.0  # "c" 在第 3 位, k=2 看不到


class TestReciprocalRank:
    """reciprocal_rank — 倒数排名."""

    def test_first_place(self):
        assert reciprocal_rank([1]) == 1.0

    def test_third_place(self):
        assert reciprocal_rank([3]) == pytest.approx(1.0 / 3)

    def test_no_match(self):
        assert reciprocal_rank([]) == 0.0

    def test_best_rank(self):
        assert reciprocal_rank([5, 2, 7]) == 0.5  # min=2


class TestDCG:
    """dcg_at_k — Discounted Cumulative Gain."""

    def test_all_relevant(self):
        assert dcg_at_k([3.0, 2.0, 1.0], k=3) > 0

    def test_all_irrelevant(self):
        assert dcg_at_k([0.0, 0.0, 0.0], k=3) == 0.0

    def test_k_limits(self):
        # 仅前 k 个计入: k=1 时只看第一个(rel=0), k=3 时包含第三个(rel=3)
        full = dcg_at_k([0.0, 0.0, 3.0], k=3)
        limited = dcg_at_k([0.0, 0.0, 3.0], k=1)
        assert limited < full

    def test_monotonic_rel(self):
        # 相关性越高 DCG 越大
        assert dcg_at_k([2.0], k=1) > dcg_at_k([1.0], k=1)


class TestNDCG:
    """ndcg_at_k — Normalized DCG."""

    def test_perfect_ranking(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=5) == 1.0

    def test_imperfect_ranking(self):
        # ["x", "a", "b"] 相关性=[0, 1, 1], 理想=[1, 1, 0] → NDCG < 1
        score = ndcg_at_k(["x", "a", "b"], ["a", "b"], k=5)
        assert 0.0 < score < 1.0

    def test_all_irrelevant(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=5) == 0.0

    def test_empty_expected(self):
        assert ndcg_at_k(["a", "b"], [], k=5) == 0.0


class TestEvaluateOne:
    """evaluate_one — 单样本全指标计算."""

    def test_returns_result(self):
        sample = EvalSample(
            question="test", expected_specs=["38.300"],
            expected_sections=["5.1"], difficulty="easy",
        )
        result = evaluate_one(sample, ["38.300", "38.211"])
        assert isinstance(result, EvalResult)
        assert result.sample == sample
        assert result.recall_at_k[5] == 1.0
        assert result.reciprocal_rank == 1.0
        assert result.ndcg_at_10 == 1.0

    def test_no_match(self):
        sample = EvalSample(
            question="test", expected_specs=["38.300"],
            expected_sections=["5.1"], difficulty="hard",
        )
        result = evaluate_one(sample, ["38.211", "38.213"])
        assert result.recall_at_k[5] == 0.0
        assert result.reciprocal_rank == 0.0
        assert result.ndcg_at_10 == 0.0


class TestEvaluateBatch:
    """evaluate_batch — 批量聚合."""

    def test_empty(self):
        report = evaluate_batch([])
        assert report.total == 0
        assert report.mrr == 0.0

    def test_single(self):
        sample = EvalSample(
            question="q", expected_specs=["38.300"], expected_sections=[],
        )
        result = evaluate_one(sample, ["38.300"])
        report = evaluate_batch([result])
        assert report.total == 1
        assert report.recall_at_5 == 1.0
        assert report.mrr == 1.0

    def test_multiple(self):
        s1 = EvalSample(
            question="q1", expected_specs=["a"], expected_sections=[],
            difficulty="easy",
        )
        s2 = EvalSample(
            question="q2", expected_specs=["b"], expected_sections=[],
            difficulty="hard",
        )
        r1 = evaluate_one(s1, ["a"])
        r2 = evaluate_one(s2, ["x", "y"])
        report = evaluate_batch([r1, r2])
        assert report.total == 2
        assert report.recall_at_5 == 0.5
        assert report.mrr == 0.5
        assert "easy" in report.by_difficulty
        assert "hard" in report.by_difficulty

    def test_multi_hop_grouping(self):
        s1 = EvalSample(
            question="q1", expected_specs=["a"], expected_sections=[],
            multi_hop=True,
        )
        s2 = EvalSample(
            question="q2", expected_specs=["b"], expected_sections=[],
            multi_hop=False,
        )
        r1 = evaluate_one(s1, ["a"])
        r2 = evaluate_one(s2, ["b"])
        report = evaluate_batch([r1, r2])
        assert True in report.by_multi_hop
        assert False in report.by_multi_hop
