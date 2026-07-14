"""评测指标计算: Recall@K, MRR, NDCG@K, Hallucination Rate."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    """单条评测样本."""
    question: str
    expected_specs: list[str]          # 预期涉及的规范编号, e.g. ["38.300", "38.211"]
    expected_sections: list[str]        # 预期涉及的章节, e.g. ["5.2.1", "6.3"]
    difficulty: str = "medium"          # easy / medium / hard
    multi_hop: bool = False             # 是否需要跨规范检索


@dataclass
class EvalResult:
    """单条评测结果."""
    sample: EvalSample
    retrieved_specs: list[str]          # 实际检索到的规范编号
    relevant_ranks: list[int]           # 相关结果在检索列表中的排名 (1-based)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    reciprocal_rank: float = 0.0
    ndcg_at_10: float = 0.0


@dataclass
class EvalReport:
    """批量评测报告."""
    total: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float                           # Mean Reciprocal Rank
    ndcg_at_10: float
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    by_multi_hop: dict[bool, dict[str, float]] = field(default_factory=dict)


def compute_relevant_ranks(retrieved_specs: list[str], expected_specs: list[str]) -> list[int]:
    """计算预期规范在检索结果中的排名 (1-based).

    返回每个命中 expected_spec 的排名列表，未命中则跳过.
    """
    expected_set = {s.lower() for s in expected_specs}
    ranks: list[int] = []
    for rank, spec in enumerate(retrieved_specs, start=1):
        if spec.lower() in expected_set:
            ranks.append(rank)
    return ranks


def recall_at_k(retrieved_specs: list[str], expected_specs: list[str], k: int) -> float:
    """Recall@K: Top-K 结果中命中的预期规范占比."""
    if not expected_specs:
        return 1.0
    top_k = retrieved_specs[:k]
    expected_set = {s.lower() for s in expected_specs}
    hits = sum(1 for s in top_k if s.lower() in expected_set)
    return hits / len(expected_specs)


def reciprocal_rank(relevant_ranks: list[int]) -> float:
    """Reciprocal Rank: 1 / (第一个相关结果的排名)."""
    if not relevant_ranks:
        return 0.0
    return 1.0 / min(relevant_ranks)


def dcg_at_k(relevance_scores: list[float], k: int) -> float:
    """DCG@K: Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        dcg += (2 ** rel - 1) / math.log2(i + 1)
    return dcg


def ndcg_at_k(retrieved_specs: list[str], expected_specs: list[str], k: int) -> float:
    """NDCG@K: Normalized DCG (二值相关: 命中=1, 未命中=0)."""
    expected_set = {s.lower() for s in expected_specs}
    relevance = [1.0 if s.lower() in expected_set else 0.0 for s in retrieved_specs[:k]]
    dcg = dcg_at_k(relevance, k)
    # IDEAL DCG: 所有相关结果排在最前
    ideal_relevance = sorted(relevance, reverse=True)
    idcg = dcg_at_k(ideal_relevance, k)
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_one(sample: EvalSample, retrieved_specs: list[str]) -> EvalResult:
    """对单条样本计算全部指标."""
    ranks = compute_relevant_ranks(retrieved_specs, sample.expected_specs)
    return EvalResult(
        sample=sample,
        retrieved_specs=retrieved_specs,
        relevant_ranks=ranks,
        recall_at_k={
            5: recall_at_k(retrieved_specs, sample.expected_specs, 5),
            10: recall_at_k(retrieved_specs, sample.expected_specs, 10),
            20: recall_at_k(retrieved_specs, sample.expected_specs, 20),
        },
        reciprocal_rank=reciprocal_rank(ranks),
        ndcg_at_10=ndcg_at_k(retrieved_specs, sample.expected_specs, 10),
    )


def evaluate_batch(results: list[EvalResult]) -> EvalReport:
    """聚合批量评测结果."""
    n = len(results)
    if n == 0:
        return EvalReport(total=0, recall_at_5=0, recall_at_10=0, recall_at_20=0, mrr=0, ndcg_at_10=0)

    r5 = sum(r.recall_at_k.get(5, 0) for r in results) / n
    r10 = sum(r.recall_at_k.get(10, 0) for r in results) / n
    r20 = sum(r.recall_at_k.get(20, 0) for r in results) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    ndcg10 = sum(r.ndcg_at_10 for r in results) / n

    # 按难度分组
    by_diff: dict[str, dict[str, float]] = {}
    for diff in sorted(set(r.sample.difficulty for r in results)):
        subset = [r for r in results if r.sample.difficulty == diff]
        sn = len(subset)
        by_diff[diff] = {
            "count": sn,
            "recall@5": sum(r.recall_at_k.get(5, 0) for r in subset) / sn,
            "recall@10": sum(r.recall_at_k.get(10, 0) for r in subset) / sn,
            "mrr": sum(r.reciprocal_rank for r in subset) / sn,
            "ndcg@10": sum(r.ndcg_at_10 for r in subset) / sn,
        }

    # 按多跳分组
    by_hop: dict[bool, dict[str, float]] = {}
    for hop in (False, True):
        subset = [r for r in results if r.sample.multi_hop == hop]
        if not subset:
            continue
        sn = len(subset)
        by_hop[hop] = {
            "count": sn,
            "recall@5": sum(r.recall_at_k.get(5, 0) for r in subset) / sn,
            "recall@10": sum(r.recall_at_k.get(10, 0) for r in subset) / sn,
            "mrr": sum(r.reciprocal_rank for r in subset) / sn,
            "ndcg@10": sum(r.ndcg_at_10 for r in subset) / sn,
        }

    return EvalReport(
        total=n,
        recall_at_5=r5,
        recall_at_10=r10,
        recall_at_20=r20,
        mrr=mrr,
        ndcg_at_10=ndcg10,
        by_difficulty=by_diff,
        by_multi_hop=by_hop,
    )
