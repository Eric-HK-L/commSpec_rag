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
    retrieved_specs: list[str]          # 实际检索到的规范编号 (重排后最终列表)
    relevant_ranks: list[int]           # 相关结果在检索列表中的排名 (1-based)
    recall_at_k: dict[int, float] = field(default_factory=dict)
    section_recall_at_k: dict[int, float] = field(default_factory=dict)  # 章节级召回
    reciprocal_rank: float = 0.0
    ndcg_at_10: float = 0.0
    initial_retrieved_specs: list[str] = field(default_factory=list)  # 精排前候选池
    initial_recall_at_k: dict[int, float] = field(default_factory=dict)  # 初检召回


@dataclass
class EvalReport:
    """批量评测报告."""
    total: int
    recall_at_5: float
    recall_at_10: float
    recall_at_20: float
    mrr: float                           # Mean Reciprocal Rank
    ndcg_at_10: float
    initial_recall_at_5: float = 0.0     # 精排前候选池的 Recall@5 (初检召回)
    section_recall_at_5: float = 0.0     # 章节级 Recall@5
    section_recall_at_10: float = 0.0    # 章节级 Recall@10
    by_difficulty: dict[str, dict[str, float]] = field(default_factory=dict)
    by_multi_hop: dict[bool, dict[str, float]] = field(default_factory=dict)


def compute_relevant_ranks(retrieved_specs: list[str], expected_specs: list[str]) -> list[int]:
    """计算预期规范在检索结果中的首次排名 (1-based, 去重).

    每个 expected_spec 只记录第一次命中的排名.
    """
    expected_set = {s.lower() for s in expected_specs}
    seen: set[str] = set()
    ranks: list[int] = []
    for rank, spec in enumerate(retrieved_specs, start=1):
        key = spec.lower()
        if key in expected_set and key not in seen:
            ranks.append(rank)
            seen.add(key)
    return ranks


def recall_at_k(retrieved_specs: list[str], expected_specs: list[str], k: int) -> float:
    """Recall@K: Top-K 结果中命中的唯一预期规范占比 (去重)."""
    if not expected_specs:
        return 1.0
    top_k = retrieved_specs[:k]
    expected_set = {s.lower() for s in expected_specs}
    unique_hits = {s.lower() for s in top_k if s.lower() in expected_set}
    return len(unique_hits) / len(expected_set)


def section_recall_at_k(
    retrieved_pairs: list[tuple[str, str]],
    expected_specs: list[str],
    expected_sections: list[str],
    k: int,
) -> float:
    """章节级 Recall@K: 命中需 spec 匹配 expected_specs 且 section 前缀匹配 expected_sections.

    retrieved_pairs: [(spec_number, section_number), ...]. section 前缀匹配:
    expected "6.3.3" 命中 retrieved "6.3.3" / "6.3.3.1", 不命中 "6.3.2".
    expected_sections 为空时退化为 spec 级判定。
    """
    if not expected_specs:
        return 1.0
    exp_specs = {s.lower() for s in expected_specs}
    exp_sections = [s.lower().lstrip("§") for s in expected_sections if s and s.strip()]
    matched_specs: set[str] = set()
    for spec, section in retrieved_pairs[:k]:
        spec_l = spec.lower()
        if spec_l not in exp_specs:
            continue
        sec = (section or "").lower().lstrip("§")
        if not exp_sections or any(sec.startswith(es) for es in exp_sections):
            matched_specs.add(spec_l)
    return len(matched_specs) / len(exp_specs)


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


def evaluate_one(
    sample: EvalSample,
    retrieved_specs: list[str],
    initial_specs: list[str] | None = None,
    retrieved_pairs: list[tuple[str, str]] | None = None,
) -> EvalResult:
    """对单条样本计算全部指标.

    Args:
        sample: 评测样本
        retrieved_specs: 重排后最终检索到的规范编号
        initial_specs: 精排前候选池的规范编号 (初检召回; None 则不计算)
        retrieved_pairs: [(spec, section), ...] 用于章节级召回 (None 则不计算)
    """
    ranks = compute_relevant_ranks(retrieved_specs, sample.expected_specs)
    initial = list(initial_specs or [])
    pairs = list(retrieved_pairs or [])
    return EvalResult(
        sample=sample,
        retrieved_specs=retrieved_specs,
        relevant_ranks=ranks,
        recall_at_k={
            5: recall_at_k(retrieved_specs, sample.expected_specs, 5),
            10: recall_at_k(retrieved_specs, sample.expected_specs, 10),
            20: recall_at_k(retrieved_specs, sample.expected_specs, 20),
        },
        section_recall_at_k={
            5: section_recall_at_k(pairs, sample.expected_specs, sample.expected_sections, 5),
            10: section_recall_at_k(pairs, sample.expected_specs, sample.expected_sections, 10),
            20: section_recall_at_k(pairs, sample.expected_specs, sample.expected_sections, 20),
        },
        reciprocal_rank=reciprocal_rank(ranks),
        ndcg_at_10=ndcg_at_k(retrieved_specs, sample.expected_specs, 10),
        initial_retrieved_specs=initial,
        initial_recall_at_k={
            5: recall_at_k(initial, sample.expected_specs, 5),
            10: recall_at_k(initial, sample.expected_specs, 10),
            20: recall_at_k(initial, sample.expected_specs, 20),
        },
    )


def evaluate_batch(results: list[EvalResult]) -> EvalReport:
    """聚合批量评测结果."""
    n = len(results)
    if n == 0:
        return EvalReport(total=0, recall_at_5=0, recall_at_10=0, recall_at_20=0, mrr=0, ndcg_at_10=0)

    r5 = sum(r.recall_at_k.get(5, 0) for r in results) / n
    r10 = sum(r.recall_at_k.get(10, 0) for r in results) / n
    r20 = sum(r.recall_at_k.get(20, 0) for r in results) / n
    sr5 = sum(r.section_recall_at_k.get(5, 0) for r in results) / n
    sr10 = sum(r.section_recall_at_k.get(10, 0) for r in results) / n
    mrr = sum(r.reciprocal_rank for r in results) / n
    ndcg10 = sum(r.ndcg_at_10 for r in results) / n
    initial_r5 = sum(r.initial_recall_at_k.get(5, 0) for r in results) / n

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
            "initial_recall@5": sum(r.initial_recall_at_k.get(5, 0) for r in subset) / sn,
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
            "initial_recall@5": sum(r.initial_recall_at_k.get(5, 0) for r in subset) / sn,
        }

    return EvalReport(
        total=n,
        recall_at_5=r5,
        recall_at_10=r10,
        recall_at_20=r20,
        section_recall_at_5=sr5,
        section_recall_at_10=sr10,
        mrr=mrr,
        ndcg_at_10=ndcg10,
        initial_recall_at_5=initial_r5,
        by_difficulty=by_diff,
        by_multi_hop=by_hop,
    )
