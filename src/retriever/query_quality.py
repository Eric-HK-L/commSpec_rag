"""查询质量评估 — 检索结果多维度评分 + 低质量自动触发策略.

对每次检索的结果计算密度、多样性、覆盖率、置信度四个维度的分数，
对低质量查询自动触发改写、扩展或建议策略。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .search import RetrievalResult

logger = logging.getLogger(__name__)


# ── 阈值配置 ──

@dataclass
class QualityThresholds:
    """质量评估阈值."""
    density_min: float = 0.5       # 平均相似度下限
    diversity_min: float = 0.3     # spec 多样性下限
    coverage_min: int = 3          # 最少章节层级数
    confidence_min: float = 1.5    # Top-1 / Top-K 平均值比 (越高越集中)


# 默认阈值实例
DEFAULT_THRESHOLDS = QualityThresholds()


@dataclass
class QualityScore:
    """检索结果的多维度质量评分."""
    density: float = 0.0           # 平均相似度 (0-1)
    diversity: float = 0.0         # spec_number 去重数 / Top-K
    coverage: int = 0              # 结果覆盖的不同章节层级数
    confidence: float = 0.0        # Top-1 / Top-K 平均值

    @property
    def is_low_density(self) -> bool:
        return self.density < DEFAULT_THRESHOLDS.density_min

    @property
    def is_low_diversity(self) -> bool:
        return self.diversity < DEFAULT_THRESHOLDS.diversity_min

    @property
    def is_low_coverage(self) -> bool:
        return self.coverage < DEFAULT_THRESHOLDS.coverage_min

    @property
    def overall_ok(self) -> bool:
        """综合判断: 是否有严重质量问题."""
        severe = sum([
            self.is_low_density,
            self.is_low_diversity,
            self.is_low_coverage,
        ])
        return severe <= 1  # 至多一项不达标仍认为可接受


@dataclass
class QualityAction:
    """基于质量评分的推荐动作."""
    should_rewrite: bool = False        # 触发查询改写
    should_expand: bool = False         # 扩展检索范围
    should_suggest: bool = False        # 建议用户细化查询
    reason: str = ""


def evaluate_quality(results: list[RetrievalResult]) -> QualityScore:
    """对检索结果计算多维度质量评分.

    Args:
        results: 检索结果列表 (至少 1 条).

    Returns:
        质量评分对象.
    """
    if not results:
        return QualityScore()

    n = len(results)

    # 1. 密度: Top-K 平均相似度
    scores = [r.score for r in results]
    density = sum(scores) / n

    # 2. 多样性: spec_number 去重比例
    unique_specs = len({r.spec_number for r in results if r.spec_number})
    diversity = unique_specs / n if n > 0 else 0.0

    # 3. 覆盖率: 不同 parent_section_id 的 sub-clause 层级数
    section_depths: set[int] = set()
    for r in results:
        if r.parent_section_id:
            # 计算章节深度: "6.3.3.1" → 4 层
            depth = len(r.parent_section_id.split("."))
            section_depths.add(depth)
    coverage = len(section_depths)

    # 4. 置信度: Top-1 / 平均 (识别"一片混沌"查询)
    avg_score = density
    confidence = scores[0] / avg_score if avg_score > 0 else 0.0

    return QualityScore(
        density=density,
        diversity=diversity,
        coverage=coverage,
        confidence=confidence,
    )


def diagnose_quality(score: QualityScore, results_count: int) -> QualityAction:
    """基于质量评分生成推荐动作.

    Args:
        score: 质量评分.
        results_count: 检索结果总数 (用于辅助判断).

    Returns:
        推荐动作.
    """
    reasons: list[str] = []
    should_rewrite = False
    should_expand = False
    should_suggest = False

    # 密度低 → 查询改写
    if score.is_low_density:
        should_rewrite = True
        reasons.append(f"平均相似度偏低 ({score.density:.3f} < {DEFAULT_THRESHOLDS.density_min})")

    # 多样性低 → 扩展检索范围
    if score.is_low_diversity:
        should_expand = True
        reasons.append(f"规范来源过于集中 (多样性 {score.diversity:.2f})")

    # 覆盖率低 → 建议细化
    if score.is_low_coverage and results_count >= 3:
        should_suggest = True
        reasons.append(f"章节覆盖层数不足 ({score.coverage} < {DEFAULT_THRESHOLDS.coverage_min})")

    reason = "; ".join(reasons) if reasons else "检索质量正常"

    return QualityAction(
        should_rewrite=should_rewrite,
        should_expand=should_expand,
        should_suggest=should_suggest,
        reason=reason,
    )


def should_auto_rewrite(score: QualityScore) -> bool:
    """判断是否应触发自动查询改写."""
    return score.is_low_density


def compute_retrieval_noise_ratio(results: list[RetrievalResult], threshold: float = 0.4) -> float:
    """计算检索噪声比: 低于阈值的得分在 Top-K 中的占比.

    噪声比高 → 检索结果与查询相关性差.
    """
    if not results:
        return 1.0
    noise_count = sum(1 for r in results if r.score < threshold)
    return noise_count / len(results)


def _percentile(values: list[float], percentile: float) -> float:
    """线性插值分位数 (等价 numpy.percentile 默认 method='linear')."""
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * percentile
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def filter_noise(
    results: list[RetrievalResult],
    min_score: float | None = None,
    percentile: float = 0.25,
) -> list[RetrievalResult]:
    """过滤低分噪声结果 — 相对/分位数策略.

    分数标定背景: 主结果经 planner._rerank 归一化到 [0,1]; 补充结果
    (multi_hop / graph_expand / cross_ref) 带 RRF 尺度分数 (≈0.01-0.03) 或
    rank-based 分数. 旧实现的绝对阈值 (min_score=0.3) 会把 RRF 尺度补充结果全部误杀.

    策略:
    - 补充结果通道 (_source_tag 非空): score > 0 即保留 — 其分数已由上游归一化
      到与主结果可比的范围, 不再参与主结果的分位数比较.
    - 主结果通道: 以主结果分数分布的第 percentile 百分位为阈值, 低于阈值的视为噪声.
    - 显式契约: 全部被过滤时返回空列表 (不再静默降级为原始结果), 由调用方显式处理.

    向后兼容: 显式传入 min_score 时走旧绝对阈值路径 (至少保留一条).
    """
    if min_score is not None:
        filtered = [r for r in results if r.score >= min_score]
        return filtered or results

    mains = [r for r in results if not getattr(r, "_source_tag", None)]
    threshold = _percentile([r.score for r in mains], percentile) if mains else 0.0

    kept = [
        r for r in results
        if (getattr(r, "_source_tag", None) and r.score > 0)
        or (not getattr(r, "_source_tag", None) and r.score >= threshold)
    ]
    if len(kept) < len(results):
        logger.debug(
            "噪声过滤 (分位数 %.0f%%): %d/%d 条结果被移除 (threshold=%.3f)",
            percentile * 100, len(results) - len(kept), len(results), threshold,
        )
    return kept
