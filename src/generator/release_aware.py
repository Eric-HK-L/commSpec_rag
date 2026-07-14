"""Release 版本感知 — 检测查询中的 Release 意图 + 元数据过滤 + 对比答案模板.

识别用户查询中的版本约束（如 "R18"、"R17 vs R18"），对检索结果进行
Release 过滤或分版本对比，并生成相应的提示词模板。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .search import RetrievalResult

logger = logging.getLogger(__name__)


class IntentType(Enum):
    """Release 意图类型."""
    NONE = "none"             # 无版本指定
    SINGLE = "single"         # 单版本: "R18", "Release 18"
    COMPARE = "compare"       # 多版本对比: "R17 vs R18"


@dataclass
class ReleaseIntent:
    """从查询中提取的 Release 意图."""
    type: IntentType
    releases: list[str] = field(default_factory=list)     # 如 ["R17", "R18"]
    is_comparative: bool = False                           # 是否对比类查询
    raw_matches: list[str] = field(default_factory=list)   # 原始匹配文本

    @property
    def primary_release(self) -> str | None:
        """主 Release (单版本模式下)."""
        if self.releases:
            return self.releases[0]
        return None


# ── 正则规则库 ──

# 多版本对比: R17 vs R18 / R17 versus R18 / R17 compared to R18
_RE_COMPARE = re.compile(
    r"(?:Release\s+|Rel[.-]?\s*|R\s*)(?P<r1>\d{2})"
    r"\s*(?:vs\.?|versus|compared?\s*(?:to|with)|和|与|对比|相比|比较|versus)\s*"
    r"(?:Release\s+|Rel[.-]?\s*|R\s*)?(?P<r2>\d{2})",
    re.IGNORECASE,
)

# 单版本指定: Release 18 / Rel-18 / R18 / Rel.18
_RE_SINGLE = re.compile(
    r"(?:Release\s+|Rel[.-]?\s*|R\s*)(?P<r>\d{2})"
    r"(?!\s*(?:vs\.?|versus|compared?|和|对比))",  # 排除对比模式
    re.IGNORECASE,
)

# 对比关键词: difference, change, new, 更新, 新增, 变更, 区别, 不同, 变化
_RE_COMPARATIVE = re.compile(
    r"(?:difference|change|new|update|delta|comparison"
    r"|更新|新增|变更|区别|不同|变化|差异|修改|演进|增强|add|remove|deprecat)",
    re.IGNORECASE,
)


def detect_release_intent(query: str) -> ReleaseIntent:
    """从查询文本中检测 Release 版本意图.

    Args:
        query: 用户查询文本.

    Returns:
        结构化的 ReleaseIntent.
    """
    # 1. 检测多版本对比
    compare_m = _RE_COMPARE.search(query)
    if compare_m:
        r1, r2 = compare_m.group("r1"), compare_m.group("r2")
        return ReleaseIntent(
            type=IntentType.COMPARE,
            releases=[f"R{r1}", f"R{r2}"],
            is_comparative=True,
            raw_matches=[compare_m.group(0)],
        )

    # 2. 检测单版本指定
    single_matches = _RE_SINGLE.findall(query)
    if single_matches:
        # 取第一个匹配
        r = single_matches[0]
        is_comp = bool(_RE_COMPARATIVE.search(query))
        return ReleaseIntent(
            type=IntentType.SINGLE,
            releases=[f"R{r}"],
            is_comparative=is_comp,
            raw_matches=[f"R{r}"],
        )

    # 3. 仅有对比关键词但无明确版本号
    if _RE_COMPARATIVE.search(query):
        return ReleaseIntent(
            type=IntentType.NONE,
            is_comparative=True,
        )

    return ReleaseIntent(type=IntentType.NONE)


def filter_by_release(
    results: list[RetrievalResult],
    release: str,
) -> list[RetrievalResult]:
    """按 Release 标签过滤检索结果.

    Args:
        results: 检索结果列表.
        release: 目标 Release, e.g. "R18".

    Returns:
        仅含目标 Release 的结果.
    """
    filtered = [r for r in results if r.release.upper() == release.upper()]
    if len(filtered) < len(results):
        logger.debug(
            "Release 过滤 [%s]: %d/%d 条保留",
            release, len(filtered), len(results),
        )
    return filtered


def group_by_release(
    results: list[RetrievalResult],
) -> dict[str, list[RetrievalResult]]:
    """按 Release 分组检索结果.

    Returns:
        {"R17": [...], "R18": [...], ...}
    """
    groups: dict[str, list[RetrievalResult]] = {}
    for r in results:
        rel = r.release.upper() if r.release else "UNKNOWN"
        groups.setdefault(rel, []).append(r)
    return groups


def build_release_context(
    results: list[RetrievalResult],
    intent: ReleaseIntent,
) -> tuple[list[RetrievalResult], str]:
    """基于 Release 意图构建检索上下文和提示说明.

    Args:
        results: 原始检索结果.
        intent: Release 意图.

    Returns:
        (过滤后的结果列表, 版本提示文本).
    """
    if intent.type == IntentType.SINGLE and intent.primary_release:
        filtered = filter_by_release(results, intent.primary_release)
        if not filtered:
            # 过滤后无结果，回退到全量
            logger.warning("Release [%s] 过滤后无结果, 使用全量", intent.primary_release)
            note = f"（注：未找到 {intent.primary_release} 相关内容，以下为全版本结果）"
            return list(results), note
        note = f"（以下内容限定为 {intent.primary_release}）"
        return filtered, note

    elif intent.type == IntentType.COMPARE:
        groups = group_by_release(results)
        r1, r2 = intent.releases[0], intent.releases[1]
        r1_results = groups.get(r1.upper(), [])
        r2_results = groups.get(r2.upper(), [])

        note_parts = []
        if r1_results:
            note_parts.append(f"{r1}: {len(r1_results)} 条")
        else:
            note_parts.append(f"{r1}: 无结果")
        if r2_results:
            note_parts.append(f"{r2}: {len(r2_results)} 条")
        else:
            note_parts.append(f"{r2}: 无结果")
        note = "版本对比 | " + "，".join(note_parts)

        # 合并两组结果（标注 release 来源）
        merged = []
        for r in r1_results:
            r._compare_release = r1
            merged.append(r)
        for r in r2_results:
            r._compare_release = r2
            merged.append(r)

        # 如果两组都有结果，按 release 分组排列；否则按原顺序
        if r1_results and r2_results:
            return merged, note

        # 某一组无结果时回退
        note += "（注：部分版本无结果）"
        return merged or list(results), note

    else:
        # 无版本意图
        if intent.is_comparative:
            note = "（对比类查询，以下为全版本结果，请关注版本差异）"
        else:
            note = ""
        return list(results), note


# ── 提示词增强 ──

def build_release_note_for_prompt(intent: ReleaseIntent, note: str) -> str:
    """生成嵌入到 RAG 提示词的 Release 说明.

    Args:
        intent: Release 意图.
        note: build_release_context 返回的版本提示.

    Returns:
        可追加到 System Prompt 的说明文本.
    """
    if not note:
        return ""

    lines = ["\n## Release 版本说明"]

    if intent.type == IntentType.SINGLE:
        lines.append(f"- 用户指定了版本范围: {intent.primary_release}")
        lines.append("- 以下上下文已限定为该版本的内容")
    elif intent.type == IntentType.COMPARE:
        r1, r2 = intent.releases[0], intent.releases[1]
        lines.append(f"- 用户要求对比 {r1} 与 {r2} 的差异")
        lines.append("- 以下上下文按版本分组标注，请以对比表格或逐项对比的方式回答")
    elif intent.is_comparative:
        lines.append("- 用户意图为对比类查询，但未指定具体版本")
        lines.append("- 请关注上下文中的版本差异信息")

    lines.append(f"- {note}")
    return "\n".join(lines)
