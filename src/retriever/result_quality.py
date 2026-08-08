"""结果质量策略 — 低信息密度章节过滤的唯一实现.

深模块: 接口仅 is_low_quality / filter_low_quality 两个函数,
全部判定规则 (标题关键词 / 结构性前缀 / 短文本 / 章节号黑名单)
隐藏在实现内部。管线层与 API 层必须通过本模块统一调用,
禁止各自维护副本。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.retriever.search import RetrievalResult

# ── 规则数据 (实现细节, 不属于接口) ──

# 标题关键词: parent_title 子串命中即视为低质量章节 (如 "3.3 Abbreviations")
_LOW_QUALITY_TITLE_KEYWORDS = ("abbreviation", "definition", "symbol", "reference")

# 标题豁免词: 命中关键词但包含这些词时属于正常内容章节 (如 "Reference operation")
_TITLE_EXEMPTIONS = ("operation", "procedure", "function", "configure", "establish")

# 结构性垃圾文本特征: 文本前缀匹配命中即过滤 (目录/Foreword/Scope/References)
_STRUCTURAL_PREFIXES = (
    "#  Contents", "#  Foreword", "#  1 Scope", "# 1 Scope",
    "#  2 References", "# 2 References",
)

# 短文本结构关键词: 文本过短且仅包含这些内容即过滤
_SHORT_TEXT_KEYWORDS = ("Contents", "Foreword", "Scope")
_SHORT_TEXT_MAX_LEN = 50

# 低信息密度顶层章节号: 协议规范中前几个 section 通常是 Scope/Refs/Defs
_LOW_INFO_SECTION_IDS = ("1", "2", "3")  # 主章节号, 非子章节

# 配合顶层章节号判定的标题关键词
_LOW_INFO_TITLE_PARTS = {"scope", "reference", "definition", "abbreviation", "symbol"}


def is_low_quality(result: "RetrievalResult") -> bool:
    """判断单个检索结果是否为低质量 (低信息密度) chunk."""
    text = (result.text or "").strip()
    title = result.parent_title or ""
    title_lower = title.lower()

    # 规则1: parent_title 子串命中低质量章节名, 且不含豁免词
    if title:
        for kw in _LOW_QUALITY_TITLE_KEYWORDS:
            if kw in title_lower and not any(op in title_lower for op in _TITLE_EXEMPTIONS):
                return True

    # 规则2: 文本前缀命中结构性垃圾 (目录/Foreword/Scope/References)
    if text.startswith(_STRUCTURAL_PREFIXES):
        return True

    # 规则3: 短文本仅包含目录/缩写关键词
    if len(text) < _SHORT_TEXT_MAX_LEN and any(kw in text for kw in _SHORT_TEXT_KEYWORDS):
        return True

    # 规则4: parent_section_id 命中 Scope/Refs/Defs 纯顶层章节号
    sid = result.parent_section_id or ""
    if sid in _LOW_INFO_SECTION_IDS and title:
        if any(p in title_lower for p in _LOW_INFO_TITLE_PARTS):
            return True

    return False


def filter_low_quality(results: list["RetrievalResult"], target_k: int) -> list["RetrievalResult"]:
    """过滤低信息密度章节 (缩写表/符号表/目录/参考文献), 保留 target_k 条高质量结果.

    高质量结果不足 target_k 时, 从低质量结果中补充 (排在末尾)。
    """
    quality: list["RetrievalResult"] = []
    for r in results:
        if not is_low_quality(r):
            quality.append(r)
            if len(quality) >= target_k:
                break
    # 如果过滤后不够, 从低质量中补充 (排在末尾)
    if len(quality) < target_k:
        for r in results:
            if is_low_quality(r) and r not in quality and len(quality) < target_k:
                quality.append(r)
    return quality
