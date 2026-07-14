"""3GPP 交叉引用解析 — 识别规范引用 + 二次检索补充上下文.

负责将检索结果中出现的 3GPP 规范引用（如 "TS 38.413 §8.3.1"）
提取为结构化 SpecRef，并通过二次检索将被引用文档的上下文补充进来。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .search import RetrievalResult

logger = logging.getLogger(__name__)

# ── 3GPP 引用正则库 ──

# 规范引用: TS 38.413 / TR 38.901 / 3GPP TS 23.501
_RE_SPEC_REF = re.compile(
    r"(?:3GPP\s+)?(?P<type>TS|TR)\s+(?P<series>\d{2})\.(?P<number>\d{3})",
    re.IGNORECASE,
)

# 章节引用: § 5.2.1 / clause 6.3.2 / subclause 7.1.1 / Annex A
_RE_CLAUSE = re.compile(
    r"(?:§\s*|clause\s+|subclause\s+|sub-clause\s+|section\s+|Annex\s+)"
    r"(?P<clause>[A-Z]?\d+(?:\.\d+)*)",
    re.IGNORECASE,
)

# 表格引用: Table 7.3.1-1 / Table A.2-3
_RE_TABLE = re.compile(
    r"Table\s+(?P<table_ref>[A-Z]?\d+(?:\.\d+)*(?:-[A-Z]?\d+(?:\.\d+)*)?)",
    re.IGNORECASE,
)

# 附图引用: Figure 4.1-1
_RE_FIGURE = re.compile(
    r"Figure\s+(?P<figure_ref>[A-Z]?\d+(?:\.\d+)*(?:-[A-Z]?\d+(?:\.\d+)*)?)",
    re.IGNORECASE,
)

# 整本规范 + 章节联合匹配（「TS 38.413 §8.3.1」或「TS 38.413 clause 8.3.1」）
_RE_SPEC_WITH_CLAUSE = re.compile(
    r"(?:3GPP\s+)?(?P<type>TS|TR)\s+(?P<series>\d{2})\.(?P<number>\d{3})"
    r"(?:\s*(?:§|clause|subclause|section|,)\s*"
    r"(?P<clause>[A-Z]?\d+(?:\.\d+)*))?",
    re.IGNORECASE,
)

# 规范标题中嵌入的引用: 3GPP TS 38.331 V16.5.0
_RE_SPEC_WITH_VERSION = re.compile(
    r"(?:3GPP\s+)?(?P<type>TS|TR)\s+(?P<series>\d{2})\.(?P<number>\d{3})"
    r"\s+V\d+\.\d+\.\d+",
    re.IGNORECASE,
)

# 最大引用递归深度（防止无限二次检索）
MAX_REF_DEPTH = 2


@dataclass
class SpecRef:
    """结构化的 3GPP 规范引用."""
    spec_type: str          # "TS" 或 "TR"
    series: int             # 规范系列号, e.g. 38
    spec_number: str        # 完整规范编号, e.g. "38.413"
    clause: str = ""        # 章节引用, e.g. "8.3.1"
    table: str = ""         # 表格引用, e.g. "7.3.1-1"
    figure: str = ""        # 附图引用
    raw_text: str = ""      # 原始引用文本
    start_pos: int = 0      # 在原文中的起始位置
    end_pos: int = 0        # 在原文中的结束位置

    @property
    def lookup_key(self) -> str:
        """用于去重的唯一键."""
        return f"{self.spec_type}{self.spec_number}:{self.clause}"

    def to_search_query(self) -> str:
        """生成用于二次检索的查询."""
        parts = [f"{self.spec_type} {self.spec_number}"]
        if self.clause:
            parts.append(f"clause {self.clause}")
        if self.table:
            parts.append(f"Table {self.table}")
        return " ".join(parts)


def extract_references(text: str) -> list[SpecRef]:
    """从 chunk 文本中提取所有结构化 3GPP 引用.

    识别以下引用模式:
    - 规范引用: TS 38.413 / TR 38.901
    - 章节引用: §5.2.1 / clause 6.3.2 / subclause 7.1.1
    - 表格引用: Table 7.3.1-1
    - 附图引用: Figure 4.1-1
    - 联合引用: TS 38.413 §8.3.1

    返回按在原文中出现位置排序的 SpecRef 列表.
    """
    refs: list[SpecRef] = []

    # 1. 联合匹配: 规范 + 可选章节
    for m in _RE_SPEC_WITH_CLAUSE.finditer(text):
        refs.append(SpecRef(
            spec_type=m.group("type").upper(),
            series=int(m.group("series")),
            spec_number=f"{m.group('series')}.{m.group('number')}",
            clause=(m.group("clause") or "").strip(),
            raw_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        ))

    # 2. 补充: 独立出现的章节引用（前面无规范编号的）
    spec_positions = {(r.start_pos, r.end_pos) for r in refs}
    for m in _RE_CLAUSE.finditer(text):
        # 跳过已包含在联合匹配中的
        if any(s <= m.start() <= e for s, e in spec_positions):
            continue
        # 只在前 50 字符内出现过 "TS", "TR", "3GPP", "specif" 等词时才认为是 3GPP 相关
        context_start = max(0, m.start() - 50)
        context_before = text[context_start:m.start()].lower()
        if any(kw in context_before for kw in ("ts ", "tr ", "3gpp", "specif", "clause", "section")):
            refs.append(SpecRef(
                spec_type="?",
                series=0,
                spec_number="?",
                clause=m.group("clause") or "",
                raw_text=m.group(0),
                start_pos=m.start(),
                end_pos=m.end(),
            ))

    # 3. 表格引用
    for m in _RE_TABLE.finditer(text):
        refs.append(SpecRef(
            spec_type="?",
            series=0,
            spec_number="?",
            table=m.group("table_ref") or "",
            raw_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        ))

    # 4. 附图引用
    for m in _RE_FIGURE.finditer(text):
        refs.append(SpecRef(
            spec_type="?",
            series=0,
            spec_number="?",
            figure=m.group("figure_ref") or "",
            raw_text=m.group(0),
            start_pos=m.start(),
            end_pos=m.end(),
        ))

    # 按位置排序
    refs.sort(key=lambda r: r.start_pos)
    return refs


def _deduplicate_refs(refs: list[SpecRef]) -> list[SpecRef]:
    """按 lookup_key 去重，保留首次出现."""
    seen: set[str] = set()
    unique: list[SpecRef] = []
    for r in refs:
        key = r.lookup_key
        if key not in seen and r.spec_number != "?":
            seen.add(key)
            unique.append(r)
    return unique


def resolve_cross_refs(
    chunks: list[RetrievalResult],
    retriever=None,
    top_k: int = 5,
    depth: int = 0,
) -> list[RetrievalResult]:
    """从检索结果中提取交叉引用并补充上下文.

    流程:
    1. 扫描每个 chunk 中的引用
    2. 去重（同一 spec+clause 只查一次）
    3. 对每个引用发起二次检索
    4. 追加命中结果到原始 context

    Args:
        chunks: 原始检索结果
        retriever: 检索器实例 (需要实现 search() 方法). 为 None 时跳过二次检索.
        top_k: 每个引用的二次检索 Top-K
        depth: 当前递归深度 (防止无限循环)

    Returns:
        原始结果 + 二次检索补充结果（标注 source='cross_ref'）
    """
    if depth >= MAX_REF_DEPTH:
        return list(chunks)

    # 1. 提取并去重
    all_refs: list[SpecRef] = []
    for chunk in chunks:
        all_refs.extend(extract_references(chunk.text))
    unique_refs = _deduplicate_refs(all_refs)

    if not unique_refs:
        logger.debug("交叉引用解析: 未发现有效引用")
        return list(chunks)

    logger.info(
        "交叉引用解析: 发现 %d 个引用 → %d 个去重后",
        len(all_refs), len(unique_refs),
    )

    # 2. 跳过不可靠引用（无 spec_number）
    concrete_refs = [r for r in unique_refs if r.spec_number != "?"]
    if not concrete_refs:
        return list(chunks)

    # 3. 二次检索
    supplement: list[RetrievalResult] = []
    if retriever is not None:
        for ref in concrete_refs:
            query = ref.to_search_query()
            try:
                results = retriever.search(query, top_k=top_k)
                # 标注来源
                for r in results:
                    r._source_tag = "cross_ref"
                    r._ref_from = ref.raw_text
                supplement.extend(results)
                logger.debug("  二次检索 [%s] → %d 条", ref.lookup_key, len(results))
            except Exception as e:
                logger.warning("二次检索失败 [%s]: %s", ref.lookup_key, e)
    else:
        logger.info("未提供检索器, 跳过二次检索 (仅提取引用)")

    # 4. 合并：原始 → 补充（去重）
    merged = list(chunks)
    seen_ids = {c.chunk_id for c in merged}
    for r in supplement:
        if r.chunk_id not in seen_ids:
            seen_ids.add(r.chunk_id)
            merged.append(r)

    return merged


def get_ref_summary(chunks: list[RetrievalResult]) -> dict[str, list[str]]:
    """生成引用摘要: {spec_number: [clause1, clause2, ...]}.

    用于前端展示「此答案引用了以下规范」面板.
    """
    ref_map: dict[str, set[str]] = {}
    for chunk in chunks:
        refs = extract_references(chunk.text)
        for ref in refs:
            if ref.spec_number == "?":
                continue
            if ref.spec_number not in ref_map:
                ref_map[ref.spec_number] = set()
            if ref.clause:
                ref_map[ref.spec_number].add(ref.clause)

    return {
        spec: sorted(clauses) if clauses else ["整体引用"]
        for spec, clauses in sorted(ref_map.items())
    }
