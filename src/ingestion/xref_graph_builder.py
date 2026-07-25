"""离线交叉引用图构建器 — 从 Milvus 读取全部 chunk，构建结构化引用图。

5 阶段构建流程:
  1. 节点生成 + 索引构建
  2. 结构边 (PARENT_CHILD / SIBLING / NEXT_SECTION)
  3. 引用边 (REFERENCES) — 多模式正则提取规范间引用
  4. IE 定义提取 (DEFINES)
  5. adjacency 邻接索引输出

输出: data/processed/xref_graph.json
  含 nodes / edges / adjacency / references_sections
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger(__name__)

# ── 引用提取正则 (5 个模式，按精度从高到低) ──

# 模式 0（最高精度）："Clause X.Y.Z of TS 38.AAA" — spec + clause 联合
_CLAUSE_OF_TS = re.compile(
    r"(?:Clause|clause|section|Section)\s+([\d\.A-Za-z]*)\s+of\s+(?:3GPP\s+)?"
    r"(?P<type>TS|TR)\s+(?P<series>\d{2})\.(?P<number>\d{3})(?:-\d)?",
    re.IGNORECASE,
)

# 模式 0b："Clause X.Y.Z of [N, TS 38.AAA]" — bracket 形态
_CLAUSE_OF_BRACKET = re.compile(
    r"(?:Clause|clause|section|Section)\s+([\d\.A-Za-z]*)\s+of\s+\[(\d+),\s+"
    r"(?:3GPP\s+)?(?P<type>TS|TR)\s+(?P<series>\d{2})\.(?P<number>\d{3})\]",
    re.IGNORECASE,
)

# 模式 1："TS 38.214" — spec 引用（匹配后在 ±80 字上下文找 clause）
_SPEC_REF = re.compile(
    r"(?:3GPP\s+)?(?P<type>TS|TR)\s+(?P<series>\d{2})\.(?P<number>\d{3})(?:-\d)?",
    re.IGNORECASE,
)

# 模式 2："[6, TS 38.214]" — bracket 引用
_BRACKET_REF = re.compile(
    r"\[(\d+),\s+(?:3GPP\s+)?(?P<type>TS|TR)\s+(?P<series>\d{2})\.(?P<number>\d{3})\]",
    re.IGNORECASE,
)

# 模式 3："clause 9.2.5.4" — 同 spec 内引用
_SAME_SPEC_CLAUSE = re.compile(
    r"(?:clause|section|subsection|sub-clause)\s+([\d\.A-Za-z]*)",
    re.IGNORECASE,
)

# 同行检查：是否有 TS 引用（区分 cross-spec）
_TS_REF_INLINE = re.compile(r"TS\s+\d{2}\.\d{3}", re.IGNORECASE)

# ASN.1 IE 定义提取
_ASN1_PATTERN = re.compile(
    r"(\w[\w-]*)\s*::=\s*"
    r"(INTEGER|SEQUENCE|ENUMERATED|BIT\s+STRING|OCTET\s+STRING|CHOICE|BOOLEAN)",
    re.IGNORECASE,
)

# 通用词过滤（IE 名字白名单）
_IE_GENERIC_NAMES = {
    "true", "false", "null", "integer", "boolean", "string",
    "choice", "sequence", "enumerated", "bit", "octet",
}


class XrefGraphBuilder:
    """3GPP 规范交叉引用图构建器。

    从 Milvus collection 读取全部 chunk，构建含节点/边/邻接索引的引用图。
    """

    def __init__(
        self,
        store,
        target_series: set[int] | None = None,
    ):
        """初始化图构建器。

        Args:
            store: MilvusStore 实例（需已连接且 collection 已加载）。
            target_series: 限定处理的规范系列号，默认 {38} (NR)。
                           None 表示处理所有系列。
        """
        self._store = store
        self._target_series = target_series or {38}
        # 内部状态
        self._nodes: list[dict] = []
        self._edges: list[dict] = []
        self._section_index: dict[tuple, list[str]] = defaultdict(list)
        self._doc_to_spec: dict[str, str] = {}
        self._doc_chunks: dict[str, list[dict]] = defaultdict(list)
        self._references_sections: dict[str, str] = {}
        self._ie_names_by_spec: dict[str, set[str]] = defaultdict(set)

    # ═══════════════════════════════════════════════════════════════════
    # 公共接口
    # ═══════════════════════════════════════════════════════════════════

    def build(self, output_path: str | Path) -> Path:
        """执行完整 5 阶段构建，输出 xref_graph.json。

        Returns:
            输出文件路径。
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("=== Xref Graph 构建开始 ===")
        self._reset()

        self._stage1_load_nodes()
        self._stage2_structure_edges()
        self._stage3_reference_edges()
        self._stage4_ie_defines()
        self._stage5_adjacency()

        graph = self._serialize()
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph, f, ensure_ascii=False, indent=2)

        logger.info(
            "Xref Graph 构建完成: %d 节点, %d 边 → %s",
            len(self._nodes), len(self._edges), output_path,
        )
        self._log_stats()
        return output_path

    def _reset(self) -> None:
        """重置内部状态."""
        self._nodes = []
        self._edges = []
        self._edge_keys: set = set()
        self._section_index = defaultdict(list)
        self._doc_to_spec = {}
        self._doc_chunks = defaultdict(list)
        self._references_sections = {}
        self._ie_names_by_spec = defaultdict(set)

    def _serialize(self) -> dict:
        """序列化为 JSON 结构."""
        adjacency = defaultdict(lambda: defaultdict(list))
        for edge in self._edges:
            adjacency[edge["from"]][edge["type"]].append(edge["to"])

        edges_by_type: dict[str, int] = defaultdict(int)
        for e in self._edges:
            edges_by_type[e["type"]] += 1

        return {
            "metadata": {
                "total_nodes": len(self._nodes),
                "total_edges": len(self._edges),
                "edges_by_type": dict(edges_by_type),
            },
            "nodes": self._nodes,
            "edges": self._edges,
            "adjacency": {k: dict(v) for k, v in adjacency.items()},
            "references_sections": self._references_sections,
        }

    def _log_stats(self) -> None:
        """打印构建统计."""
        edges_by_type: dict[str, int] = defaultdict(int)
        for e in self._edges:
            edges_by_type[e["type"]] += 1
        for etype, count in sorted(edges_by_type.items()):
            logger.info("  %s: %d", etype, count)

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 1: 节点生成 + 索引构建
    # ═══════════════════════════════════════════════════════════════════

    def _stage1_load_nodes(self) -> None:
        """从 Milvus 分批读取全部 chunk，创建节点 + 构建索引。"""
        logger.info("阶段 1: 节点生成与索引构建...")
        chunks = self._load_all_chunks()
        for c in chunks:
            spec = c.get("spec_number", "")
            series = int(c.get("series", 0))
            if self._target_series and series not in self._target_series:
                continue

            doc_id = c.get("doc_id", "")
            section_number = c.get("section_number", "")
            chunk_id = c.get("id", "")

            # 节点
            node = {
                "id": str(chunk_id),
                "type": "SPEC_SECTION",
                "doc_id": doc_id,
                "spec": spec,
                "series": series,
                "release": c.get("release", ""),
                "section_number": section_number,
                "section_title": c.get("section_title", ""),
                "section_path": c.get("section_path", ""),
                "parent_section_id": c.get("parent_section_id", ""),
                "parent_title": c.get("parent_title", ""),
                "chunk_index": c.get("chunk_index", 0),
                "text": c.get("text", ""),
            }
            self._nodes.append(node)
            self._doc_chunks[doc_id].append(node)
            self._doc_to_spec[doc_id] = spec

            # section_index: (spec, section_number) → [chunk_id, ...]
            if section_number:
                key = (spec, section_number)
                self._section_index[key].append(str(chunk_id))
                # 同时索引 parent_section_number (用于 PARENT_CHILD 查找)
                parent_sid = c.get("parent_section_id", "")
                if parent_sid:
                    pk = (spec, parent_sid)
                    if pk != key:
                        self._section_index[pk].append(str(chunk_id))

        logger.info("  节点: %d, 索引 entries: %d", len(self._nodes), len(self._section_index))
        self._extract_ie_names()

    def _load_all_chunks(self) -> list[dict]:
        """从 Milvus 游标分页读取全部 chunk 数据。"""
        all_chunks: list[dict] = []
        batch_size = 5000
        last_id = -1

        while True:
            try:
                results = self._store._collection.query(
                    expr=f"id > {last_id}",
                    output_fields=[
                        "id", "doc_id", "series", "spec_number", "release",
                        "section_number", "section_title", "section_path",
                        "parent_section_id", "parent_title", "chunk_index", "text",
                    ],
                    limit=batch_size,
                )
            except Exception as e:
                logger.error("读取 chunk 失败: %s", e)
                break

            if not results:
                break

            all_chunks.extend(results)
            last_id = results[-1]["id"]
            if len(results) < batch_size:
                break

        return all_chunks

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 2: 结构边生成
    # ═══════════════════════════════════════════════════════════════════

    def _stage2_structure_edges(self) -> None:
        """生成 PARENT_CHILD / SIBLING / NEXT_SECTION 结构边。"""
        logger.info("阶段 2: 结构边生成...")
        count = 0
        for doc_id, doc_chunks in self._doc_chunks.items():
            spec = self._doc_to_spec.get(doc_id, "")
            count += self._build_structure_edges_for_doc(doc_id, spec, doc_chunks)
        logger.info("  结构边: %d", count)

    def _build_structure_edges_for_doc(
        self, doc_id: str, spec: str, doc_chunks: list[dict],
    ) -> int:
        """为单个文档生成结构边。"""
        count = 0

        # PARENT_CHILD 边
        for node in doc_chunks:
            parent_sid = node.get("parent_section_id", "")
            if not parent_sid:
                continue
            parent_key = (spec, parent_sid)
            parent_candidates = self._section_index.get(parent_key, [])
            for pc in parent_candidates[:1]:  # 取第一个匹配
                count += self._add_edge(
                    from_id=pc,
                    to_id=node["id"],
                    edge_type="PARENT_CHILD",
                    weight=0.5,
                )

        # SIBLING / NEXT_SECTION 边
        count += self._build_sibling_edges(doc_chunks)
        count += self._build_next_section_edges(doc_chunks)

        return count

    def _build_sibling_edges(self, doc_chunks: list[dict]) -> int:
        """为同父章节的兄弟 chunk 生成 SIBLING 边。"""
        count = 0
        # 按 parent 分组
        by_parent: dict[str, list[dict]] = defaultdict(list)
        for node in doc_chunks:
            parent_sid = node.get("parent_section_id", "") or "__root__"
            by_parent[parent_sid].append(node)

        for parent_sid, siblings in by_parent.items():
            if len(siblings) < 2:
                continue
            # 按 section_number 数字排序
            def _sort_key(n: dict) -> list[int]:
                sn = n.get("section_number", "")
                try:
                    return [int(x) for x in sn.split(".")]
                except (ValueError, AttributeError):
                    return [n.get("chunk_index", 0)]

            siblings.sort(key=_sort_key)
            for i in range(len(siblings) - 1):
                count += self._add_edge(
                    from_id=siblings[i]["id"],
                    to_id=siblings[i + 1]["id"],
                    edge_type="SIBLING",
                    weight=0.3,
                )

        return count

    def _build_next_section_edges(self, doc_chunks: list[dict]) -> int:
        """为顶层连续编号章节生成 NEXT_SECTION 边。"""
        count = 0
        # 只取顶层章节（无 parent_section_id）
        top_level = [
            n for n in doc_chunks
            if not n.get("parent_section_id", "") and n.get("section_number", "")
        ]

        def _top_sort_key(n: dict) -> list[int]:
            try:
                return [int(x) for x in n["section_number"].split(".")]
            except (ValueError, AttributeError):
                return [n.get("chunk_index", 0)]

        top_level.sort(key=_top_sort_key)
        for i in range(len(top_level) - 1):
            n1_parts = _top_sort_key(top_level[i])
            n2_parts = _top_sort_key(top_level[i + 1])
            if (
                len(n1_parts) == 1 and len(n2_parts) == 1
                and n2_parts[0] == n1_parts[0] + 1
            ):
                count += self._add_edge(
                    from_id=top_level[i]["id"],
                    to_id=top_level[i + 1]["id"],
                    edge_type="NEXT_SECTION",
                    weight=0.4,
                )

        return count

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 3: 引用边生成 (REFERENCES) — 核心
    # ═══════════════════════════════════════════════════════════════════

    def _stage3_reference_edges(self) -> None:
        """从 chunk 文本中提取规范引用，生成 REFERENCES 边。"""
        logger.info("阶段 3: 引用边生成...")
        count = 0

        for node in self._nodes:
            text = node.get("text", "")
            if not text:
                continue
            spec = node.get("spec", "")

            # 跳过 §2 References 章节
            if self._is_references_section(node):
                doc_id = node.get("doc_id", "")
                if doc_id not in self._references_sections:
                    self._references_sections[doc_id] = text[:2000]
                continue

            matched_spans: set[tuple[int, int]] = set()

            # 模式 0: "Clause X.Y.Z of TS 38.AAA"
            count += self._extract_pattern0(text, spec, node["id"], matched_spans)

            # 模式 0b: "Clause X.Y.Z of [N, TS 38.AAA]"
            count += self._extract_pattern0b(text, spec, node["id"], matched_spans)

            # 模式 2: "[6, TS 38.214]"
            count += self._extract_pattern2(text, spec, node["id"], matched_spans)

            # 模式 1: "TS 38.214" + 邻近 clause
            count += self._extract_pattern1(text, spec, node["id"], matched_spans)

            # 模式 3: "clause 9.2.5.4" (同 spec)
            count += self._extract_pattern3(text, spec, node["id"], matched_spans)

        logger.info("  引用边: %d", count)

    def _is_references_section(self, node: dict) -> bool:
        """判断是否 §2 References 章节（跳过，避免噪声）。"""
        sid = node.get("section_number", "")
        if sid == "2":
            title = node.get("section_title", "").lower()
            pt = node.get("parent_title", "").lower()
            return "reference" in title or "reference" in pt
        return False

    def _extract_pattern0(
        self, text: str, source_spec: str, source_id: str,
        matched_spans: set,
    ) -> int:
        """模式 0: "Clause X.Y.Z of TS 38.AAA" — 最高精度。"""
        count = 0
        for m in _CLAUSE_OF_TS.finditer(text):
            if self._in_matched(m.start(), m.end(), matched_spans):
                continue
            matched_spans.add((m.start(), m.end()))

            clause = m.group(1).rstrip(".")
            target_spec = f"{m.group('series')}.{m.group('number')}"
            if not self._is_target_series(target_spec):
                continue

            count += self._resolve_and_add_ref_edge(
                source_id, source_spec, target_spec, clause,
                text, m.start(), m.end(),
            )
        return count

    def _extract_pattern0b(
        self, text: str, source_spec: str, source_id: str,
        matched_spans: set,
    ) -> int:
        """模式 0b: "Clause X.Y.Z of [N, TS 38.AAA]" — bracket 形态。"""
        count = 0
        for m in _CLAUSE_OF_BRACKET.finditer(text):
            if self._in_matched(m.start(), m.end(), matched_spans):
                continue
            matched_spans.add((m.start(), m.end()))

            clause = m.group(1).rstrip(".")
            target_spec = f"{m.group('series')}.{m.group('number')}"
            if not self._is_target_series(target_spec):
                continue

            count += self._resolve_and_add_ref_edge(
                source_id, source_spec, target_spec, clause,
                text, m.start(), m.end(),
            )
        return count

    def _extract_pattern2(
        self, text: str, source_spec: str, source_id: str,
        matched_spans: set,
    ) -> int:
        """模式 2: "[6, TS 38.214]" — bracket 引用。"""
        count = 0
        for m in _BRACKET_REF.finditer(text):
            if self._in_matched(m.start(), m.end(), matched_spans):
                continue
            matched_spans.add((m.start(), m.end()))

            target_spec = f"{m.group('series')}.{m.group('number')}"
            if not self._is_target_series(target_spec):
                continue

            # bracket 引用通常没有 clause
            count += self._resolve_and_add_ref_edge(
                source_id, source_spec, target_spec, "",
                text, m.start(), m.end(),
            )
        return count

    def _extract_pattern1(
        self, text: str, source_spec: str, source_id: str,
        matched_spans: set,
    ) -> int:
        """模式 1: "TS 38.214" + 邻近上下文 clause (±80 字)。"""
        count = 0
        for m in _SPEC_REF.finditer(text):
            if self._in_matched(m.start(), m.end(), matched_spans):
                continue
            matched_spans.add((m.start(), m.end()))

            target_spec = f"{m.group('series')}.{m.group('number')}"
            if not self._is_target_series(target_spec):
                continue

            # 在 ±80 字上下文找 clause
            ctx_start = max(0, m.start() - 80)
            ctx_end = min(len(text), m.end() + 80)
            context = text[ctx_start:ctx_end]

            clause = self._find_nearby_clause(context)
            count += self._resolve_and_add_ref_edge(
                source_id, source_spec, target_spec, clause,
                text, m.start(), m.end(),
            )
        return count

    def _extract_pattern3(
        self, text: str, source_spec: str, source_id: str,
        matched_spans: set,
    ) -> int:
        """模式 3: "clause 9.2.5.4" (同 spec 内引用)。"""
        count = 0
        for m in _SAME_SPEC_CLAUSE.finditer(text):
            if self._in_matched(m.start(), m.end(), matched_spans):
                continue
            matched_spans.add((m.start(), m.end()))

            # 同行检查: 是否有 TS 引用（区分 cross-spec）
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            if line_end == -1:
                line_end = len(text)
            line = text[line_start:line_end]
            if _TS_REF_INLINE.search(line):
                continue  # 同行有 TS 引用，留给模式 0/1

            clause = m.group(1).rstrip(".")
            # 同 spec 引用，目标 spec = 源 spec
            count += self._resolve_and_add_ref_edge(
                source_id, source_spec, source_spec, clause,
                text, m.start(), m.end(),
            )
        return count

    def _find_nearby_clause(self, context: str) -> str:
        """在上下文中找最近的 clause 编号。"""
        clause_match = re.search(
            r"(?:clause|section|subsection|sub-clause)\s+([\d\.A-Za-z]*)",
            context, re.IGNORECASE,
        )
        if clause_match:
            return clause_match.group(1).rstrip(".")
        return ""

    def _resolve_and_add_ref_edge(
        self, source_id: str, source_spec: str, target_spec: str,
        clause: str, text: str, start: int, end: int,
    ) -> int:
        """解析 target_spec + clause → chunk_id，添加 REFERENCES 边。"""
        count = 0
        # 边去重 key: (from, to, target_spec, clause)
        added: set[tuple] = set()

        # section → chunk 匹配
        chunk_ids = self._find_chunks_by_section(target_spec, clause)
        is_cross_spec = (source_spec != target_spec)
        weight = 1.0 if is_cross_spec else 0.6

        # 取上下文作为 evidence
        ev_start = max(0, start - 40)
        ev_end = min(len(text), end + 40)
        evidence = text[ev_start:ev_end].strip()

        for cid in chunk_ids:
            key = (source_id, cid, target_spec, clause)
            if key in added:
                continue
            added.add(key)
            count += self._add_edge(
                from_id=source_id,
                to_id=cid,
                edge_type="REFERENCES",
                weight=weight,
                extra={
                    "target_spec": target_spec,
                    "target_clause": clause,
                    "is_cross_spec": is_cross_spec,
                    "evidence": evidence[:200],
                },
            )

        return count

    def _find_chunks_by_section(self, spec: str, clause: str) -> list[str]:
        """按 (spec, clause) 匹配 chunk_id，优先级：精确 > 前缀 > 上级回退。

        匹配策略 (参考 rel18):
        1. 精确匹配: ("38.213", "9.2.3") 直接查
        2. 前缀匹配: "9.2.3" 开头的子章节 (9.2.3.1, 9.2.3.2)
        3. 上级回退: 9.2.3 → 9.2 → 9，逐级向上
        4. 上级前缀匹配: 上级章节前缀开头的所有章节
        """
        if not clause:
            # 无 clause → 尝试匹配 spec 的顶层章节
            all_ids: list[str] = []
            for (s, sn), ids in self._section_index.items():
                if s == spec:
                    all_ids.extend(ids)
            return all_ids[:5]

        # 1. 精确匹配
        exact_key = (spec, clause)
        if exact_key in self._section_index:
            return self._section_index[exact_key]

        # 2. 前缀匹配
        prefix_ids: list[str] = []
        prefix = clause + "."
        for (s, sn), ids in self._section_index.items():
            if s == spec and sn.startswith(prefix):
                prefix_ids.extend(ids)
        if prefix_ids:
            return prefix_ids

        # 3. 上级回退
        parts = clause.split(".")
        for i in range(len(parts) - 1, 0, -1):
            parent_clause = ".".join(parts[:i])
            parent_key = (spec, parent_clause)
            if parent_key in self._section_index:
                return self._section_index[parent_key]

            # 4. 上级前缀匹配
            parent_prefix = parent_clause + "."
            for (s, sn), ids in self._section_index.items():
                if s == spec and sn.startswith(parent_prefix):
                    prefix_ids.extend(ids)
            if prefix_ids:
                return prefix_ids

        return []

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 4: IE 定义提取 (DEFINES)
    # ═══════════════════════════════════════════════════════════════════

    def _extract_ie_names(self) -> None:
        """离线提取每个 spec 的 ASN.1 IE 名字集合。"""
        for node in self._nodes:
            spec = node.get("spec", "")
            if not spec:
                continue
            text = node.get("text", "")
            for m in _ASN1_PATTERN.finditer(text):
                name = m.group(1)
                if len(name) < 3 or name.lower() in _IE_GENERIC_NAMES:
                    continue
                self._ie_names_by_spec[spec].add(name)

    def _stage4_ie_defines(self) -> None:
        """提取 ASN.1 IE 定义，生成 DEFINES 边 + IE 节点。"""
        logger.info("阶段 4: IE 定义提取...")
        ie_count = 0
        edge_count = 0
        ie_nodes_added: set[str] = set()

        for node in self._nodes:
            text = node.get("text", "")
            spec = node.get("spec", "")
            if not text or not spec:
                continue

            for m in _ASN1_PATTERN.finditer(text):
                name = m.group(1)
                asn1_type = m.group(2)
                if len(name) < 3 or name.lower() in _IE_GENERIC_NAMES:
                    continue

                ie_id = f"IE_{name}"

                # IE 节点去重
                if ie_id not in ie_nodes_added:
                    ie_nodes_added.add(ie_id)
                    self._nodes.append({
                        "id": ie_id,
                        "type": "IE",
                        "name": name,
                        "asn1_type": asn1_type,
                        "spec": spec,
                        "section_number": node.get("section_number", ""),
                        "section_title": node.get("section_title", ""),
                        "doc_id": node.get("doc_id", ""),
                    })
                    ie_count += 1

                # DEFINES 边
                edge_count += self._add_edge(
                    from_id=node["id"],
                    to_id=ie_id,
                    edge_type="DEFINES",
                    weight=0.8,
                    extra={"ie_name": name, "asn1_type": asn1_type},
                )

        logger.info("  IE 节点: %d, DEFINES 边: %d", ie_count, edge_count)

    # ═══════════════════════════════════════════════════════════════════
    # 阶段 5: adjacency 邻接索引 (在 _serialize 中构建)
    # ═══════════════════════════════════════════════════════════════════

    def _stage5_adjacency(self) -> None:
        """邻接索引在序列化时懒惰构建，此处仅作占位日志。"""
        logger.info("阶段 5: adjacency 将在序列化时构建")

    # ═══════════════════════════════════════════════════════════════════
    # 辅助方法
    # ═══════════════════════════════════════════════════════════════════

    def _is_target_series(self, spec_number: str) -> bool:
        """检查 spec 是否属于目标系列。"""
        if not self._target_series:
            return True
        try:
            series = int(spec_number.split(".")[0])
            return series in self._target_series
        except (ValueError, IndexError):
            return False

    @staticmethod
    def _in_matched(start: int, end: int, matched: set) -> bool:
        """检查区间是否与已匹配区间重叠。"""
        return any(s <= start <= e or s <= end <= e for s, e in matched)

    def _add_edge(
        self,
        from_id: str,
        to_id: str,
        edge_type: str,
        weight: float,
        extra: dict | None = None,
    ) -> int:
        """添加边（自动去重）。"""
        key = (from_id, to_id, edge_type)
        if not hasattr(self, "_edge_keys"):
            self._edge_keys: set = set()
        if key in self._edge_keys:
            return 0
        self._edge_keys.add(key)

        edge = {
            "from": from_id,
            "to": to_id,
            "type": edge_type,
            "weight": weight,
        }
        if extra:
            edge.update(extra)

        self._edges.append(edge)
        return 1
