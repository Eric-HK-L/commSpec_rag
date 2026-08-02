"""MCP Tool 定义与处理器 — 4 个 CommSpec RAG 工具."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ── Tool Schema 定义 ──

TOOL_SCHEMAS = [
    {
        "name": "search_specs",
        "description": "搜索 3GPP 规范文档, 返回最相关的 Top-K 片段 (含规范编号和章节信息).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索查询, 支持自然语言和技术术语",
                },
                "top_k": {
                    "type": "integer",
                    "description": "返回结果数量 (默认 10, 最大 50)",
                    "default": 10,
                },
                "release": {
                    "type": "string",
                    "description": "按 Release 过滤, 如 'R18' (可选)",
                },
                "series": {
                    "type": "string",
                    "description": "按 Series 过滤, 如 '38' (可选)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_spec_content",
        "description": "获取指定 3GPP 规范的章节内容.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "spec_number": {
                    "type": "string",
                    "description": "规范编号, 如 '38.300' 或 '23.501'",
                },
                "section_id": {
                    "type": "string",
                    "description": "章节编号, 如 '6.1.2' (可选, 不填则返回全部 chunks)",
                },
            },
            "required": ["spec_number"],
        },
    },
    {
        "name": "ask_3gpp_expert",
        "description": "基于 3GPP 规范进行 RAG 专家问答, 返回带溯源的答案 + 幻觉验证结果.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "要咨询的 3GPP 技术问题",
                },
                "release": {
                    "type": "string",
                    "description": "按 Release 过滤, 如 'R18' (可选)",
                },
            },
            "required": ["question"],
        },
    },
    {
        "name": "list_releases",
        "description": "列出向量库中所有可用的 3GPP Release 及对应的文档/Series 统计.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# ── Tool Handler ──

class MCPToolHandler:
    """MCP Tool 调用处理器 — 桥接 RAG Pipeline."""

    def __init__(self):
        self._pipeline = None

    @property
    def pipeline(self):
        if self._pipeline is None:
            from src.generator.llm_client import LLMClient
            from src.generator.pipeline import RAGPipeline

            if _global_store is None:
                raise RuntimeError("MCP Server 未初始化向量库连接")
            self._pipeline = RAGPipeline(vector_store=_global_store, llm_client=LLMClient())
        return self._pipeline

    def handle_tool_call(self, name: str, arguments: dict[str, Any]) -> list[dict]:
        """执行 Tool 调用, 返回 MCP content 列表."""
        if name == "search_specs":
            return self._search_specs(arguments)
        elif name == "get_spec_content":
            return self._get_spec_content(arguments)
        elif name == "ask_3gpp_expert":
            return self._ask_expert(arguments)
        elif name == "list_releases":
            return self._list_releases()
        else:
            return [{"type": "text", "text": f"未知工具: {name}"}]

    def _search_specs(self, args: dict) -> list[dict]:
        query = args["query"]
        top_k = min(args.get("top_k", 10), 50)
        release = args.get("release")
        series = args.get("series")

        results = self.pipeline.search(query, top_k=top_k)
        # 客户端过滤
        if release:
            results = [r for r in results if r.release.upper() == release.upper()]
        if series:
            results = [r for r in results if str(r.series) == series]

        if not results:
            return [{"type": "text", "text": "未找到匹配的 3GPP 规范内容."}]

        lines = [f"找到 {len(results)} 条结果:\n"]
        for i, r in enumerate(results[:top_k], 1):
            header = f"TS {r.spec_number}"
            if r.parent_section_id:
                header += f" §{r.parent_section_id}"
            lines.append(
                f"---\n### [{i}] {header} (score: {r.score:.3f})\n{r.text[:500]}"
            )
        return [{"type": "text", "text": "\n".join(lines)}]

    def _get_spec_content(self, args: dict) -> list[dict]:
        spec_number = args["spec_number"]
        section_id = args.get("section_id", "")

        store = self.pipeline._store
        store._ensure_connected()
        if store._collection is None:
            return [{"type": "text", "text": "向量库未加载, 请先连接 Milvus."}]

        # 通过 Milvus query 按 spec_number 过滤 (替代已删除的 FAISS _id_to_meta)
        try:
            expr = f'spec_number == "{spec_number}"'
            if section_id:
                expr += f' and parent_section_id == "{section_id}"'

            raw = store._collection.query(
                expr=expr,
                output_fields=["text", "parent_section_id", "chunk_index", "doc_id"],
                limit=10000,
            )
        except Exception as e:
            logger.error("查询规范 %s 失败: %s", spec_number, e)
            return [{"type": "text", "text": f"查询失败: {e}"}]

        if not raw:
            return [{"type": "text", "text": f"未找到规范 {spec_number} 的内容."}]

        raw.sort(key=lambda r: int(r.get("chunk_index", 0)))

        lines = [f"TS {spec_number} — {len(raw)} chunks:\n"]
        for i, r in enumerate(raw[:20]):
            sec = str(r.get("parent_section_id", ""))
            text = str(r.get("text", ""))
            header = f"§{sec}" if sec else f"chunk {i + 1}"
            lines.append(f"---\n### {header}\n{text[:500]}")
        if len(raw) > 20:
            lines.append(f"\n... 共 {len(raw)} chunks (仅显示前 20)")

        return [{"type": "text", "text": "\n".join(lines)}]

    def _ask_expert(self, args: dict) -> list[dict]:
        question = args["question"]

        response = self.pipeline.ask(question)

        lines = [
            f"### 回答\n{response.answer}\n",
            f"### 验证\n- 已验证: {'✅' if response.verified else '⚠️'}\n"
            f"- 覆盖率: {response.coverage:.0%}\n"
            f"- 警告: {', '.join(response.warnings) if response.warnings else '无'}\n",
            "### 溯源",
        ]
        for i, src in enumerate(response.sources[:5], 1):
            lines.append(
                f"{i}. TS {src.spec_number}"
                + (f" §{src.parent_section_id}" if src.parent_section_id else "")
                + f" (score: {src.score:.3f})\n   {src.text[:200]}"
            )

        return [{"type": "text", "text": "\n".join(lines)}]

    def _list_releases(self) -> list[dict]:
        store = self.pipeline._store
        store._ensure_connected()
        if store._collection is None:
            return [{"type": "text", "text": "向量库未加载, 请先连接 Milvus."}]

        try:
            raw = store._collection.query(
                expr="id >= 0",
                output_fields=["release", "spec_number", "series"],
                limit=300000,
            )
        except Exception as e:
            logger.error("查询 Release 统计失败: %s", e)
            return [{"type": "text", "text": f"查询失败: {e}"}]

        from collections import defaultdict
        releases: dict[str, dict] = defaultdict(lambda: {"docs": set(), "series": set(), "chunks": 0})
        for r in raw:
            rel = str(r.get("release", "unknown"))
            releases[rel]["docs"].add(str(r.get("spec_number", "")))
            releases[rel]["series"].add(r.get("series", 0))
            releases[rel]["chunks"] += 1

        lines = ["可用 3GPP Release:\n"]
        for r, info in sorted(releases.items()):
            lines.append(
                f"- **{r}**: {len(info['docs'])} 规范, "
                f"{len(info['series'])} Series, {info['chunks']} chunks"
            )

        return [{"type": "text", "text": "\n".join(lines)}]


# ── 全局状态 (由 server.py 注入) ──

_global_store = None


def set_global_store(store):
    global _global_store
    _global_store = store
