"""多跳检索 — 迭代 Agent 模式，自动发现并补充检索缺口.

当单次检索无法覆盖跨规范查询（如 "QoS Flow → DRB binding"）时，
通过 LLM 缺口分析生成子查询 → 二次检索 → 合并结果，实现语义链式检索。

流程:
  原始查询 → 第1轮检索 → LLM 缺口分析 → 生成子查询 → 并行二次检索
  → 合并去重 → (可选)第3轮 → 返回完整上下文
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .search import RetrievalResult

logger = logging.getLogger(__name__)

# LLM 配置
MAX_ROUNDS = 3          # 最大检索轮次
MAX_SUB_QUERIES = 4     # 每轮最多子查询数
SUB_QUERY_TOP_K = 5     # 子查询检索数


# ── LLM 缺口分析提示词 ──

_GAP_ANALYSIS_SYSTEM = """你是一个 3GPP 规范检索专家。分析当前检索结果是否足以回答用户问题。

规则:
1. 逐一检查用户问题的每个子问题是否都有规范片段覆盖
2. 如果已充分覆盖，只输出 "SUFFICIENT"
3. 如果有缺口，生成具体的补充查询（使用 3GPP 术语）
4. 每个子查询必须独立完整，可直接用于检索
5. 避免生成与已检索内容重复的子查询
6. 子查询之间用换行分隔，每行一个，不要编号

输出格式：
SUFFICIENT
（或）
NR PUSCH 的 DMRS 配置方式
NR 载波聚合中 PUCCH group 的定义"""


def _build_gap_analysis_prompt(
    question: str,
    retrieved_summary: str,
) -> list[dict[str, str]]:
    """构建缺口分析消息."""
    return [
        {"role": "system", "content": _GAP_ANALYSIS_SYSTEM},
        {"role": "user", "content": f"""## 用户问题

{question}

## 已检索到的规范片段摘要

{retrieved_summary}

请分析是否需要补充检索。"""},
    ]


def _parse_gap_response(response: str) -> tuple[bool, list[str]]:
    """解析 LLM 缺口分析结果.

    Returns:
        (is_sufficient, sub_queries). is_sufficient=True 表示无需补充.
    """
    text = response.strip()
    if text.upper().startswith("SUFFICIENT"):
        return True, []

    # 按行拆分，过滤空行
    sub_queries = [
        line.strip()
        for line in text.split("\n")
        if line.strip() and len(line.strip()) > 5
    ]
    # 限制数量
    return False, sub_queries[:MAX_SUB_QUERIES]


def _build_chunk_summary(
    chunks: list[RetrievalResult],
    max_chunks: int = 8,
) -> str:
    """将检索结果摘要化为缺口分析的输入.

    每个 chunk: [TS xx.xxx §section] 前 120 字符的文本.
    """
    lines: list[str] = []
    for i, chunk in enumerate(chunks[:max_chunks]):
        spec = chunk.spec_number or "?"
        section = chunk.parent_section_id or "?"
        snippet = chunk.text[:120].replace("\n", " ").strip()
        lines.append(f"[{spec} §{section}] {snippet}")
    return "\n".join(lines)


# ── 结果合并 ──

def _merge_results(
    original: list[RetrievalResult],
    supplement: list[RetrievalResult],
    sub_query_label: str = "",
) -> list[RetrievalResult]:
    """合并原始与补充结果，去重.

    原始结果优先排序，补充结果追加末尾.
    """
    merged = list(original)
    seen_ids = {str(r.chunk_id) for r in merged}  # 统一 str 避免 int/str 混合
    for r in supplement:
        if str(r.chunk_id) not in seen_ids:
            seen_ids.add(str(r.chunk_id))
            r._source_tag = "multi_hop"
            if sub_query_label:
                r._sub_query = sub_query_label
            merged.append(r)
    return merged


# ── 多跳检索器 ──

class MultiHopRetriever:
    """多跳检索器 — 迭代 Agent 模式.

    用法:
        mh = MultiHopRetriever(retriever, llm_client, embed_fn)
        results = mh.search(query, query_embedding)
    """

    def __init__(
        self,
        retriever,       # HybridRetriever
        llm_client,      # LLMClient
        embed_fn,        # (str) -> np.ndarray, 嵌入生成函数
        max_rounds: int = MAX_ROUNDS,
        sub_query_top_k: int = SUB_QUERY_TOP_K,
    ):
        self._retriever = retriever
        self._llm = llm_client
        self._embed = embed_fn
        self._max_rounds = max_rounds
        self._sub_top_k = sub_query_top_k

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
    ) -> list[RetrievalResult]:
        """执行多跳检索.

        Args:
            query: 原始查询文本.
            query_embedding: 查询嵌入向量.

        Returns:
            合并后的检索结果 (原始 + 多跳补充).
        """
        # Round 1: 标准检索
        results = self._retriever.search(query, query_embedding)
        total_results = len(results)
        logger.info(
            "多跳检索 Round 1: %d 条结果 (query=%s)",
            total_results, query[:60],
        )

        # 若结果太少，跳过缺口分析
        if len(results) < 3:
            logger.debug("结果过少，跳过多跳扩展")
            return results

        # 迭代扩展
        for round_idx in range(2, self._max_rounds + 1):
            # 缺口分析
            summary = _build_chunk_summary(results)
            messages = _build_gap_analysis_prompt(query, summary)

            try:
                response = self._llm.chat(messages, temperature=0.3, max_tokens=1024)
            except Exception as e:
                logger.warning("缺口分析 LLM 调用失败: %s", e)
                break

            is_sufficient, sub_queries = _parse_gap_response(response)
            if is_sufficient:
                logger.info("多跳检索 Round %d: LLM 判定信息充分, 停止", round_idx)
                break

            if not sub_queries:
                logger.debug("多跳检索 Round %d: 无有效子查询", round_idx)
                break

            logger.info(
                "多跳检索 Round %d: %d 个子查询 → %s",
                round_idx, len(sub_queries),
                [q[:40] for q in sub_queries],
            )

            # 并行二次检索
            round_supplement: list[RetrievalResult] = []
            for sub_q in sub_queries:
                try:
                    sub_embed = self._embed(sub_q)
                    sub_results = self._retriever.search(sub_q, sub_embed)
                    for r in sub_results:
                        r._source_tag = "multi_hop"
                        r._sub_query = sub_q[:80]
                        r._round = round_idx
                    round_supplement.extend(sub_results)
                except Exception as e:
                    logger.warning("子查询检索失败 [%s]: %s", sub_q[:40], e)

            # 合并
            before = len(results)
            results = _merge_results(results, round_supplement)
            added = len(results) - before
            logger.info(
                "多跳检索 Round %d: +%d 条补充 (总 %d)",
                round_idx, added, len(results),
            )

            if added == 0:
                break  # 新结果无增量，停止

        logger.info("多跳检索完成: 原始 %d → 最终 %d 条", total_results, len(results))
        return results


# ── 便捷函数 ──

def needs_multi_hop(
    results: list[RetrievalResult],
    min_diversity: float = 0.25,
) -> bool:
    """快速判断是否需要多跳检索.

    启发式: spec_number 多样性过低且跨规范 query → 可能需要多跳.
    """
    if len(results) < 3:
        return False
    unique_specs = len({r.spec_number for r in results if r.spec_number})
    diversity = unique_specs / len(results)
    return diversity <= min_diversity
