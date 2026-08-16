"""检索规划器 — 深模块: 窄接口 plan()/search() 隐藏全部检索策略实现.

职责: 将用户查询规划为检索上下文 (RetrievalContext), 封装
翻译/扩展 → 混合检索 → 分类分解 → 精排 → 多跳 → 交叉引用 →
质量评估 → Release 感知 → 在线补充 的完整策略链.

调用方 (RAGPipeline) 只看到两个入口:
- plan(): 问答场景的完整检索规划
- search(): 仅检索 (不生成答案) 场景
"""

from __future__ import annotations

import copy
import hashlib
import logging
import re
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from cachetools import TTLCache

from src.config import settings
from src.generator.i18n import detect_language, translate_to_english
from src.generator.llm_client import LLMClient
from src.generator.prompt import build_query_expansion_prompt, is_taxonomy_query
from src.generator.release_aware import (
    build_release_context,
    build_release_note_for_prompt,
    detect_release_intent,
)
from src.retriever.cross_ref import _deduplicate_refs, extract_references
from src.retriever.glossary import expand_abbreviations
from src.retriever.graph_expander import GraphExpander
from src.retriever.milvus_store import _escape_milvus_expr
from src.retriever.multi_hop import MultiHopRetriever, _is_cross_protocol_query, needs_multi_hop
from src.retriever.online_supplement import OnlineSupplement
from src.retriever.query_quality import diagnose_quality, evaluate_quality, filter_noise
from src.retriever.reranker import get_reranker
from src.retriever.result_quality import filter_low_quality
from src.retriever.search import HybridRetriever, RetrievalResult
from src.retriever.vector_store import VectorStore
from src.utils.monitoring import record_error, record_multi_hop, record_search

logger = logging.getLogger(__name__)


# 合并翻译+扩展的 system prompt — 一次 LLM 调用替代两次串行调用 (省 ~2-4s)
_TRANSLATE_EXPAND_SYSTEM = """你是通信规范查询优化器（3GPP / O-RAN）。将用户查询翻译为技术英文，并扩展为更适合检索的关键词组合。
规则：
1. 保留规范编号（TS xx.xxx / TR xx.xxx）与技术缩写（NR, AMF, SMF, SLPP 等）
2. 使用精确的通信标准术语
3. 添加同义词和相关协议名
4. 严格输出两行（不要任何解释或空行）：
TRANSLATED: <英文翻译>
EXPANDED: <扩展后的英文检索查询>"""


_PRECISE_SPEC_RE = re.compile(r"\b\d{2}\.\d{3}\b")
_PRECISE_SECTION_RE = re.compile(r"§\s*\d+|section\s+\d+", re.IGNORECASE)


def _is_precise_query(query: str) -> bool:
    """查询已含规范号 + 章节引用 → 足够精确, 跳过 LLM 扩展."""
    return bool(_PRECISE_SPEC_RE.search(query) and _PRECISE_SECTION_RE.search(query))


def _history_fingerprint(history: list[dict[str, str]] | None) -> str:
    """多轮历史指纹 — 用于查询扩展缓存 Key."""
    if not history:
        return ""
    return "|".join(
        f"{m.get('role', '')}:{m.get('content', '')[:200]}" for m in history[-6:]
    )


@dataclass
class RetrievalContext:
    """检索规划的完整产出 — plan() 的返回类型.

    生成阶段所需的一切上下文都从这里取, 不再使用字符串键 dict.
    initial_results 为精排前的候选池 (供评测对比"初检召回" vs "重排后召回").
    """

    query_lang: str
    search_query: str
    expanded_query: str
    results: list[RetrievalResult] = field(default_factory=list)
    initial_results: list[RetrievalResult] = field(default_factory=list)
    release_note: str = ""
    online_context: str = ""


class RetrievalPlanner:
    """检索规划器 — 查询 → 检索上下文的全部策略都在这里.

    接口窄: plan() + search() + expand_adjacent_chunks();
    实现厚: 翻译扩展/混合检索/分类分解/精排/多跳/交叉引用/质量评估等.
    """

    def __init__(self, vector_store: VectorStore, llm_client: LLMClient):
        self._store = vector_store
        self._llm = llm_client
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.bm25_top_k,
            final_top_k=settings.max_search_results,
        )
        self._multi_hop = MultiHopRetriever(
            retriever=self._retriever,
            llm_client=self._llm,
            embed_batch_fn=self._get_query_embeddings,
            max_rounds=settings.multi_hop_max_rounds,
            max_sub_queries=settings.multi_hop_max_sub_queries,
        )
        self._online = OnlineSupplement(
            google_api_key=settings.google_api_key,
            google_cse_id=settings.google_cse_id,
            tspec_url=settings.tspec_llm_url,
            score_threshold=settings.online_score_threshold,
        )

        # 离线 Xref Graph 扩展器 (可选加载)
        self._graph_expander: GraphExpander | None = None
        # 查询扩展结果缓存 (query+history 指纹, TTL 1h)
        self._expand_cache: TTLCache = TTLCache(maxsize=512, ttl=3600)
        # TTLCache 非线程安全 — SSE 后台线程与事件循环线程并发访问, 需加锁
        self._expand_lock = threading.Lock()
        xref_path = settings.data_abs_dir / "processed" / "xref_graph.json"
        if xref_path.exists():
            try:
                self._graph_expander = GraphExpander(xref_path, store=vector_store)
                if self._graph_expander.load():
                    logger.info("Xref Graph 扩展器已就绪")
            except Exception as e:
                logger.warning("Xref Graph 加载失败: %s", e)

    # ── 公开接口 ──

    def plan(
        self,
        query: str,
        reranker_enabled: bool = True,
        history: list[dict[str, str]] | None = None,
        release: str | None = None,
        series: str | None = None,
        doc_type: str | None = None,
    ) -> RetrievalContext:
        """检索阶段 (Step 0-3.9) — 翻译/扩展/混合检索/精排/多跳/交叉引用/上下文扩展.

        供 ask() 与 ask_stream() 共用, 避免双份逻辑漂移.
        """
        # Step 0: 多语言处理 — 检测语言, 非英文查询翻译为英文再检索
        query_lang = detect_language(query)
        # Step 1: 查询扩展 (含多轮对话上下文)
        # 优化: 非英文查询时合并 翻译+扩展 为一次 LLM 调用 (原串行两次, 省 ~2-4s)
        if query_lang == "en":
            search_query = query
            expanded_query = self._expand_query(query, history)
        else:
            search_query, expanded_query = self._translate_and_expand(query, query_lang, history)

        # Step 2: 生成查询嵌入
        query_embedding = self._get_query_embedding(expanded_query)

        # Step 3: 混合检索
        t_search = time.time()
        filter_expr = self._build_filter_expr(release=release, series=series, doc_type=doc_type)
        # 若启用 reranker, 先用大池子检索再精排 (final_top_k 作为参数传入, 避免共享可变状态)
        search_top_k = settings.reranker_top_k if reranker_enabled else settings.max_search_results
        results = self._retriever.search(
            expanded_query, query_embedding,
            filter_expr=filter_expr, final_top_k=search_top_k,
        )

        # Step 3.1: 分类列举查询 → 多角度分解检索
        if is_taxonomy_query(search_query):
            results = self._taxonomy_decompose_search(
                search_query, query_embedding, results, filter_expr=filter_expr,
            )

        # 记录精排前候选池 — 供评测对比"初检召回"(rerank 前) 与"重排后召回"
        initial_results = results

        # Spec-aware + Cross-Encoder Reranker 精排
        results = self._post_process_results(
            query=query, expanded_query=expanded_query,
            query_embedding=query_embedding, results=results,
            top_k=settings.max_search_results, reranker_enabled=reranker_enabled,
            filter_expr=filter_expr,
        )

        # Step 3.1b: 分类列举查询 → 元数据 boost
        # (注: 曾尝试对所有查询生效以修复 "LTE 概述压过权威", 但实测 RAN 召回
        #  -0.026 且 LTE 无改善 — 根因是 36.211 池外漏召而非排序, boost 无效; 回滚)
        if is_taxonomy_query(search_query):
            results = self._apply_metadata_boost(results)

        # 过滤低信息密度章节 (缩写表/参考文献等)
        results = filter_low_quality(results, settings.max_search_results)
        search_dt = time.time() - t_search

        # Step 3.2: 多跳检索 — 触发依据优先 query 跨协议信号, 而非结果多样性
        if needs_multi_hop(results, query=search_query):
            record_multi_hop()
            logger.info(
                "触发多跳检索 (多样性=%.2f)",
                len({r.spec_number for r in results if r.spec_number}) / len(results),
            )
            try:
                results = self._multi_hop.search(
                    search_query, query_embedding,
                    initial_results=results, filter_expr=filter_expr,
                )
            except Exception as e:
                logger.warning("多跳检索失败, 使用单跳结果: %s", e)
                record_error("multi_hop_failed")

        # Step 3.5: 交叉引用解析 — 优先用离线图扩展，降级为在线二次检索
        if self._graph_expander and self._graph_expander.is_loaded:
            expanded = self._graph_expander.expand(results, max_per_chunk=5, top_n=10)
            if expanded:
                expanded_results = [
                    RetrievalResult.from_search_result(r) for r in expanded
                ]
                results = results + expanded_results
                logger.info("图增强检索: +%d 条 cross-spec chunk", len(expanded))
        else:
            results = self._resolve_cross_refs(results, max_refs=5, filter_expr=filter_expr)

        # Step 3.6: 检索质量评估
        quality = evaluate_quality(results)
        action = diagnose_quality(quality, len(results))
        logger.info(
            "检索质量: 密度=%.3f 多样性=%.2f 覆盖=%d | %s | 建议: rewrite=%s expand=%s suggest=%s",
            quality.density, quality.diversity, quality.coverage, action.reason,
            action.should_rewrite, action.should_expand, action.should_suggest,
        )
        if not quality.overall_ok:
            logger.warning("检索质量不达标, 触发降级策略")
        results = filter_noise(results)

        # 记录检索指标
        avg_score = sum(r.score for r in results) / len(results) if results else 0.0
        record_search(len(results), avg_score, search_dt)

        # Step 3.7: Release 版本感知
        release_intent = detect_release_intent(search_query)
        release_note = ""
        if release_intent.type.value != "none":
            logger.info(
                "Release 意图: type=%s releases=%s comparative=%s",
                release_intent.type.value, release_intent.releases, release_intent.is_comparative,
            )
            results, release_note = build_release_context(results, release_intent)
            release_note = build_release_note_for_prompt(release_intent, release_note)

        # Step 3.8: 在线搜索补充 — 离线不足时自动补 Google/TSpec-LLM
        online_context = ""
        if settings.enable_online_search and self._online.enabled:
            best_score = max((r.score for r in results), default=0.0)
            if self._online.should_supplement(best_score, len(results)):
                online_results = self._online.supplement_if_needed(
                    search_query, best_score, len(results),
                )
                online_context = self._online.format_as_context(online_results)

        # Step 3.95: spec 多样性重排 (可选) — 避免单一 spec 霸榜前列
        if settings.diversify_topk_max_per_spec > 0:
            max_per_spec = settings.diversify_topk_max_per_spec
            # 跨协议题需要覆盖多个 spec, 用更激进的上限 (=1) 确保每 spec 至少进前列;
            # 普通题保持默认 (=2), 避免单 spec 题被无关 spec 稀释
            if _is_cross_protocol_query(search_query):
                max_per_spec = 1
            results = self._diversify_topk(results, max_per_spec)

        # Step 3.9: small-to-big 父上下文扩展 — 命中子 chunk 附带父 section 文本
        self.expand_parent_context(results)

        return RetrievalContext(
            query_lang=query_lang,
            search_query=search_query,
            expanded_query=expanded_query,
            results=results,
            initial_results=initial_results,
            release_note=release_note,
            online_context=online_context,
        )

    def search(
        self, query: str, top_k: int | None = None, reranker_enabled: bool = True,
        release: str | None = None, series: str | None = None,
        doc_type: str | None = None,
    ) -> list[RetrievalResult]:
        """仅执行检索 (不生成答案) — 含 spec-aware + Cross-Encoder 重排序.

        Phase 1: 常规混合检索 (Dense+BM25+RRF).
        Phase 2: 若 LLM 扩展查询中包含规范号, 对提示规范做定向检索并合并.
        Phase 3: Cross-Encoder Reranker 对候选精排 (可选启用).
        """
        _top_k = top_k or settings.max_search_results
        # 若启用 reranker, 混合检索用更大的候选池 (reranker 从 N 条中精选 top_k)
        pool_size = settings.reranker_top_k if reranker_enabled else _top_k
        expanded_query = self._expand_query(query)
        query_embedding = self._get_query_embedding(expanded_query)
        filter_expr = self._build_filter_expr(release=release, series=series, doc_type=doc_type)
        results = self._retriever.search(
            expanded_query, query_embedding,
            filter_expr=filter_expr, final_top_k=top_k if top_k is not None else pool_size,
        )

        # Cross-Encoder Reranker 精排 (先执行, 提升候选池质量)
        results = self._rerank(expanded_query, results, _top_k, reranker_enabled=reranker_enabled)

        # Spec-aware 两阶段检索: 对 LLM 识别出的规范号做定向补充+加权 (最后执行, 确保覆盖)
        spec_hints = _extract_spec_numbers(expanded_query)
        if spec_hints:
            results = _spec_aware_rerank(
                results, spec_hints, self._retriever._store,
                query_embedding, _top_k, filter_expr=filter_expr,
            )

        # small-to-big 父上下文扩展 — 命中子 chunk 附带父 section 文本
        self.expand_parent_context(results)
        return results

    def expand_parent_context(
        self,
        results: list[RetrievalResult],
        top_n: int = 10,
        max_chars: int = 1500,
    ) -> None:
        """small-to-big 父上下文扩展 — 命中子 chunk 时附带父 section 完整文本.

        将检索结果携带的 parent_text (摄入时写入) 提升为 parent_context,
        供 prompt 组装注入 LLM 上下文。跳过条件:
        - parent_text 为空 (chunk 本身即完整 section, 无父级)
        - parent_text 与 chunk 文本几乎相同 (冗余, 避免重复注入)
        仅扩展前 top_n 条, 且每条截断至 max_chars, 控制上下文总量.
        """
        expanded = 0
        for r in results[:top_n]:
            parent = getattr(r, "parent_text", "") or ""
            if not parent:
                continue
            if parent == r.text or parent in r.text:
                continue
            r.parent_context = parent[:max_chars]
            expanded += 1
        if expanded:
            logger.info(
                "small-to-big 父上下文扩展: %d/%d 条结果附带父 section 文本",
                expanded, min(len(results), top_n),
            )

    def expand_adjacent_chunks(
        self,
        results: list[RetrievalResult],
        top_n: int = 5,
        window: int = 2,
    ) -> None:
        """为 Top-N 命中 chunk 拉取同文档相邻 chunk，扩充上下文.

        解决表格/列表类内容因 chunk 碎片化导致的召回不完整问题。
        对每个命中 chunk，查询同 doc_id 下 chunk_index ± window 的相邻块，
        将其文本存入 adjacent_chunks 字段，供 to_context_str 拼入 LLM 上下文。

        Args:
            results: 检索结果列表（原地修改 adjacent_chunks 字段）.
            top_n: 仅为前 top_n 条结果拉取相邻 chunk (控制延迟).
            window: 左右各取 window 个相邻 chunk.
        """
        if not hasattr(self._store, 'get_adjacent_chunks'):
            return

        expanded = 0
        for r in results[:top_n]:
            if not r.doc_id or r.chunk_index < 0:
                continue
            try:
                adjacent = self._store.get_adjacent_chunks(
                    r.doc_id, r.chunk_index, window=window,
                )
                if adjacent:
                    r.adjacent_chunks = [a.text for a in adjacent]
                    expanded += 1
            except Exception as e:
                logger.debug("相邻 chunk 查询跳过 (doc=%s idx=%d): %s",
                             r.doc_id, r.chunk_index, e)

        if expanded > 0:
            logger.info("相邻 chunk 上下文扩展: %d/%d 条结果已扩充 (±%d)",
                        expanded, min(len(results), top_n), window)

    @staticmethod
    def _diversify_topk(
        results: list[RetrievalResult],
        max_per_spec: int,
    ) -> list[RetrievalResult]:
        """按 spec 分散前列排序 — 避免单一 spec 挤占 top-k.

        贪心: 遍历原始 (已重排) 顺序, 每个 spec 在前列最多保留 max_per_spec 条,
        超额者顺延到末尾 (保持各自原相对顺序)。不丢结果、只重排。
        单 spec 题不受影响: 该 spec 的前 max_per_spec 条仍在最前。
        """
        if max_per_spec <= 0 or len(results) <= 1:
            return results
        head: list[RetrievalResult] = []
        tail: list[RetrievalResult] = []
        seen: dict[str, int] = {}
        for r in results:
            spec = r.spec_number or ""
            if seen.get(spec, 0) < max_per_spec:
                head.append(r)
                seen[spec] = seen.get(spec, 0) + 1
            else:
                tail.append(r)
        return head + tail

    # ── 查询翻译/扩展 ──

    def _expand_query(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        """查询扩展 — 精确查询跳过 + 结果缓存, 减少 LLM 调用."""
        if not settings.query_expansion_enabled:
            return query
        if _is_precise_query(query):
            logger.debug("查询已含规范号+章节引用, 跳过扩展: %.60s", query)
            return query

        cache_key = hashlib.sha256(
            f"{query.lower().strip()}|{_history_fingerprint(history)}".encode()
        ).hexdigest()
        with self._expand_lock:
            cached = self._expand_cache.get(cache_key)
        if cached is not None:
            logger.debug("查询扩展缓存命中: %.60s", query)
            return cached

        # 缩写展开 — 缓存 key 基于原始查询 (缓存兼容), 展开结果喂给 LLM 扩展
        abbr_query = expand_abbreviations(query)
        if abbr_query != query:
            logger.debug("缩写展开: %.60s → %.120s", query, abbr_query)

        try:
            messages = build_query_expansion_prompt(abbr_query, history)
            expanded = self._llm.chat(messages)
            if expanded and len(expanded) > 5:
                with self._expand_lock:
                    self._expand_cache[cache_key] = expanded
                return expanded
        except Exception as e:
            logger.warning("查询扩展失败: %s", e)
        return query

    def _get_query_embeddings(self, texts: list[str]) -> list[np.ndarray]:
        """批量生成查询嵌入 — 一次模型推理替代逐条调用."""
        if not texts:
            return []
        try:
            embeddings = self._llm.embed(texts)
            return [np.array(e, dtype=np.float32) for e in embeddings]
        except Exception as e:
            logger.warning("批量嵌入失败, 逐条降级: %s", e)
            return [self._get_query_embedding(t) for t in texts]

    def _translate_and_expand(
        self,
        query: str,
        source_lang: str,
        history: list[dict[str, str]] | None = None,
    ) -> tuple[str, str]:
        """合并执行 翻译(→EN) + 查询扩展 — 一次 LLM 调用替代两次串行调用.

        Returns:
            (search_query, expanded_query). 失败时回退到原串行路径, 不损失质量.
        """
        # 非英文查询同样走缓存 (修复: 此前只缓存英文扩展, 非英文每次重复调 LLM)
        cache_key = hashlib.sha256(
            f"tr|{source_lang}|{query.lower().strip()}|{_history_fingerprint(history)}".encode()
        ).hexdigest()
        with self._expand_lock:
            cached = self._expand_cache.get(cache_key)
        if cached is not None:
            logger.debug("翻译+扩展缓存命中: %.60s", query)
            return cached

        try:
            # 缩写展开 — 中英混合查询中的英文缩写同样展开, 让 LLM 看到全称
            abbr_query = expand_abbreviations(query)
            user_content = f"用户查询: {abbr_query}"
            if history and len(history) >= 2:
                history_lines = ["## 对话历史 (用于理解当前问题的上下文)"]
                for msg in history[-6:]:
                    role = "用户" if msg["role"] == "user" else "助手"
                    history_lines.append(f"{role}: {msg['content'][:200]}")
                history_lines.append(f"\n## 当前问题 (需要翻译+扩展为英文检索查询)\n{abbr_query}")
                user_content = "\n".join(history_lines)
            result = self._llm.chat(
                [
                    {"role": "system", "content": _TRANSLATE_EXPAND_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
                max_tokens=512,
            )
            translated = expanded = ""
            for line in result.splitlines():
                line = line.strip()
                if line.startswith("TRANSLATED:"):
                    translated = line[len("TRANSLATED:"):].strip()
                elif line.startswith("EXPANDED:"):
                    expanded = line[len("EXPANDED:"):].strip()
            if len(translated) > 3 and len(expanded) > 3:
                logger.info("翻译+扩展合并: %s → EN (%.60s)", source_lang, translated)
                with self._expand_lock:
                    self._expand_cache[cache_key] = (translated, expanded)
                return translated, expanded
            logger.warning("合并解析失败, 回退串行路径: %.80s", result[:80])
        except Exception as e:
            logger.warning("合并翻译+扩展失败, 回退串行路径: %s", e)
        # 回退: 原串行路径 (翻译 → 扩展), 保证质量不降级
        search_query = translate_to_english(query, source_lang, self._llm)
        return search_query, self._expand_query(search_query, history)

    def _get_query_embedding(self, query: str) -> np.ndarray:
        # 查询侧嵌入输入为纯查询文本, 与文档侧 src.ingestion.embedder.embedding_text
        # 的纯正文构成保持一致 —— 两侧向量空间对齐 (BGE-M3 对输入前部敏感,
        # 文档侧不得拼接 section_title/section_path, 否则产生 domain shift)。
        try:
            embeddings = self._llm.embed([query])
            return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning("嵌入生成失败，使用零向量: %s", e)
            return np.zeros(1024, dtype=np.float32)

    # ── 后处理: 精排与加权 ──

    def _post_process_results(
        self,
        query: str,
        expanded_query: str,
        query_embedding: np.ndarray,
        results: list[RetrievalResult],
        top_k: int,
        reranker_enabled: bool = True,
        filter_expr: str | None = None,
    ) -> list[RetrievalResult]:
        """检索后处理: Cross-Encoder 精排 → spec-aware 定向补充+加权.

        顺序很重要: Cross-Encoder 先精排候选池质量, 然后 spec-aware
        再对目标规范加权/补充——确保协议相关内容不受 Cross-Encoder
        窄域低区分度影响而被稀释。
        """
        # Step 1: Cross-Encoder Reranker (先精排, 提升候选池整体质量)
        results = self._rerank(expanded_query, results, top_k, reranker_enabled=reranker_enabled)
        # Step 2: Spec-aware 定向补充 + 加权覆盖 (最后执行, 确保目标规范排到前面)
        spec_hints = _extract_spec_numbers(expanded_query)
        if spec_hints:
            results = _spec_aware_rerank(
                results, spec_hints, self._retriever._store,
                query_embedding, top_k, filter_expr=filter_expr,
            )
        return results

    def _rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int,
        reranker_enabled: bool = True,
    ) -> list[RetrievalResult]:
        """Cross-Encoder 精排 — 分数融合 (保留 spec-aware/RRF 信息).

        Cross-Encoder 在窄域 (如 3GPP 规范) 内判别力有限, 完全替换原有分数
        会导致已调优的 spec-aware + RRF 排序退化。因此采用加权融合:
        final = α × norm_reranker + (1-α) × norm_original
        默认 α=0.6 (reranker 主, 原始辅).
        """
        if not reranker_enabled:
            return results[:top_k]
        reranker = get_reranker()
        if reranker is None or len(results) <= top_k:
            return results[:top_k]

        # 保存原始分数 (按 chunk_id 映射, 供融合时对齐 — reranker 会改变顺序)
        orig_by_id = {str(r.chunk_id): r.score for r in results}
        orig_scores = np.array(list(orig_by_id.values()), dtype=np.float32)
        orig_min = float(orig_scores.min()) if len(orig_scores) else 0.0
        orig_max = float(orig_scores.max()) if len(orig_scores) else 0.0

        try:
            # 对全候选池打分 (不先截断 top_k) — RRF 靠前但 reranker 排 21+
            # 的真阳性必须参与融合, 否则在 reranker 截断时直接出局
            reranked = reranker.rerank(query, results, top_k=len(results))
        except Exception as e:
            logger.warning("Reranker 失败, 降级返回原始候选: %s", e)
            return results[:top_k]

        if not reranked:
            return results[:top_k]

        # 归一化 reranker 分数到 [0, 1]
        rerank_scores = np.array([r.score for r in reranked], dtype=np.float32)
        rerank_min, rerank_max = rerank_scores.min(), rerank_scores.max()

        # 加权融合: reranker 60% + 原始 40%
        ALPHA = 0.6
        for r in reranked:
            orig = orig_by_id.get(str(r.chunk_id), 0.0)
            if orig_max > orig_min:
                orig_norm = (orig - orig_min) / (orig_max - orig_min)
            else:
                orig_norm = 0.5
            if rerank_max > rerank_min:
                rerank_norm = (r.score - rerank_min) / (rerank_max - rerank_min)
            else:
                rerank_norm = 0.5
            # 必须转回 Python float — numpy 标量 (np.float32) 无法被 json.dumps 序列化,
            # 会导致 SSE sources 事件序列化崩溃 (TypeError: float32 not JSON serializable)
            r.score = float(ALPHA * rerank_norm + (1 - ALPHA) * orig_norm)

        # 写回融合分数并排序
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]

    def _apply_metadata_boost(
        self, results: list[RetrievalResult],
    ) -> list[RetrievalResult]:
        """按 chunk 元数据加权重新排序.

        - authoritative spec → ×1.3
        - parameter_table / definition → ×1.2
        两者可叠加，最大 boost ×1.56
        """
        for r in results:
            boost = 1.0
            if r.spec_role == "authoritative":
                boost *= 1.3
            if r.content_type in ("parameter_table", "definition"):
                boost *= 1.2
            if boost > 1.0:
                r.score = float(r.score) * boost
                logger.debug(
                    "元数据 boost: %s %s/%s score=%.3f→%.3f",
                    r.spec_number, r.spec_role, r.content_type,
                    r.score / boost, r.score,
                )
        results.sort(key=lambda r: r.score, reverse=True)
        return results

    # ── 分类列举多角度分解检索 ──

    def _taxonomy_decompose_search(
        self,
        query: str,
        query_embedding: np.ndarray,
        existing_results: list[RetrievalResult],
        filter_expr: str | None = None,
    ) -> list[RetrievalResult]:
        """对分类列举类查询做多角度分解检索.

        用 LLM 将问题分解为 3-5 个子查询 (每子查询覆盖一个类别维度),
        并行执行子查询搜索, 结果合并去重. 原始检索结果排在前面,
        子查询补充结果追加到后面.
        """
        sub_queries = self._generate_taxonomy_sub_queries(query)
        if not sub_queries:
            return existing_results

        seen_ids = {str(r.chunk_id) for r in existing_results}
        supplement: list[RetrievalResult] = []

        sub_embeds = self._get_query_embeddings(sub_queries)
        for sub_q, sub_embed in zip(sub_queries, sub_embeds):
            try:
                # 子查询用较小的 top_k, 避免单子查询占据全部上下文
                sub_results = self._retriever.search(
                    sub_q, sub_embed, filter_expr=filter_expr,
                    final_top_k=max(settings.max_search_results // 2, 5),
                )

                for r in sub_results:
                    rid = str(r.chunk_id)
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        r._source_tag = "taxonomy"
                        r._sub_query = sub_q[:80]
                        supplement.append(r)
            except Exception as e:
                logger.warning("分类子查询检索失败 [%s]: %s", sub_q[:40], e)

        if supplement:
            logger.info(
                "分类分解检索: %d 子查询 → +%d 补充 (原始 %d → 总计 %d)",
                len(sub_queries), len(supplement),
                len(existing_results), len(existing_results) + len(supplement),
            )

        return list(existing_results) + supplement

    def _generate_taxonomy_sub_queries(self, query: str) -> list[str]:
        """用 LLM 将分类列举问题分解为 3-5 个覆盖不同维度的子查询."""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个通信规范（3GPP / O-RAN）检索专家。将用户的分类列举问题分解为多个具体子查询，"
                    "每个子查询覆盖一个类别/维度。\n\n"
                    "规则：\n"
                    "1. 每个子查询应该是独立、可直接检索的关键词组合\n"
                    "2. 子查询应覆盖用户问题的所有类别，不遗漏\n"
                    "3. 使用规范术语（如 PRACH preamble, RACH occasion）\n"
                    "4. 输出每行一个子查询，不要编号，不要解释\n"
                    "5. 最多 5 个子查询"
                ),
            },
            {"role": "user", "content": f"将以下问题分解为覆盖所有维度的子查询：\n\n{query}"},
        ]
        try:
            response = self._llm.chat(messages, temperature=0.0, max_tokens=1024)
            sub_queries = [
                line.strip()
                for line in response.strip().split("\n")
                if line.strip() and len(line.strip()) > 5
            ]
            if sub_queries:
                logger.info("分类子查询生成: %d 条 → %s", len(sub_queries),
                             [q[:50] for q in sub_queries])
            return sub_queries[:5]
        except Exception as e:
            logger.warning("分类子查询生成失败: %s", e)
            return []

    # ── 交叉引用解析 ──

    def _resolve_cross_refs(
        self,
        results: list[RetrievalResult],
        max_refs: int = 5,
        filter_expr: str | None = None,
    ) -> list[RetrievalResult]:
        """从检索结果中解析交叉引用并二次检索补充上下文.

        流程:
        1. 扫描每个结果中的 3GPP 规范引用 (如 TS 38.413 §8.3.1)
        2. 去重后对每个引用执行二次检索
        3. 将新结果合并到原始列表

        Args:
            results: 原始检索结果
            max_refs: 最多处理的引用数量 (控制延迟)

        Returns:
            原始结果 + 交叉引用补充结果
        """
        # 1. 提取所有引用
        all_refs: list = []
        for chunk in results[:10]:  # 仅扫描 Top-10，控制耗时
            all_refs.extend(extract_references(chunk.text))

        unique_refs = _deduplicate_refs(all_refs)
        if not unique_refs:
            logger.debug("交叉引用解析: 未发现有效引用")
            return results

        # 2. 限制处理数量
        concrete_refs = [r for r in unique_refs if r.spec_number != "?"]
        if len(concrete_refs) > max_refs:
            concrete_refs = concrete_refs[:max_refs]

        logger.info(
            "交叉引用解析: 发现 %d 引用 (去重后 %d, 处理 %d)",
            len(all_refs), len(unique_refs), len(concrete_refs),
        )

        # 3. 二次检索
        supplement: list[RetrievalResult] = []
        seen_ids = {r.chunk_id for r in results}
        ref_queries = [(ref, ref.to_search_query()) for ref in concrete_refs]
        embeddings = self._get_query_embeddings([q for _, q in ref_queries])
        for (ref, query), embedding in zip(ref_queries, embeddings):
            try:
                ref_results = self._retriever.search(
                    query, embedding, filter_expr=filter_expr,
                )
                for r in ref_results:
                    if r.chunk_id not in seen_ids:
                        seen_ids.add(r.chunk_id)
                        r._source_tag = "cross_ref"
                        r._ref_from = ref.raw_text
                        supplement.append(r)
                logger.debug("  二次检索 [%s] → %d 条", ref.lookup_key, len(ref_results))
            except Exception as e:
                logger.warning("交叉引用二次检索失败 [%s]: %s", ref.lookup_key, e)

        # 4. 合并
        merged = list(results) + supplement
        logger.info("交叉引用解析完成: %d 原始 + %d 补充 = %d 条",
                     len(results), len(supplement), len(merged))
        return merged

    # ── 过滤表达式 ──

    @staticmethod
    def _build_filter_expr(
        release: str | None = None,
        series: str | None = None,
        doc_type: str | None = None,
    ) -> str | None:
        """构建 Milvus 标量过滤表达式.

        Release/Series 仅对 3GPP 有意义 (ORAN 无此概念).
        - 未选 doc_type 时: release/series 只约束 3GPP, ORAN 始终放行
        - 选 doc_type=3gpp 时: release/series 正常生效
        - 选 doc_type=oran 时: release/series 忽略
        """
        has_filter = release or series

        if doc_type == "oran":
            # ORAN 明确选中: 仅按 doc_type 过滤
            return 'doc_type == "oran"'

        if doc_type == "3gpp":
            # 3GPP 明确选中: release/series 正常生效
            parts: list[str] = []
            if release:
                parts.append(f'release == "{_escape_milvus_expr(release)}"')
            if series:
                parts.append(f"series == {_escape_milvus_expr(series)}")
            parts.append('doc_type == "3gpp"')
            return " && ".join(parts)

        # doc_type 未选: release/series 只约束 3GPP, ORAN 放行
        if has_filter:
            parts_3gpp: list[str] = []
            if release:
                parts_3gpp.append(f'release == "{_escape_milvus_expr(release)}"')
            if series:
                parts_3gpp.append(f"series == {_escape_milvus_expr(series)}")
            parts_3gpp.append('doc_type == "3gpp"')
            return f'(({" && ".join(parts_3gpp)}) || doc_type == "oran")'

        return None


# ── Spec-aware 重排序辅助函数 ──

_SPEC_PATTERN = re.compile(r'\b(\d{2}\.\d{3})\b')


def _extract_spec_numbers(text: str) -> set[str]:
    """从文本中提取 3GPP 规范号 (如 38.211, 23.501)."""
    return set(_SPEC_PATTERN.findall(text))


def _spec_aware_rerank(
    results: list["RetrievalResult"],
    spec_hints: set[str],
    store,
    query_embedding,
    top_k: int,
    boost_factor: float = 2.0,
    filter_expr: str | None = None,
) -> list["RetrievalResult"]:
    """两阶段 spec-aware 重排序.

    1. 对每个 hint spec 做定向 Dense 检索 (最多 3 条/spec), 补充到结果列表.
    2. 对匹配 hint spec 的结果分数加权 (×boost_factor).
    3. 按分数降序排列返回.
    """
    seen_ids: set = {str(r.chunk_id) for r in results}  # 统一 str 避免 int/str 混合导致去重失效
    supplement: list["RetrievalResult"] = []

    # Phase 2: 定向检索 hint specs (用 Milvus 标量过滤)
    for spec in list(spec_hints)[:3]:  # 最多 3 个 hint spec
        try:
            escaped_spec = _escape_milvus_expr(spec)
            spec_expr = f'spec_number == "{escaped_spec}"'
            # 保留用户级过滤 (release/series/doc_type), 避免补充结果污染过滤语义
            if filter_expr:
                spec_expr = f"({filter_expr}) && ({spec_expr})"
            spec_results = store.search_dense(
                query_embedding, top_k=5,
                filter_expr=spec_expr,
            )
            for r in spec_results[:5]:
                rid = str(r.chunk_id) if hasattr(r, 'chunk_id') else str(id(r))  # 统一 str 去重
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    nr = RetrievalResult.from_search_result(r)
                    nr.score = nr.score * boost_factor
                    supplement.append(nr)
        except Exception:
            pass

    # Boost existing matches
    boosted: list["RetrievalResult"] = []
    for r in results:
        if r.spec_number in spec_hints:
            nr = copy.copy(r)
            nr.score = r.score * boost_factor
            boosted.append(nr)
        else:
            boosted.append(r)

    merged = boosted + supplement
    merged.sort(key=lambda x: x.score, reverse=True)
    return merged
