"""RAG Pipeline — 检索 + 生成 + 验证一体化编排."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

import numpy as np
from cachetools import TTLCache

from src.config import settings
from src.generator.i18n import detect_language, translate_from_english, translate_to_english
from src.generator.llm_client import LLMClient
from src.generator.prompt import build_query_expansion_prompt, build_rag_prompt
from src.generator.release_aware import (
    build_release_context,
    build_release_note_for_prompt,
    detect_release_intent,
)
from src.generator.verifier import AnswerVerifier
from src.retriever.cross_ref import _deduplicate_refs, extract_references
from src.retriever.multi_hop import MultiHopRetriever, needs_multi_hop
from src.retriever.online_supplement import OnlineSupplement
from src.retriever.query_quality import diagnose_quality, evaluate_quality, filter_noise
from src.retriever.search import HybridRetriever, RetrievalResult
from src.retriever.vector_store import VectorStore
from src.utils.monitoring import (
    record_ask,
    record_error,
    record_llm_call,
    record_multi_hop,
    record_search,
)

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """RAG 查询的完整响应."""
    query: str
    answer: str
    sources: list[RetrievalResult]
    verified: bool
    warnings: list[str]
    coverage: float = 0.0
    expanded_query: str = ""


class RAGPipeline:
    """3GPP RAG 查询流水线.

    流程: 查询扩展 -> 混合检索 -> RAG 提示词 -> LLM 生成 -> 答案验证
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient | None = None,
    ):
        self._store = vector_store
        self._llm = llm_client or LLMClient()
        self._retriever = HybridRetriever(
            vector_store=vector_store,
            dense_top_k=settings.dense_top_k,
            sparse_top_k=settings.bm25_top_k,
            final_top_k=settings.max_search_results,
        )
        self._multi_hop = MultiHopRetriever(
            retriever=self._retriever,
            llm_client=self._llm,
            embed_fn=self._get_query_embedding,
        )
        self._verifier = AnswerVerifier()
        self._online = OnlineSupplement(
            google_api_key=settings.google_api_key,
            google_cse_id=settings.google_cse_id,
            tspec_url=settings.tspec_llm_url,
            score_threshold=settings.online_score_threshold,
        )
        # 查询级 LRU 缓存: TTL 1 小时, 最多 256 条, key=md5(query)
        self._query_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

    def ask(self, query: str) -> RAGResponse:
        """执行完整 RAG 问答."""
        # 查询缓存: 相同查询 1 小时内直接返回缓存结果
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            logger.info("查询缓存命中: %s", query[:80])
            return cached

        t_start = time.time()

        # Step 0: 多语言处理 — 检测语言, 非英文查询翻译为英文再检索
        query_lang = detect_language(query)
        search_query = translate_to_english(query, query_lang, self._llm)

        # Step 1: 查询扩展
        expanded_query = self._expand_query(search_query)

        # Step 2: 生成查询嵌入
        query_embedding = self._get_query_embedding(expanded_query)

        # Step 3: 混合检索
        t_search = time.time()
        results = self._retriever.search(expanded_query, query_embedding)
        search_dt = time.time() - t_search

        # Step 3.2: 多跳检索
        if needs_multi_hop(results):
            record_multi_hop()
            logger.info("触发多跳检索 (多样性=%.2f)",
                         len({r.spec_number for r in results if r.spec_number}) / len(results))
            try:
                results = self._multi_hop.search(search_query, query_embedding)
            except Exception as e:
                logger.warning("多跳检索失败, 使用单跳结果: %s", e)
                record_error("multi_hop_failed")

        # Step 3.5: 交叉引用解析
        results = self._resolve_cross_refs(results, max_refs=5)

        # Step 3.6: 检索质量评估
        quality = evaluate_quality(results)
        action = diagnose_quality(quality, len(results))
        logger.info(
            "检索质量: 密度=%.3f 多样性=%.2f 覆盖=%d | %s",
            quality.density, quality.diversity, quality.coverage, action.reason,
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

        if not results:
            record_ask(time.time() - t_start, success=True)
            return RAGResponse(
                query=query,
                answer="未在 3GPP 规范中找到相关内容。",
                sources=[],
                verified=True,
                warnings=[],
            )

        # Step 4: 构建 RAG 提示词 (使用英文检索查询, 保证与检索上下文语言一致)
        messages = build_rag_prompt(
            search_query, results,
            extra_system_note=release_note,
            online_context=online_context,
        )

        # Step 5: LLM 生成
        t_llm = time.time()
        answer = self._llm.chat(messages)
        llm_dt = time.time() - t_llm
        # 估算 token 消耗 (粗略: 4 char ≈ 1 token)
        prompt_chars = sum(len(m.get("content", "")) for m in messages)
        record_llm_call(
            prompt_tokens=prompt_chars // 4,
            completion_tokens=len(answer) // 4,
            duration_s=llm_dt,
        )

        # Step 5.5: 多语言回译 — 英文答案 → 用户源语言
        if query_lang != "en":
            answer = translate_from_english(answer, query_lang, self._llm)

        # Step 6: 答案验证
        verification = self._verifier.verify(answer, results)

        record_ask(time.time() - t_start)

        response = RAGResponse(
            query=query,
            answer=verification["answer"],
            sources=results,
            verified=verification["verified"],
            warnings=verification["warnings"],
            coverage=verification["coverage"],
            expanded_query=expanded_query,
        )

        # 存入查询缓存
        self._query_cache[cache_key] = response
        return response

    def search(self, query: str, top_k: int | None = None) -> list[RetrievalResult]:
        """仅执行检索 (不生成答案)."""
        if top_k is not None:
            self._retriever._final_top_k = top_k
        expanded_query = self._expand_query(query)
        query_embedding = self._get_query_embedding(expanded_query)
        return self._retriever.search(expanded_query, query_embedding)

    def _expand_query(self, query: str) -> str:
        try:
            messages = build_query_expansion_prompt(query)
            expanded = self._llm.chat(messages)
            if expanded and len(expanded) > 5:
                return expanded
        except Exception as e:
            logger.warning("查询扩展失败: %s", e)
        return query

    def _get_query_embedding(self, query: str) -> np.ndarray:
        try:
            embeddings = self._llm.embed([query])
            return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning("嵌入生成失败，使用零向量: %s", e)
            return np.zeros(1024, dtype=np.float32)

    def _resolve_cross_refs(
        self,
        results: list[RetrievalResult],
        max_refs: int = 5,
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
        for ref in concrete_refs:
            query = ref.to_search_query()
            try:
                embedding = self._get_query_embedding(query)
                ref_results = self._retriever.search(query, embedding)
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

    def _warmup(self) -> None:
        """预热嵌入模型，避免首次查询加载延迟."""
        try:
            logger.info("预热嵌入模型...")
            self._get_query_embedding("warmup")
            logger.info("嵌入模型预热完成")
        except Exception as e:
            logger.warning("嵌入模型预热失败: %s", e)
