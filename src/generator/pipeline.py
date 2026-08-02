"""RAG Pipeline — 检索 + 生成 + 验证一体化编排."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cachetools import TTLCache

from src.config import settings
from src.generator.i18n import detect_language, translate_from_english, translate_to_english
from src.generator.llm_client import LLMClient
from src.generator.prompt import _is_taxonomy_query, build_query_expansion_prompt, build_rag_prompt
from src.generator.release_aware import (
    build_release_context,
    build_release_note_for_prompt,
    detect_release_intent,
)
from src.generator.verifier import AnswerVerifier
from src.retriever.cross_ref import _deduplicate_refs, extract_references
from src.retriever.graph_expander import GraphExpander
from src.retriever.multi_hop import MultiHopRetriever, needs_multi_hop
from src.retriever.online_supplement import OnlineSupplement
from src.retriever.query_quality import diagnose_quality, evaluate_quality, filter_noise
from src.retriever.reranker import get_reranker
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


# 合并翻译+扩展的 system prompt — 一次 LLM 调用替代两次串行调用 (省 ~2-4s)
_TRANSLATE_EXPAND_SYSTEM = """你是通信规范查询优化器（3GPP / O-RAN）。将用户查询翻译为技术英文，并扩展为更适合检索的关键词组合。
规则：
1. 保留规范编号（TS xx.xxx / TR xx.xxx）与技术缩写（NR, AMF, SMF, SLPP 等）
2. 使用精确的通信标准术语
3. 添加同义词和相关协议名
4. 严格输出两行（不要任何解释或空行）：
TRANSLATED: <英文翻译>
EXPANDED: <扩展后的英文检索查询>"""


# ── 语言兜底与流式切片工具 ──


def _ensure_answer_language(answer: str, target_lang: str, llm: LLMClient) -> str:
    """确保回答语言符合用户语言 — 仅当 LLM 输出语言不符时触发回译.

    优化: 主路径 (prompt 已要求用用户语言回答) 零额外 LLM 调用;
    回译仅在异常路径 (LLM 仍输出英文) 兜底, 不损失回答质量.
    """
    if target_lang == "en":
        return answer
    zh_ratio = sum(1 for ch in answer if "\u4e00" <= ch <= "\u9fff") / max(len(answer), 1)
    if zh_ratio >= 0.05:
        return answer
    logger.info("回答语言不符 (中文占比 %.1f%%), 触发回译兜底 EN→%s", zh_ratio * 100, target_lang)
    return translate_from_english(answer, target_lang, llm)


def _split_for_stream(text: str, size: int = 32) -> list[str]:
    """将完整回答切成小段供流式推送 (缓存命中时使用)."""
    return [text[i:i + size] for i in range(0, len(text), size)]


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
    """CommSpec RAG 查询流水线.

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

        # 离线 Xref Graph 扩展器 (可选加载)
        self._graph_expander: GraphExpander | None = None
        xref_path = settings.data_abs_dir / "processed" / "xref_graph.json"
        if xref_path.exists():
            try:
                self._graph_expander = GraphExpander(xref_path, store=vector_store)
                if self._graph_expander.load():
                    logger.info("Xref Graph 扩展器已就绪")
            except Exception as e:
                logger.warning("Xref Graph 加载失败: %s", e)

    def ask(
        self, query: str, reranker_enabled: bool = True,
        history: list[dict[str, str]] | None = None,
        release: str | None = None,
        series: str | None = None,
        doc_type: str | None = None,
    ) -> RAGResponse:
        """执行完整 RAG 问答.

        Args:
            query: 用户查询
            reranker_enabled: 是否启用 Cross-Encoder 精排 (请求级覆盖全局配置)
            history: 可选的多轮对话历史 [{"role": "user/assistant", "content": "..."}]
            release: Release 过滤, 如 'R18' (仅检索指定 Release 的文档)
            series: Series 过滤, 如 '38' (仅检索指定 Series 的文档)
            doc_type: 文档类型过滤, '3gpp' 或 'oran' (仅检索指定类型的文档)
        """
        # 查询缓存: 相同查询 1 小时内直接返回缓存结果
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            logger.info("查询缓存命中: %s", query[:80])
            return cached

        t_start = time.time()

        # Step 0-3.9: 检索阶段 (翻译/扩展/混合检索/精排/多跳/交叉引用/上下文)
        ctx = self._retrieve_context(
            query, reranker_enabled=reranker_enabled,
            history=history, release=release, series=series, doc_type=doc_type,
        )
        query_lang = ctx["query_lang"]
        search_query = ctx["search_query"]
        expanded_query = ctx["expanded_query"]
        results = ctx["results"]
        release_note = ctx["release_note"]
        online_context = ctx["online_context"]

        if not results:
            record_ask(time.time() - t_start, success=True)
            return RAGResponse(
                query=query,
                answer="未在规范库中找到相关内容。",
                sources=[],
                verified=True,
                warnings=[],
            )

        # Step 3.9: 相邻 chunk 上下文扩展 — 分类列举问题扩大范围
        if _is_taxonomy_query(search_query):
            self._expand_adjacent_chunks(results, top_n=10, window=3)
        else:
            self._expand_adjacent_chunks(results)

        # Step 4: 构建 RAG 提示词 (使用英文检索查询, 保证与检索上下文语言一致)
        # answer_lang: user 消息末尾语言强指令, 避免 DeepSeek 因英文上下文输出英文再回译
        messages = build_rag_prompt(
            search_query, results,
            extra_system_note=release_note,
            online_context=online_context,
            history=history,
            answer_lang=query_lang,
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

        # Step 5.5: 多语言兜底 — 仅当 LLM 输出语言与用户不符时回译 (正常路径零额外调用)
        if not answer or not answer.strip():
            answer = "抱歉，模型未能生成回答，请重试或换个问法。"
        if query_lang != "en":
            answer = _ensure_answer_language(answer, query_lang, self._llm)

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

    def _retrieve_context(
        self,
        query: str,
        reranker_enabled: bool = True,
        history: list[dict[str, str]] | None = None,
        release: str | None = None,
        series: str | None = None,
        doc_type: str | None = None,
    ) -> dict:
        """检索阶段 (Step 0-3.9) — 翻译/扩展/混合检索/精排/多跳/交叉引用/上下文扩展.

        供 ask() 与 ask_stream() 共用, 避免双份逻辑漂移.

        Returns:
            dict: query_lang, search_query, expanded_query, results,
                  release_note, online_context
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
        # 若启用 reranker, 先用大池子检索再精排
        search_top_k = settings.reranker_top_k if reranker_enabled else settings.max_search_results
        old_final = self._retriever._final_top_k
        self._retriever._final_top_k = search_top_k
        results = self._retriever.search(expanded_query, query_embedding, filter_expr=filter_expr)
        self._retriever._final_top_k = old_final

        # Step 3.1: 分类列举查询 → 多角度分解检索
        if _is_taxonomy_query(search_query):
            results = self._taxonomy_decompose_search(
                search_query, query_embedding, results, filter_expr=filter_expr,
            )

        # Spec-aware + Cross-Encoder Reranker 精排
        results = self._post_process_results(
            query=query, expanded_query=expanded_query,
            query_embedding=query_embedding, results=results,
            top_k=settings.max_search_results, reranker_enabled=reranker_enabled,
        )

        # Step 3.1b: 分类列举查询 → 元数据 boost
        if _is_taxonomy_query(search_query):
            results = self._apply_metadata_boost(results)

        # 过滤低信息密度章节 (缩写表/参考文献等)
        results = _filter_low_quality(results, settings.max_search_results)
        search_dt = time.time() - t_search

        # Step 3.2: 多跳检索
        if needs_multi_hop(results):
            record_multi_hop()
            logger.info(
                "触发多跳检索 (多样性=%.2f)",
                len({r.spec_number for r in results if r.spec_number}) / len(results),
            )
            try:
                results = self._multi_hop.search(search_query, query_embedding)
            except Exception as e:
                logger.warning("多跳检索失败, 使用单跳结果: %s", e)
                record_error("multi_hop_failed")

        # Step 3.5: 交叉引用解析 — 优先用离线图扩展，降级为在线二次检索
        if self._graph_expander and self._graph_expander.is_loaded:
            expanded = self._graph_expander.expand(results, max_per_chunk=5, top_n=10)
            if expanded:
                from src.retriever.search import RetrievalResult
                expanded_results = [
                    RetrievalResult.from_search_result(r) for r in expanded
                ]
                results = results + expanded_results
                logger.info("图增强检索: +%d 条 cross-spec chunk", len(expanded))
        else:
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

        return {
            "query_lang": query_lang,
            "search_query": search_query,
            "expanded_query": expanded_query,
            "results": results,
            "release_note": release_note,
            "online_context": online_context,
        }

    def ask_stream(
        self,
        query: str,
        reranker_enabled: bool = True,
        history: list[dict[str, str]] | None = None,
        release: str | None = None,
        series: str | None = None,
        doc_type: str | None = None,
    ):
        """流式问答 — 生成器产出事件元组.

        与 ask() 检索阶段完全一致, 仅 LLM 生成改为逐段流式.

        Events:
            ("sources", list[RetrievalResult]) — 检索结果, 只推送一次
            ("chunk", str)                     — 回答片段, 多次
            ("done", dict)                     — 收尾: answer/verified/warnings/coverage/expanded_query
        """
        cache_key = hashlib.md5(query.lower().strip().encode()).hexdigest()
        cached = self._query_cache.get(cache_key)
        if cached is not None:
            logger.info("查询缓存命中(流式): %s", query[:80])
            yield ("sources", cached.sources)
            for piece in _split_for_stream(cached.answer):
                yield ("chunk", piece)
            yield ("done", {
                "answer": cached.answer,
                "verified": cached.verified,
                "warnings": cached.warnings,
                "coverage": cached.coverage,
                "expanded_query": cached.expanded_query,
            })
            return

        t_start = time.time()
        ctx = self._retrieve_context(
            query, reranker_enabled=reranker_enabled,
            history=history, release=release, series=series, doc_type=doc_type,
        )
        query_lang = ctx["query_lang"]
        search_query = ctx["search_query"]
        expanded_query = ctx["expanded_query"]
        results = ctx["results"]
        release_note = ctx["release_note"]
        online_context = ctx["online_context"]

        if not results:
            yield ("sources", [])
            yield ("done", {
                "answer": "未在规范库中找到相关内容。",
                "verified": True,
                "warnings": [],
                "coverage": 0.0,
                "expanded_query": expanded_query,
            })
            return

        messages = build_rag_prompt(
            search_query, results,
            extra_system_note=release_note,
            online_context=online_context,
            history=history,
            answer_lang=query_lang,
        )

        # 先推送检索结果, 再流式生成回答
        yield ("sources", results)

        # Step 5: LLM 流式生成 — 边生成边推送
        t_llm = time.time()
        parts: list[str] = []
        buffer = ""
        try:
            for delta in self._llm.chat_stream(messages):
                parts.append(delta)
                buffer += delta
                # 按 32 字符粒度推送, 减少事件数量
                while len(buffer) >= 32:
                    yield ("chunk", buffer[:32])
                    buffer = buffer[32:]
        except Exception as e:
            logger.error("流式生成失败: %s", e)
            raise
        if buffer:
            yield ("chunk", buffer)
        answer = "".join(parts)
        # API 空流兜底: DeepSeek 偶发返回空 content, 非流式重试一次
        if not answer.strip():
            logger.warning("流式输出为空, 非流式重试一次")
            try:
                answer = self._llm.chat(messages)
            except Exception as e:
                logger.error("非流式重试失败: %s", e)
        llm_dt = time.time() - t_llm
        record_llm_call(
            prompt_tokens=sum(len(m.get("content", "")) for m in messages) // 4,
            completion_tokens=len(answer) // 4,
            duration_s=llm_dt,
        )

        # Step 5.5: 多语言兜底 — 仅当 LLM 输出语言与用户不符时回译 (正常路径零额外调用)
        if not answer or not answer.strip():
            answer = "抱歉，模型未能生成回答，请重试或换个问法。"
        if query_lang != "en":
            answer = _ensure_answer_language(answer, query_lang, self._llm)

        # Step 6: 答案验证
        verification = self._verifier.verify(answer, results)
        record_ask(time.time() - t_start)

        response = RAGResponse(
            query=query,
            answer=answer,
            sources=results,
            verified=verification["verified"],
            warnings=verification["warnings"],
            coverage=verification["coverage"],
            expanded_query=expanded_query,
        )
        self._query_cache[cache_key] = response
        yield ("done", {
            "answer": answer,
            "verified": verification["verified"],
            "warnings": verification["warnings"],
            "coverage": verification["coverage"],
            "expanded_query": expanded_query,
        })

    def search(self, query: str, top_k: int | None = None, reranker_enabled: bool = True,
               release: str | None = None, series: str | None = None, doc_type: str | None = None) -> list[RetrievalResult]:
        """仅执行检索 (不生成答案) — 含 spec-aware + Cross-Encoder 重排序.

        Phase 1: 常规混合检索 (Dense+BM25+RRF).
        Phase 2: 若 LLM 扩展查询中包含规范号, 对提示规范做定向检索并合并.
        Phase 3: Cross-Encoder Reranker 对候选精排 (可选启用).

        Args:
            query: 用户查询
            top_k: 返回条数
            reranker_enabled: 是否启用 Cross-Encoder 精排 (请求级覆盖全局配置)
            release: Release 过滤, 如 'R18'
            series: Series 过滤, 如 '38'
            doc_type: 文档类型过滤, '3gpp' 或 'oran'
        """
        _top_k = top_k or settings.max_search_results
        # 若启用 reranker, 混合检索用更大的候选池 (reranker 从 N 条中精选 top_k)
        pool_size = settings.reranker_top_k if reranker_enabled else _top_k
        if top_k is not None:
            self._retriever._final_top_k = top_k
        else:
            self._retriever._final_top_k = pool_size
        expanded_query = self._expand_query(query)
        query_embedding = self._get_query_embedding(expanded_query)
        filter_expr = self._build_filter_expr(release=release, series=series, doc_type=doc_type)
        results = self._retriever.search(expanded_query, query_embedding, filter_expr=filter_expr)

        # Cross-Encoder Reranker 精排 (先执行, 提升候选池质量)
        results = self._rerank(expanded_query, results, _top_k, reranker_enabled=reranker_enabled)

        # Spec-aware 两阶段检索: 对 LLM 识别出的规范号做定向补充+加权 (最后执行, 确保覆盖)
        spec_hints = _extract_spec_numbers(expanded_query)
        if spec_hints:
            results = _spec_aware_rerank(
                results, spec_hints, self._retriever._store,
                query_embedding, _top_k,
            )

        return results

    def _expand_query(self, query: str, history: list[dict[str, str]] | None = None) -> str:
        try:
            messages = build_query_expansion_prompt(query, history)
            expanded = self._llm.chat(messages)
            if expanded and len(expanded) > 5:
                return expanded
        except Exception as e:
            logger.warning("查询扩展失败: %s", e)
        return query

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
        try:
            user_content = f"用户查询: {query}"
            if history and len(history) >= 2:
                history_lines = ["## 对话历史 (用于理解当前问题的上下文)"]
                for msg in history[-6:]:
                    role = "用户" if msg["role"] == "user" else "助手"
                    history_lines.append(f"{role}: {msg['content'][:200]}")
                history_lines.append(f"\n## 当前问题 (需要翻译+扩展为英文检索查询)\n{query}")
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
                return translated, expanded
            logger.warning("合并解析失败, 回退串行路径: %.80s", result[:80])
        except Exception as e:
            logger.warning("合并翻译+扩展失败, 回退串行路径: %s", e)
        # 回退: 原串行路径 (翻译 → 扩展), 保证质量不降级
        search_query = translate_to_english(query, source_lang, self._llm)
        return search_query, self._expand_query(search_query, history)

    def _post_process_results(
        self,
        query: str,
        expanded_query: str,
        query_embedding: np.ndarray,
        results: list[RetrievalResult],
        top_k: int,
        reranker_enabled: bool = True,
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
                query_embedding, top_k,
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

        # 保存原始分数并归一化到 [0, 1]
        import copy
        orig_scores = np.array([r.score for r in results], dtype=np.float32)
        orig_min, orig_max = orig_scores.min(), orig_scores.max()
        if orig_max > orig_min:
            orig_norm = (orig_scores - orig_min) / (orig_max - orig_min)
        else:
            orig_norm = np.ones_like(orig_scores) * 0.5

        try:
            # 获取 Cross-Encoder 分数 (reranker 内部会设置 result.score)
            reranked = reranker.rerank(query, results, top_k=len(results))
        except Exception as e:
            logger.warning("Reranker 失败, 降级返回原始候选: %s", e)
            return results[:top_k]

        # 归一化 reranker 分数到 [0, 1]
        rerank_scores = np.array([r.score for r in reranked], dtype=np.float32)
        rerank_min, rerank_max = rerank_scores.min(), rerank_scores.max()
        if rerank_max > rerank_min:
            rerank_norm = (rerank_scores - rerank_min) / (rerank_max - rerank_min)
        else:
            rerank_norm = np.ones_like(rerank_scores) * 0.5

        # 加权融合: reranker 60% + 原始 40%
        ALPHA = 0.6
        combined = ALPHA * rerank_norm + (1 - ALPHA) * orig_norm

        # 写回融合分数并排序
        for r, s in zip(reranked, combined):
            r.score = float(s)
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

    def _get_query_embedding(self, query: str) -> np.ndarray:
        try:
            embeddings = self._llm.embed([query])
            return np.array(embeddings[0], dtype=np.float32)
        except Exception as e:
            logger.warning("嵌入生成失败，使用零向量: %s", e)
            return np.zeros(1024, dtype=np.float32)

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

        for sub_q in sub_queries:
            try:
                sub_embed = self._get_query_embedding(sub_q)
                # 子查询用较小的 top_k, 避免单子查询占据全部上下文
                old_final = self._retriever._final_top_k
                self._retriever._final_top_k = max(settings.max_search_results // 2, 5)
                sub_results = self._retriever.search(sub_q, sub_embed, filter_expr=filter_expr)
                self._retriever._final_top_k = old_final

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
            response = self._llm.chat(messages, temperature=0.3, max_tokens=1024)
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

    def _expand_adjacent_chunks(
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
                parts.append(f'release == "{release}"')
            if series:
                parts.append(f"series == {series}")
            parts.append('doc_type == "3gpp"')
            return " && ".join(parts)

        # doc_type 未选: release/series 只约束 3GPP, ORAN 放行
        if has_filter:
            parts_3gpp: list[str] = []
            if release:
                parts_3gpp.append(f'release == "{release}"')
            if series:
                parts_3gpp.append(f"series == {series}")
            parts_3gpp.append('doc_type == "3gpp"')
            return f'(({" && ".join(parts_3gpp)}) || doc_type == "oran")'

        return None

    def _warmup(self) -> None:
        """预热嵌入模型，避免首次查询加载延迟."""
        try:
            logger.info("预热嵌入模型...")
            self._get_query_embedding("warmup")
            logger.info("嵌入模型预热完成")
        except Exception as e:
            logger.warning("嵌入模型预热失败: %s", e)


# ── Spec-aware 重排序辅助函数 ──

_SPEC_PATTERN = __import__("re").compile(r'\b(\d{2}\.\d{3})\b')


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
) -> list["RetrievalResult"]:
    """两阶段 spec-aware 重排序.

    1. 对每个 hint spec 做定向 Dense 检索 (最多 3 条/spec), 补充到结果列表.
    2. 对匹配 hint spec 的结果分数加权 (×boost_factor).
    3. 按分数降序排列返回.
    """
    import copy
    from src.retriever.search import RetrievalResult

    seen_ids: set = {str(r.chunk_id) for r in results}  # 统一 str 避免 int/str 混合导致去重失效
    supplement: list["RetrievalResult"] = []

    # Phase 2: 定向检索 hint specs (用 Milvus 标量过滤)
    for spec in list(spec_hints)[:3]:  # 最多 3 个 hint spec
        try:
            from src.retriever.milvus_store import _escape_milvus_expr
            escaped_spec = _escape_milvus_expr(spec)
            spec_results = store.search_dense(
                query_embedding, top_k=5,
                filter_expr=f'spec_number == "{escaped_spec}"',
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


# ── 低质量章节过滤 ──

# 低信息密度章节: parent_title 命中直接过滤
_LOW_QUALITY_TITLES = {"Abbreviations", "Definitions", "Symbols", "References"}

# 结构性垃圾文本特征: prefix 匹配命中直接过滤 (目录/Foreword/Scope/Reference)
_STRUCTURAL_PREFIXES = (
    "#  Contents", "#  Foreword", "#  1 Scope", "# 1 Scope",
    "#  2 References", "# 2 References",
)

# 低信息密度 section_id: 协议规范中前几个 section 通常是 Scope/Refs/Defs 结构章节
_LOW_INFO_SECTIONS = ("1", "2", "3")  # 主章节号, 非子章节


def _is_low_quality(r: "RetrievalResult") -> bool:
    """判断单个结果是否为低质量 (低信息密度) chunks."""
    text = (r.text or "").strip()
    title = r.parent_title or ""
    # 规则1: parent_title 子串命中低质量章节名 (如 "3.3 Abbreviations")
    if title:
        title_lower = title.lower()
        for kw in ("abbreviation", "definition", "symbol", "reference"):
            if kw in title_lower and not any(
                op in title_lower
                for op in ("operation", "procedure", "function", "configure", "establish")
            ):
                return True
    # 规则2: 文本前缀命中结构性垃圾 (目录/Foreword/Scope/References)
    if text.startswith(_STRUCTURAL_PREFIXES):
        return True
    # 规则3: 短文本仅包含目录/缩写关键词
    if len(text) < 50 and ("Contents" in text or "Foreword" in text or "Scope" in text):
        return True
    # 规则4: parent_section_id 命中 Scope/Refs/Defs 纯顶层章节号
    sid = r.parent_section_id or ""
    if sid in _LOW_INFO_SECTIONS and title:
        low_title_parts = {"scope", "reference", "definition", "abbreviation", "symbol"}
        if any(p in title_lower for p in low_title_parts):
            return True
    return False


def _filter_low_quality(results: list["RetrievalResult"], target_k: int) -> list["RetrievalResult"]:
    """过滤低信息密度章节 (缩写表/符号表/目录/参考文献), 保留 target_k 条高质量结果."""
    quality: list["RetrievalResult"] = []
    for r in results:
        if not _is_low_quality(r):
            quality.append(r)
            if len(quality) >= target_k:
                break
    # 如果过滤后不够, 从低质量中补充 (排在末尾)
    if len(quality) < target_k:
        for r in results:
            if _is_low_quality(r) and r not in quality and len(quality) < target_k:
                quality.append(r)
    return quality
