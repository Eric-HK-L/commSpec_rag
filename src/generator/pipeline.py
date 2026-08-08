"""RAG 问答流水线 — 薄编排层.

职责: 检索规划 (委托 RetrievalPlanner) → RAG 提示词 → LLM 生成 → 答案验证.
全部检索策略实现隐藏在 src.retriever.planner 深模块中.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from cachetools import TTLCache

from src.generator.i18n import translate_from_english
from src.generator.llm_client import LLMClient
from src.generator.prompt import build_rag_prompt, is_taxonomy_query
from src.generator.verifier import AnswerVerifier
from src.retriever.planner import RetrievalContext, RetrievalPlanner
from src.retriever.search import RetrievalResult
from src.retriever.vector_store import VectorStore
from src.utils.monitoring import record_ask, record_llm_call

logger = logging.getLogger(__name__)


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


def _build_cache_key(
    query: str,
    *,
    release: str | None = None,
    series: str | None = None,
    doc_type: str | None = None,
    reranker_enabled: bool = True,
    history: list[dict[str, str]] | None = None,
) -> str:
    """生成查询缓存 Key — 包含全部影响结果的参数, 避免跨过滤条件串缓存."""
    parts = [
        query.lower().strip(),
        str(release or ""),
        str(series or ""),
        str(doc_type or ""),
        str(reranker_enabled),
    ]
    if history:
        parts.extend(f"{m.get('role', '')}:{m.get('content', '')}" for m in history)
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


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
    """CommSpec RAG 查询流水线 — 薄编排层.

    流程: 检索规划 (RetrievalPlanner) -> RAG 提示词 -> LLM 生成 -> 答案验证
    """

    def __init__(
        self,
        vector_store: VectorStore,
        llm_client: LLMClient | None = None,
    ):
        self._store = vector_store
        self._llm = llm_client or LLMClient()
        self._planner = RetrievalPlanner(
            vector_store=vector_store,
            llm_client=self._llm,
        )
        self._verifier = AnswerVerifier()
        # 查询级 LRU 缓存: TTL 1 小时, 最多 256 条, key=md5(query)
        self._query_cache: TTLCache = TTLCache(maxsize=256, ttl=3600)

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
        cache_key = _build_cache_key(
            query,
            release=release,
            series=series,
            doc_type=doc_type,
            reranker_enabled=reranker_enabled,
            history=history,
        )
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
        query_lang = ctx.query_lang
        search_query = ctx.search_query
        expanded_query = ctx.expanded_query
        results = ctx.results
        release_note = ctx.release_note
        online_context = ctx.online_context

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
        if is_taxonomy_query(search_query):
            self._planner.expand_adjacent_chunks(results, top_n=10, window=3)
        else:
            self._planner.expand_adjacent_chunks(results)

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
    ) -> RetrievalContext:
        """检索阶段委托 — 供 ask() 与 ask_stream() 共用, 保留可替换接缝."""
        return self._planner.plan(
            query, reranker_enabled=reranker_enabled,
            history=history, release=release, series=series, doc_type=doc_type,
        )

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
        cache_key = _build_cache_key(
            query,
            release=release,
            series=series,
            doc_type=doc_type,
            reranker_enabled=reranker_enabled,
            history=history,
        )
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
        query_lang = ctx.query_lang
        search_query = ctx.search_query
        expanded_query = ctx.expanded_query
        results = ctx.results
        release_note = ctx.release_note
        online_context = ctx.online_context

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
        """仅执行检索 (不生成答案) — 委托检索规划器.

        Args:
            query: 用户查询
            top_k: 返回条数
            reranker_enabled: 是否启用 Cross-Encoder 精排 (请求级覆盖全局配置)
            release: Release 过滤, 如 'R18'
            series: Series 过滤, 如 '38'
            doc_type: 文档类型过滤, '3gpp' 或 'oran'
        """
        return self._planner.search(
            query, top_k=top_k, reranker_enabled=reranker_enabled,
            release=release, series=series, doc_type=doc_type,
        )

    def _warmup(self) -> None:
        """预热嵌入模型，避免首次查询加载延迟."""
        try:
            logger.info("预热嵌入模型...")
            self._planner._get_query_embedding("warmup")
            logger.info("嵌入模型预热完成")
        except Exception as e:
            logger.warning("嵌入模型预热失败: %s", e)


# ── 低质量章节过滤: 统一由 src.retriever.result_quality 提供 (filter_low_quality) ──
