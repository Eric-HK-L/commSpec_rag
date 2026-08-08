"""流式问答与语言兜底测试 — chat_stream / ask_stream / 语言兜底 / 合并翻译扩展."""

from unittest.mock import MagicMock

from src.generator.llm_client import LLMClient
from src.generator.pipeline import (
    RAGPipeline,
    _ensure_answer_language,
    _split_for_stream,
)
from src.generator.prompt import build_rag_prompt
from src.retriever.planner import RetrievalContext
from src.retriever.search import RetrievalResult

# ── chat_stream ──


class TestChatStream:
    def test_has_chat_stream_method(self):
        client = LLMClient()
        assert hasattr(client, "chat_stream")
        assert callable(client.chat_stream)

    def test_chat_stream_yields_content_deltas(self):
        """Mock OpenAI 流式响应 → 逐段产出 content."""
        class _Delta:
            def __init__(self, content):
                self.content = content

        class _Choice:
            def __init__(self, content):
                self.delta = _Delta(content)

        class _Chunk:
            def __init__(self, content):
                self.choices = [_Choice(content)] if content is not None else []

        stream = iter([
            _Chunk(None),       # 无 choices
            _Chunk("你好"),
            _Chunk("，世界"),
            _Chunk(""),
        ])
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = stream

        client = LLMClient()
        client._client = mock_client  # 覆盖 __init__ 真实 client
        parts = list(client.chat_stream([{"role": "user", "content": "hi"}]))
        assert parts == ["你好", "，世界"]
        # stream=True 必须被传递
        _, kwargs = mock_client.chat.completions.create.call_args
        assert kwargs["stream"] is True

    def test_chat_stream_empty_choices_skipped(self):
        class _Chunk:
            def __init__(self):
                self.choices = []

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = iter([_Chunk(), _Chunk()])
        client = LLMClient()
        client._client = mock_client
        assert list(client.chat_stream([{"role": "user", "content": "x"}])) == []


# ── _split_for_stream ──


class TestSplitForStream:
    def test_splits_by_size(self):
        assert _split_for_stream("abcdefgh", 3) == ["abc", "def", "gh"]

    def test_empty_text(self):
        assert _split_for_stream("") == []

    def test_exact_multiple(self):
        assert _split_for_stream("abcdef", 3) == ["abc", "def"]

    def test_default_size_32(self):
        pieces = _split_for_stream("x" * 100)
        assert pieces[0] == "x" * 32
        assert "".join(pieces) == "x" * 100


# ── _ensure_answer_language ──


class TestEnsureAnswerLanguage:
    def test_zh_ratio_above_threshold_returns_unchanged(self):
        """中文占比 ≥5% → 直接返回, 不触发回译 (零额外 LLM 调用)."""
        llm = MagicMock()
        answer = "根据 TS 38.305 定义，SLPP 过程包含多种类型。"
        result = _ensure_answer_language(answer, "zh", llm)
        assert result == answer
        llm.chat.assert_not_called()

    def test_english_answer_triggers_backtranslation(self):
        llm = MagicMock()
        llm.chat.return_value = "这是一个用于测试的中文回译结果内容。"
        result = _ensure_answer_language("This is a pure English answer.", "zh", llm)
        assert result == "这是一个用于测试的中文回译结果内容。"
        llm.chat.assert_called_once()

    def test_target_en_returns_unchanged(self):
        llm = MagicMock()
        result = _ensure_answer_language("English answer", "en", llm)
        assert result == "English answer"
        llm.chat.assert_not_called()

    def test_empty_answer_triggers_fallback(self):
        llm = MagicMock()
        llm.chat.return_value = "这是空回答时的兜底翻译结果。"
        result = _ensure_answer_language("", "zh", llm)
        assert result == "这是空回答时的兜底翻译结果。"


# ── build_rag_prompt answer_lang ──


class TestAnswerLangInstruction:
    def _chunk(self) -> RetrievalResult:
        return RetrievalResult(
            chunk_id=1, text="PDU Session is a logical association.",
            score=0.9, spec_number="38300", parent_section_id="s1",
            parent_title="5G System",
        )

    def test_zh_instruction_appended(self):
        messages = build_rag_prompt("test query", [self._chunk()], answer_lang="zh")
        user_content = messages[1]["content"]
        assert "【输出语言】必须使用简体中文回答" in user_content

    def test_ko_instruction_appended(self):
        messages = build_rag_prompt("test query", [self._chunk()], answer_lang="ko")
        user_content = messages[1]["content"]
        assert "【출력 언어】반드시 한국어로 답변하세요" in user_content

    def test_unknown_lang_no_instruction(self):
        messages = build_rag_prompt("test query", [self._chunk()], answer_lang="fr")
        user_content = messages[1]["content"]
        assert "【输出语言】" not in user_content

    def test_default_no_instruction(self):
        messages = build_rag_prompt("test query", [self._chunk()])
        user_content = messages[1]["content"]
        assert "【输出语言】" not in user_content


# ── ask_stream 事件序列 ──


class TestAskStream:
    def _make_pipeline(self, llm: MagicMock) -> RAGPipeline:
        pipe = RAGPipeline(vector_store=MagicMock())
        pipe._llm = llm
        return pipe

    def test_cache_hit_streams_sources_chunks_done(self):
        """缓存命中: sources → chunks (切片) → done, 不触发 LLM."""
        from src.generator.pipeline import RAGResponse
        from src.retriever.search import RetrievalResult

        r = RetrievalResult(
            chunk_id=1, text="ctx", score=0.9, spec_number="38300",
            parent_section_id="s1", parent_title="t",
        )
        cached = RAGResponse(
            query="q", answer="缓存回答内容", sources=[r],
            verified=True, warnings=[], coverage=1.0, expanded_query="eq",
        )
        llm = MagicMock()
        pipe = self._make_pipeline(llm)
        pipe._query_cache.clear()

        from src.generator.pipeline import _build_cache_key
        key = _build_cache_key("q")
        pipe._query_cache[key] = cached

        events = list(pipe.ask_stream("q"))
        types = [e[0] for e in events]
        assert types[0] == "sources"
        assert all(t == "chunk" for t in types[1:-1])
        assert types[-1] == "done"
        assert "".join(e[1] for e in events if e[0] == "chunk") == "缓存回答内容"
        assert events[-1][1]["answer"] == "缓存回答内容"
        llm.chat.assert_not_called()

    def test_stream_uses_chat_stream_and_buffers(self):
        """正常路径: mock chat_stream 产出 → chunk 事件按 32 字符粒度推送."""
        from src.retriever.search import RetrievalResult

        llm = MagicMock()
        # 检索上下文 mock: 返回一条结果
        llm.chat.side_effect = ["What is PDU session?", "PDU session concepts"]  # 翻译+扩展
        llm.chat_stream.return_value = iter(["x" * 80])  # 一次产出 80 字符

        pipe = self._make_pipeline(llm)
        pipe._retrieve_context = MagicMock(return_value=RetrievalContext(
            query_lang="en",
            search_query="What is PDU session?",
            expanded_query="PDU session concepts",
            results=[
                RetrievalResult(
                    chunk_id=1, text="PDU session is a logical association between UE and network.",
                    score=0.9, spec_number="38300", parent_section_id="s1", parent_title="5G System",
                )
            ],
            release_note="",
            online_context="",
        ))

        events = list(pipe.ask_stream("What is PDU session?"))
        types = [e[0] for e in events]
        assert types[0] == "sources"
        assert types[-1] == "done"
        chunks = [e[1] for e in events if e[0] == "chunk"]
        # 80 字符 → 2 个 32 字符 chunk + 16 字符尾块
        assert len(chunks) == 3
        assert "".join(chunks) == "x" * 80
        # done.answer 与流式拼接一致
        assert events[-1][1]["answer"] == "x" * 80

    def test_empty_stream_retries_non_stream(self):
        """空流兜底: chat_stream 无产出 → 非流式重试一次."""
        from src.retriever.search import RetrievalResult

        llm = MagicMock()
        llm.chat_stream.return_value = iter([])  # 空流
        llm.chat.return_value = "重试生成的中文回答"

        pipe = self._make_pipeline(llm)
        pipe._retrieve_context = MagicMock(return_value=RetrievalContext(
            query_lang="zh",
            search_query="SLPP procedure",
            expanded_query="SLPP procedure types",
            results=[
                RetrievalResult(
                    chunk_id=1, text="SLPP procedure types.",
                    score=0.9, spec_number="38305", parent_section_id="s1", parent_title="t",
                )
            ],
            release_note="",
            online_context="",
        ))

        events = list(pipe.ask_stream("SLPP 过程有哪些类型？"))
        done = events[-1][1]
        assert done["answer"] == "重试生成的中文回答"
        # 仅触发非流式重试一次 (检索上下文已 mock, 翻译+扩展不产生 chat 调用)
        assert llm.chat.call_count == 1

    def test_no_results_yields_done_with_fallback_answer(self):
        llm = MagicMock()
        pipe = self._make_pipeline(llm)
        pipe._retrieve_context = MagicMock(return_value=RetrievalContext(
            query_lang="zh",
            search_query="nonexistent",
            expanded_query="nonexistent",
            results=[],
            release_note="",
            online_context="",
        ))
        events = list(pipe.ask_stream("找不到的内容"))
        types = [e[0] for e in events]
        assert types == ["sources", "done"]
        assert "未在规范库中找到" in events[-1][1]["answer"]
        llm.chat_stream.assert_not_called()


# ── _translate_and_expand ──


class TestTranslateAndExpand:
    def _make_pipeline(self) -> RAGPipeline:
        return RAGPipeline(vector_store=MagicMock())

    def test_success_parses_two_lines(self):
        llm = MagicMock()
        llm.chat.return_value = "TRANSLATED: What are the SLPP procedure types in TS 38.305?\nEXPANDED: SLPP procedure types sidelink positioning protocol 38.305"
        pipe = self._make_pipeline()
        pipe._planner._llm = llm
        translated, expanded = pipe._planner._translate_and_expand("38.305 中 SLPP 过程类型有哪些？", "zh")
        assert "What are the SLPP procedure types" in translated
        assert "sidelink positioning protocol" in expanded

    def test_parse_failure_falls_back_to_serial(self):
        """合并解析失败 → 回退原串行路径 (翻译→扩展), 质量不降级."""
        llm = MagicMock()
        llm.chat.side_effect = [
            "Some random output without markers",      # 合并调用结果 (无效)
            "What is SLPP in TS 38.305?",              # 回退: 翻译
            "SLPP positioning protocol procedure",     # 回退: 扩展
        ]
        pipe = self._make_pipeline()
        pipe._planner._llm = llm
        translated, expanded = pipe._planner._translate_and_expand("38.305 中 SLPP 是什么？", "zh")
        assert translated == "What is SLPP in TS 38.305?"
        assert expanded == "SLPP positioning protocol procedure"
        assert llm.chat.call_count == 3

    def test_exception_falls_back(self):
        llm = MagicMock()
        llm.chat.side_effect = [
            RuntimeError("API error"),
            "What is SLPP in TS 38.305?",
            "SLPP procedure",
        ]
        pipe = self._make_pipeline()
        pipe._planner._llm = llm
        translated, expanded = pipe._planner._translate_and_expand("38.305 中 SLPP 是什么？", "zh")
        assert translated == "What is SLPP in TS 38.305?"
        assert expanded == "SLPP procedure"
