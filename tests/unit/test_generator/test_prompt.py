"""prompt.py 单元测试 — 不依赖 LLM / Milvus."""

from src.generator.prompt import build_query_expansion_prompt, build_rag_prompt
from src.retriever.search import RetrievalResult


def _make_chunk(
    chunk_id="1", text="test", spec="38300", score=0.8, doc_id="doc1",
    series=38, release="R18", section="8.3.1", title="Test", idx=0,
):
    return RetrievalResult(
        chunk_id=chunk_id, text=text, score=score,
        doc_id=doc_id, series=series, spec_number=spec, release=release,
        parent_section_id=section, parent_title=title, chunk_index=idx,
    )


class TestBuildRagPrompt:

    def test_basic_structure(self):
        chunks = [_make_chunk(text="PDU Session is defined in TS 23.501.")]
        msgs = build_rag_prompt("What is PDU Session?", chunks)

        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "通信规范专家助手（3GPP / O-RAN）" in msgs[0]["content"]
        assert "What is PDU Session?" in msgs[1]["content"]
        assert "PDU Session is defined" in msgs[1]["content"]

    def test_max_context_chunks_truncation(self):
        chunks = [_make_chunk(chunk_id=str(i), text=f"chunk {i}") for i in range(15)]
        msgs = build_rag_prompt("query", chunks, max_context_chunks=5)
        user = msgs[1]["content"]

        assert "chunk 0" in user
        assert "chunk 4" in user
        assert "chunk 5" not in user  # 第6个被截断

    def test_extra_system_note(self):
        chunks = [_make_chunk()]
        note = "⚠️ 仅参考 R18 版本"
        msgs = build_rag_prompt("query", chunks, extra_system_note=note)
        assert note in msgs[0]["content"]

    def test_online_context_merged(self):
        chunks = [_make_chunk(text="offline chunk")]
        online = "## 在线补充\nGoogle: TS 38.300 describes..."
        msgs = build_rag_prompt("query", chunks, online_context=online)

        user = msgs[1]["content"]
        assert "在线补充" in user
        assert "离线检索结果" in user
        assert "offline chunk" in user
        # 在线内容应排在离线之前
        assert user.index("在线补充") < user.index("离线检索结果")

    def test_empty_chunks(self):
        msgs = build_rag_prompt("query", [])
        assert "query" in msgs[1]["content"]
        # 不应该崩溃

    def test_no_online_context(self):
        chunks = [_make_chunk()]
        msgs = build_rag_prompt("query", chunks, online_context="")
        assert "离线检索结果" not in msgs[1]["content"]


class TestBuildQueryExpansionPrompt:

    def test_structure(self):
        msgs = build_query_expansion_prompt("PDU session setup")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert "查询优化器" in msgs[0]["content"]
        assert "PDU session setup" in msgs[1]["content"]


class TestBuildRagPromptParentContext:
    """small-to-big: 上下文组装附带父 section 文本 (控制总量 1500 chars)."""

    def test_parent_context_rendered(self):
        chunk = RetrievalResult(
            chunk_id=1, text="sub chunk detail", score=0.9,
            doc_id="d", series=38, spec_number="38.413", release="R18",
            parent_section_id="5.3", parent_title="RRC Setup",
            parent_context="Parent section full text for small-to-big context",
        )
        user = build_rag_prompt("query", [chunk])[1]["content"]
        assert "Parent section full text" in user
        assert "父章节上下文" in user

    def test_no_parent_context_no_extra_block(self):
        chunk = _make_chunk()
        user = build_rag_prompt("query", [chunk])[1]["content"]
        assert "父章节上下文" not in user

    def test_parent_context_capped_at_1500(self):
        chunk = RetrievalResult(
            chunk_id=1, text="sub chunk", score=0.9,
            doc_id="d", series=38, spec_number="38.413", release="R18",
            parent_section_id="5.3", parent_title="RRC Setup",
            parent_context="x" * 5000,
        )
        user = build_rag_prompt("query", [chunk])[1]["content"]
        # 上下文总量受控: 父文本仅注入前 1500 chars
        assert user.count("x" * 100) <= 15
