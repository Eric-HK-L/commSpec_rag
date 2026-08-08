"""Chunk 序列化 round-trip 测试 — to_dict / from_dict 局部性保障."""

from __future__ import annotations

import json

import numpy as np

from src.retriever.vector_store import Chunk


def _full_chunk() -> Chunk:
    """所有字段均显式赋值的 Chunk."""
    return Chunk(
        text="The PDU Session Resource Setup procedure...",
        embedding=np.array([0.1, 0.2, 0.3]),
        doc_id="ts38413-v18.0.0",
        series=38,
        spec_number="38.413",
        release="R18",
        parent_section_id="8.3.1",
        parent_title="PDU Session Resource Setup",
        chunk_index=42,
        section_number="8.3.1",
        section_title="PDU Session Resource Setup",
        section_path="8 Procedures > 8.3 Setup > 8.3.1 PDU Session Resource Setup",
        doc_type="3gpp",
        content_type="procedure",
        spec_role="authoritative",
        topic_domain="ran_arch",
    )


class TestChunkToDict:
    def test_all_fields_present_except_embedding(self):
        d = _full_chunk().to_dict()
        assert "embedding" not in d
        assert d["text"].startswith("The PDU Session")
        assert d["series"] == 38
        assert d["topic_domain"] == "ran_arch"

    def test_covers_every_non_embedding_field(self):
        """新增字段后此测试自动覆盖 — 防止序列化漏字段."""
        from dataclasses import fields
        d = _full_chunk().to_dict()
        expected = {f.name for f in fields(Chunk)} - {"embedding"}
        assert set(d.keys()) == expected

    def test_json_serializable(self):
        json.dumps(_full_chunk().to_dict(), ensure_ascii=False)


class TestChunkRoundTrip:
    def test_full_round_trip(self):
        original = _full_chunk()
        restored = Chunk.from_dict(original.to_dict())
        assert restored.embedding is None  # 向量不参与磁盘序列化
        original.embedding = None
        assert restored == original

    def test_unknown_fields_tolerated(self):
        """旧数据含未来字段 / 新代码读旧数据时不崩溃."""
        d = _full_chunk().to_dict()
        d["future_field"] = "ignored"
        restored = Chunk.from_dict(d)
        assert restored.spec_number == "38.413"

    def test_missing_fields_use_defaults(self):
        """旧版本 interim 数据缺新字段时用默认值."""
        restored = Chunk.from_dict({"text": "legacy chunk", "doc_id": "old"})
        assert restored.text == "legacy chunk"
        assert restored.doc_type == "3gpp"  # 默认值
        assert restored.content_type == ""

    def test_embedding_in_dict_ignored(self):
        """即使 dict 误含 embedding 键也不会注入."""
        d = _full_chunk().to_dict()
        d["embedding"] = [1.0, 2.0]
        restored = Chunk.from_dict(d)
        assert restored.embedding is None


class TestOrchestratorInterimCompatibility:
    """orchestrator interim 落盘格式与 Chunk 序列化一致."""

    def test_save_load_via_chunk_methods(self, tmp_path):
        chunks = [_full_chunk(), Chunk(text="minimal")]
        path = tmp_path / "chunks.json"
        with open(path, "w") as f:
            json.dump([c.to_dict() for c in chunks], f, ensure_ascii=False)
        with open(path) as f:
            restored = [Chunk.from_dict(d) for d in json.load(f)]
        assert restored[0].spec_number == "38.413"
        assert restored[1].text == "minimal"
        assert restored[1].chunk_index == 0
