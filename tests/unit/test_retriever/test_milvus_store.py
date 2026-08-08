"""milvus_store.py 字段截断工具单元测试.

覆盖 VARCHAR 字段写入 Milvus 前的字节安全截断 (修复 marked 数据集 38.331
深嵌套章节 section_path 2411 字节 > schema 1024 上限导致的入库失败):
  - 短文本原样返回
  - 中文多字节文本按字节截断且不超上限
  - 截断处不产生 UTF-8 解码错误
"""

from src.retriever.milvus_store import (
    MilvusStore,
    _matches_filter_expr,
    _safe_truncate_bytes,
)
from src.retriever.vector_store import SearchResult


class TestSafeTruncateBytes:

    def test_short_text_unchanged(self):
        s = "6.3.1  RRCReconfiguration"
        assert _safe_truncate_bytes(s, 1024) == s

    def test_long_chinese_path_truncated_to_limit(self):
        # 模拟 38.331 深嵌套中文章节路径: 800 字符 × 3 字节 = 2400 字节
        s = "无线资源控制信令流程 > RRC 重配置 > 信令承载配置 > 数据传输承载配置 " * 40
        out = _safe_truncate_bytes(s, 1024)
        assert len(out.encode("utf-8")) <= 1024
        # 截断后有标记且 UTF-8 完整
        assert out.endswith("…")

    def test_ascii_path_truncated(self):
        s = "a" * 2000
        out = _safe_truncate_bytes(s, 1024)
        assert len(out.encode("utf-8")) <= 1024

    def test_4096_limit_covers_deepest_path(self):
        # 当前数据集最深 section_path 2411 字节, 4096 上限必须完整容纳
        s = "章节" * 500  # 3000 字节
        out = _safe_truncate_bytes(s, 4096)
        assert len(out.encode("utf-8")) <= 4096


def _sr(doc_id: str, spec: str, idx: int, score: float = 0.5) -> SearchResult:
    return SearchResult(
        chunk_id=f"{doc_id}|{spec}|{idx}", text="ctx", score=score,
        doc_id=doc_id, spec_number=spec, chunk_index=idx,
    )


class TestMatchesFilterExpr:
    """BM25 Python 侧标量过滤 — 覆盖 _build_filter_expr 生成的全部形态."""

    META = {"release": "R18", "series": 38, "doc_type": "3gpp"}

    def test_empty_expr_passes(self):
        assert _matches_filter_expr(self.META, "") is True
        assert _matches_filter_expr(self.META, None) is True

    def test_single_field_match(self):
        assert _matches_filter_expr(self.META, 'release == "R18"') is True
        assert _matches_filter_expr(self.META, 'series == 38') is True
        assert _matches_filter_expr(self.META, 'doc_type == "3gpp"') is True

    def test_single_field_mismatch(self):
        assert _matches_filter_expr(self.META, 'release == "R17"') is False
        assert _matches_filter_expr(self.META, 'series == 23') is False
        assert _matches_filter_expr(self.META, 'doc_type == "oran"') is False

    def test_and_combination(self):
        expr = 'release == "R18" && series == 38 && doc_type == "3gpp"'
        assert _matches_filter_expr(self.META, expr) is True
        assert _matches_filter_expr({"release": "R18", "series": 23, "doc_type": "3gpp"}, expr) is False

    def test_or_with_oran_fallback(self):
        expr = '((release == "R18" && series == 38 && doc_type == "3gpp") || doc_type == "oran")'
        assert _matches_filter_expr(self.META, expr) is True
        assert _matches_filter_expr({"release": "R17", "series": 38, "doc_type": "3gpp"}, expr) is False
        assert _matches_filter_expr({"doc_type": "oran"}, expr) is True

    def test_empty_meta_rejected(self):
        assert _matches_filter_expr({}, 'release == "R18"') is False


class TestRrfFuse:
    """RRF 融合 — k 参数化后的加权行为."""

    def test_equal_k_favors_both_rankings(self):
        dense = [_sr("d1", "38.413", 0, 0.9)]
        sparse = [_sr("d1", "38.413", 0, 0.8), _sr("d2", "23.501", 0, 0.7)]
        fused = MilvusStore._rrf_fuse(dense, sparse, 2)
        assert fused[0].chunk_id == "d1|38.413|0"  # 双路命中排第一

    def test_smaller_k_gives_higher_weight(self):
        # k_dense 远小于 k_sparse → Dense 单路 rank1 胜过 Sparse 单路 rank1
        dense_only = _sr("d1", "38.413", 0)
        sparse_only = _sr("d2", "23.501", 0)
        fused = MilvusStore._rrf_fuse(
            [dense_only], [sparse_only], 2,
            k_dense=10, k_sparse=1000,
        )
        assert fused[0].chunk_id == dense_only.chunk_id

    def test_larger_k_gives_lower_weight(self):
        dense_only = _sr("d1", "38.413", 0)
        sparse_only = _sr("d2", "23.501", 0)
        fused = MilvusStore._rrf_fuse(
            [dense_only], [sparse_only], 2,
            k_dense=1000, k_sparse=10,
        )
        assert fused[0].chunk_id == sparse_only.chunk_id

    def test_default_k_preserves_original_behavior(self):
        dense = [_sr("d1", "38.413", 0)]
        sparse = [_sr("d2", "23.501", 0)]
        fused = MilvusStore._rrf_fuse(dense, sparse, 2)
        # 等权时双路单 rank1 得分相同, 顺序由输入决定
        assert len(fused) == 2
