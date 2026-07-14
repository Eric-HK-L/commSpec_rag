"""检索子系统."""

from .search import HybridRetriever, RetrievalResult
from .vector_store import Chunk, SearchResult, VectorStore

# ── 懒加载: 避免强制依赖 pymilvus / torch ──
_MILVUS_CLS = None
_NNROUTER_CLS = None
_NNROUTER_MODEL_CLS = None


def MilvusStore(*args, **kwargs):
    global _MILVUS_CLS
    if _MILVUS_CLS is None:
        from .milvus_store import MilvusStore as _M
        _MILVUS_CLS = _M
    return _MILVUS_CLS(*args, **kwargs)


def NNRouter(*args, **kwargs):
    global _NNROUTER_CLS
    if _NNROUTER_CLS is None:
        from .router import NNRouter as _N
        _NNROUTER_CLS = _N
    return _NNROUTER_CLS(*args, **kwargs)


def NNRouterModel(*args, **kwargs):
    global _NNROUTER_MODEL_CLS
    if _NNROUTER_MODEL_CLS is None:
        from .router import NNRouterModel as _NM
        _NNROUTER_MODEL_CLS = _NM
    return _NNROUTER_MODEL_CLS(*args, **kwargs)


__all__ = [
    "VectorStore", "SearchResult", "Chunk",
    "MilvusStore",
    "HybridRetriever", "RetrievalResult",
    "NNRouter", "NNRouterModel",
]
