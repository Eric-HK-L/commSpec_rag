"""API 统一响应格式、分页、过滤模型."""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# ── 统一响应 ──

class APIResponse(BaseModel, Generic[T]):
    """统一 API 响应包裹器."""
    success: bool = True
    data: T | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, data: T, **meta: Any) -> "APIResponse[T]":
        return cls(success=True, data=data, meta=meta)

    @classmethod
    def fail(cls, error: str, **meta: Any) -> "APIResponse[T]":
        return cls(success=False, error=error, meta=meta)


class ErrorDetail(BaseModel):
    """结构化错误详情."""
    error_code: str
    detail: str
    suggestion: str | None = None


# ── 分页 ──

class PaginationParams(BaseModel):
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=20, ge=1, le=200)


class PaginationMeta(BaseModel):
    offset: int
    limit: int
    total: int


# ── 过滤 ──

class SearchFilters(BaseModel):
    """检索过滤条件."""
    series: str | None = Field(default=None, description="规范系列号, 如 '38'")
    release: str | None = Field(default=None, description="Release, 如 'R18'")
    spec_number: str | None = Field(default=None, description="规范编号, 如 '38.300'")
    doc_type: str | None = Field(default=None, description="文档类型, '3gpp' 或 'oran'")


# ── 排序 ──

class SortParams(BaseModel):
    sort_by: str = Field(default="score", description="排序字段: score/relevance")
    sort_order: str = Field(default="desc", pattern="^(asc|desc)$")


# ── 文档模型 ──

class DocumentItem(BaseModel):
    """文档列表项."""
    doc_id: str
    spec_number: str
    release: str
    title: str = ""
    series: int = 0
    chunk_count: int = 0
    doc_type: str = "3gpp"  # "3gpp" | "oran"


class DocumentDetail(BaseModel):
    """文档详情."""
    doc_id: str
    spec_number: str
    release: str
    version: str = ""
    title: str = ""
    series: int = 0
    chunk_count: int = 0
    source: str = ""  # docx (from_scratch ingestion)


class ChunkItem(BaseModel):
    """单个 chunk."""
    chunk_id: int
    text: str
    spec_number: str
    release: str
    series: int = 0
    parent_section_id: str = ""
    parent_title: str = ""
    chunk_index: int = 0
    section_number: str = ""     # chunk 自身的章节编号，如 "7.1.1"
    section_title: str = ""      # 章节标题，如 "UE behaviour"
    section_path: str = ""       # 层级路径
    # Phase 5: chunk 元数据
    content_type: str = ""
    spec_role: str = ""
    topic_domain: str = ""


# ── 系统统计 ──

class SystemStats(BaseModel):
    total_docs: int
    total_chunks: int
    releases: dict[str, int]  # release → doc count
    series_distribution: dict[str, int]  # series → chunk count
    vector_db: str
    embedding_dim: int = 1024
    available_series: list[str] = []  # 可用的 Series 列表
    doc_types: dict[str, int] = {}  # doc_type → chunk count
