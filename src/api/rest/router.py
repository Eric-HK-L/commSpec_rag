"""REST API — /search + /ask + /documents + /stats + SSE 流式."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.rest.schemas import (
    APIResponse,
    ChunkItem,
    DocumentDetail,
    DocumentItem,
    PaginationMeta,
    SearchFilters,
    SystemStats,
)
from src.generator.pipeline import RAGPipeline
from src.retriever.cross_ref import extract_references
from src.retriever.search import RetrievalResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["3GPP RAG"])

_pipeline: RAGPipeline | None = None


def set_pipeline(pipeline: RAGPipeline) -> None:
    global _pipeline
    _pipeline = pipeline


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="服务未就绪")
    return _pipeline


# ── 请求/响应模型 ──

class ChatMessage(BaseModel):
    role: str = Field(..., pattern=r"^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=16000)


class AskRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=50)
    release: str | None = Field(default=None, description="Release 过滤, 如 'R18'")
    series: str | None = Field(default=None, description="Series 过滤, 如 '38'")
    doc_type: str | None = Field(default=None, description="文档类型过滤, '3gpp' 或 'oran'")
    reranker_enabled: bool = Field(default=True, description="是否启用 Cross-Encoder 精排 (质量优先, 关闭可提速)")
    history: list[ChatMessage] = Field(default_factory=list, description="多轮对话历史 (用于上下文理解)")


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: SearchFilters | None = None


class SourceItem(BaseModel):
    chunk_id: str | int
    text: str
    score: float
    doc_id: str = ""
    series: int = 0
    spec_number: str = ""
    release: str = ""
    parent_section_id: str = ""
    parent_title: str = ""
    section_number: str = ""     # chunk 自身的章节编号，如 "7.1.1"
    section_title: str = ""      # 章节标题，如 "UE behaviour"
    section_path: str = ""       # 层级路径
    # Phase 5: chunk 元数据
    content_type: str = ""
    spec_role: str = ""
    topic_domain: str = ""


class AskResponse(BaseModel):
    query: str
    answer: str
    verified: bool
    warnings: list[str]
    coverage: float
    expanded_query: str
    sources: list[SourceItem]


class SearchResponse(BaseModel):
    query: str
    total: int
    results: list[SourceItem]


class HealthResponse(BaseModel):
    status: str
    vector_db: str
    chunk_count: int


# ── 端点 ──

@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    pipeline = _pipeline
    if pipeline is None:
        return HealthResponse(status="initializing", vector_db="unknown", chunk_count=0)
    return HealthResponse(
        status="ready",
        vector_db=pipeline._store.__class__.__name__,
        chunk_count=pipeline._store.count,
    )


@router.post("/search", response_model=APIResponse[SearchResponse])
async def search_endpoint(req: SearchRequest) -> APIResponse[SearchResponse]:
    pipeline = get_pipeline()
    try:
        # 多取 2x 结果以补偿低质量过滤
        results = pipeline.search(
            req.query, top_k=req.top_k * 2,
            release=req.filters.release if req.filters else None,
            series=req.filters.series if req.filters else None,
            doc_type=req.filters.doc_type if req.filters else None,
        )
        # 过滤低信息密度章节 (缩写表等)
        results = _filter_low_quality(results, req.top_k)
        # 客户端 spec_number 精细过滤
        if req.filters and req.filters.spec_number:
            results = [r for r in results if r.spec_number == req.filters.spec_number]
            results = results[:req.top_k]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")
    return APIResponse.ok(SearchResponse(
        query=req.query,
        total=len(results),
        results=[_result_to_source(r) for r in results],
    ))


@router.post("/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest) -> AskResponse:
    pipeline = get_pipeline()
    try:
        response = pipeline.ask(
            req.query,
            reranker_enabled=req.reranker_enabled,
            history=[h.model_dump() for h in req.history] if req.history else None,
            release=req.release,
            series=req.series,
            doc_type=req.doc_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成失败: {e}")
    return AskResponse(
        query=response.query,
        answer=response.answer,
        verified=response.verified,
        warnings=response.warnings,
        coverage=response.coverage,
        expanded_query=response.expanded_query,
        sources=[_result_to_source(r) for r in response.sources],
    )


def _result_to_source(r: RetrievalResult) -> SourceItem:
    return SourceItem(
        chunk_id=r.chunk_id,
        text=r.text[:500],
        score=round(r.score, 4),
        doc_id=r.doc_id,
        series=r.series,
        spec_number=r.spec_number,
        release=r.release,
        parent_section_id=r.parent_section_id,
        parent_title=r.parent_title,
        section_number=getattr(r, 'section_number', '') or '',
        section_title=getattr(r, 'section_title', '') or '',
        section_path=getattr(r, 'section_path', '') or '',
        content_type=getattr(r, 'content_type', '') or '',
        spec_role=getattr(r, 'spec_role', '') or '',
        topic_domain=getattr(r, 'topic_domain', '') or '',
    )


LOW_QUALITY_SECTIONS = {"Abbreviations", "Definitions", "Symbols", "References"}


def _filter_low_quality(results: list[RetrievalResult], target_k: int) -> list[RetrievalResult]:
    """过滤低信息密度章节（缩写表、符号表等），保留 target_k 条高质量结果."""
    quality: list[RetrievalResult] = []
    for r in results:
        if r.parent_title not in LOW_QUALITY_SECTIONS:
            quality.append(r)
            if len(quality) >= target_k:
                break
    # 如果高质量结果不够，补充低质量结果
    if len(quality) < target_k:
        for r in results:
            if r.parent_title in LOW_QUALITY_SECTIONS and len(quality) < target_k:
                quality.append(r)
    return quality


# ── 搜索增强 ──

@router.post("/search/count", response_model=APIResponse[int])
async def search_count_endpoint(req: SearchRequest) -> APIResponse[int]:
    """返回检索结果总数 (不含具体内容)."""
    pipeline = get_pipeline()
    try:
        results = pipeline.search(
            req.query, top_k=req.top_k,
            release=req.filters.release if req.filters else None,
            series=req.filters.series if req.filters else None,
            doc_type=req.filters.doc_type if req.filters else None,
        )
        if req.filters and req.filters.spec_number:
            results = [r for r in results if r.spec_number == req.filters.spec_number]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"检索失败: {e}")
    return APIResponse.ok(len(results))


class BatchSearchQuery(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=50)
    filters: SearchFilters | None = None


class BatchSearchRequest(BaseModel):
    queries: list[BatchSearchQuery] = Field(..., min_length=1, max_length=10, description="批量查询, 最多 10 条")


class BatchSearchItem(BaseModel):
    query: str
    total: int
    results: list[SourceItem]


@router.post("/search/batch", response_model=APIResponse[list[BatchSearchItem]])
async def search_batch_endpoint(req: BatchSearchRequest) -> APIResponse[list[BatchSearchItem]]:
    """批量检索: 最多 10 条查询并行执行, 返回聚合结果."""
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    pipeline = get_pipeline()

    def _search_one(q: BatchSearchQuery) -> BatchSearchItem:
        results = pipeline.search(
            q.query, top_k=q.top_k * 2,
            release=q.filters.release if q.filters else None,
            series=q.filters.series if q.filters else None,
            doc_type=q.filters.doc_type if q.filters else None,
        )
        results = _filter_low_quality(results, q.top_k)
        if q.filters and q.filters.spec_number:
            results = [r for r in results if r.spec_number == q.filters.spec_number]
            results = results[:q.top_k]
        return BatchSearchItem(
            query=q.query,
            total=len(results),
            results=[_result_to_source(r) for r in results],
        )

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=min(len(req.queries), 5)) as pool:
        tasks = [loop.run_in_executor(pool, _search_one, q) for q in req.queries]
        batch_results = await asyncio.gather(*tasks)

    return APIResponse.ok(list(batch_results))


# ── 文档管理 CRUD ──

@router.get("/documents", response_model=APIResponse[list[DocumentItem]])
async def list_documents(
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=1000),
    series: str | None = Query(None),
    release: str | None = Query(None),
) -> APIResponse[list[DocumentItem]]:
    """文档列表, 支持分页和多字段筛选."""
    pipeline = get_pipeline()
    doc_map = _get_document_map(pipeline)
    docs = list(doc_map.values())
    if series:
        docs = [d for d in docs if str(d.series) == series]
    if release:
        docs = [d for d in docs if d.release.upper() == release.upper()]
    total = len(docs)
    docs = docs[offset : offset + limit]
    return APIResponse.ok(docs, pagination=PaginationMeta(offset=offset, limit=limit, total=total).model_dump())


@router.get("/documents/{doc_id}", response_model=APIResponse[DocumentDetail])
async def get_document(doc_id: str) -> APIResponse[DocumentDetail]:
    """单篇文档详情."""
    pipeline = get_pipeline()
    doc_map = _get_document_map(pipeline)
    if doc_id not in doc_map:
        raise HTTPException(status_code=404, detail=f"文档不存在: {doc_id}")
    doc = doc_map[doc_id]
    return APIResponse.ok(DocumentDetail(
        doc_id=doc.doc_id, spec_number=doc.spec_number,
        release=doc.release, title=doc.title, series=doc.series,
        chunk_count=doc.chunk_count, source="docx",
    ))


@router.get("/documents/{doc_id}/chunks", response_model=APIResponse[list[ChunkItem]])
async def list_document_chunks(
    doc_id: str,
    offset: int = Query(0, ge=0),
    limit: int = Query(20, ge=1, le=200),
) -> APIResponse[list[ChunkItem]]:
    """文档下所有 chunks (分页)."""
    pipeline = get_pipeline()
    chunks = _get_document_chunks(pipeline, doc_id)
    if not chunks:
        raise HTTPException(status_code=404, detail=f"文档不存在或无 chunks: {doc_id}")
    total = len(chunks)
    return APIResponse.ok(
        chunks[offset : offset + limit],
        pagination=PaginationMeta(offset=offset, limit=limit, total=total).model_dump(),
    )


@router.delete("/documents/{doc_id}", response_model=APIResponse[dict])
async def delete_document(doc_id: str) -> APIResponse[dict]:
    """删除文档及其全部 chunks，同步更新摄入清单。"""
    pipeline = get_pipeline()
    try:
        deleted = pipeline._store.delete_by_filter(f"doc_id == '{doc_id}'")
        # 同步删除 manifest 中对应条目
        try:
            from src.ingestion.manifest import IngestionManifest
            m = IngestionManifest()
            m.load()
            # 通过 doc_id 反查 spec_number+release (doc_id 格式: "38300-i10", "23700-18-i00")
            import re as _re
            for key in list(m._records.keys()):
                sn, rel = key.split("|", 1)
                rec = m._records[key]
                if rec.file_path and doc_id in rec.file_path:
                    m.remove(sn, rel)
                    logger.info("manifest 已同步删除: %s|%s", sn, rel)
                    break
            m.save()
        except Exception as e:
            logger.warning("manifest 同步删除失败: %s", e)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除失败: {e}")
    return APIResponse.ok({"doc_id": doc_id, "deleted_chunks": deleted})


# ── SSE 流式生成 ──

@router.post("/ask/stream")
async def ask_stream_endpoint(req: AskRequest):
    """SSE 流式生成 — 走完整 Pipeline，逐词推送 LLM 输出."""
    pipeline = get_pipeline()

    async def event_stream():
        import re
        t0 = time.time()
        try:
            # 走完整 Pipeline (检索+扩展+交叉引用+多跳+验证+生成)
            response = pipeline.ask(
                req.query,
                reranker_enabled=req.reranker_enabled,
                history=[h.model_dump() for h in req.history] if req.history else None,
                release=req.release,
                series=req.series,
                doc_type=req.doc_type,
            )

            # 1. 推送 sources (来自完整 Pipeline 的检索结果)
            sources_data = [
                {
                    "chunk_id": str(r.chunk_id),
                    "text": r.text[:300],
                    "score": round(r.score, 4),
                    "spec_number": r.spec_number,
                    "parent_section_id": r.parent_section_id,
                    "parent_title": r.parent_title,
                    "section_number": getattr(r, 'section_number', '') or '',
                    "section_title": getattr(r, 'section_title', '') or '',
                    "section_path": getattr(r, 'section_path', '') or '',
                    "content_type": getattr(r, 'content_type', '') or '',
                    "spec_role": getattr(r, 'spec_role', '') or '',
                    "topic_domain": getattr(r, 'topic_domain', '') or '',
                }
                for r in response.sources
            ]
            yield f"data: {json.dumps({'type': 'sources', 'data': sources_data}, ensure_ascii=False)}\n\n"

            # 2. 逐词推送 answer (按空白+标点边界切分, 避免中文字符被截断)
            words = re.split(r'(\s+)', response.answer)
            for word in words:
                if word:
                    yield f"data: {json.dumps({'type': 'chunk', 'content': word}, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.02)

            # 3. 完成
            dt = (time.time() - t0) * 1000
            yield f"data: {json.dumps({'type': 'done', 'elapsed_ms': round(dt), 'warnings': response.warnings, 'verified': response.verified}, ensure_ascii=False)}\n\n"

        except Exception as e:
            logger.error("SSE 流式失败: %s", e)
            yield f"data: {json.dumps({'type': 'error', 'detail': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── 系统统计 ──

@router.get("/stats", response_model=APIResponse[SystemStats])
async def system_stats() -> APIResponse[SystemStats]:
    """系统统计信息."""
    pipeline = get_pipeline()
    doc_map = _get_document_map(pipeline)

    releases: dict[str, int] = defaultdict(int)
    series_dist: dict[str, int] = defaultdict(int)
    doc_types: dict[str, int] = defaultdict(int)
    for doc in doc_map.values():
        dt = getattr(doc, "doc_type", "3gpp") or "3gpp"
        # release 维度仅对 3GPP 有意义，ORAN 版本号 (如 v21.00) 不应混入
        if doc.release and dt == "3gpp":
            releases[doc.release] += 1
        series_dist[str(doc.series)] += doc.chunk_count
        doc_types[dt] += 1

    available_series = sorted(set(str(doc.series) for doc in doc_map.values() if doc.series > 0))

    return APIResponse.ok(SystemStats(
        total_docs=len(doc_map),
        total_chunks=pipeline._store.count,
        releases=dict(releases),
        series_distribution=dict(series_dist),
        vector_db=pipeline._store.__class__.__name__,
        available_series=available_series,
        doc_types=dict(doc_types),
    ))


# ── 内部辅助 ──

def _get_document_map(pipeline: RAGPipeline) -> dict[str, DocumentItem]:
    """从 Milvus 向量库构建文档索引."""
    store = pipeline._store
    doc_map: dict[str, DocumentItem] = {}
    summary = store.get_documents_summary()
    for doc_id, info in summary.items():
        doc_map[doc_id] = DocumentItem(
            doc_id=info.get("doc_id", doc_id),
            spec_number=info.get("spec_number", ""),
            release=info.get("release", ""),
            title=info.get("title", ""),
            series=info.get("series", 0),
            chunk_count=info.get("chunk_count", 0),
            doc_type=info.get("doc_type", "3gpp"),
        )
    return doc_map


def _get_document_chunks(pipeline: RAGPipeline, doc_id: str) -> list[ChunkItem]:
    """获取文档的所有 chunks (按 chunk_index 排序)."""
    store = pipeline._store
    raw_chunks = store.get_document_chunks(doc_id)
    return [
        ChunkItem(
            chunk_id=int(r.get("id", i)),
            text=str(r.get("text", ""))[:500],
            spec_number=str(r.get("spec_number", "")),
            release=str(r.get("release", "")),
            series=int(r.get("series", 0)),
            parent_section_id=str(r.get("parent_section_id", "")),
            parent_title=str(r.get("parent_title", "")),
            chunk_index=int(r.get("chunk_index", 0)),
            section_number=str(r.get("section_number", "")),
            section_title=str(r.get("section_title", "")),
            section_path=str(r.get("section_path", "")),
            content_type=str(r.get("content_type", "")),
            spec_role=str(r.get("spec_role", "")),
            topic_domain=str(r.get("topic_domain", "")),
        )
        for i, r in enumerate(raw_chunks)
    ]


# ── 引用图谱 ──

@router.get("/refs/graph")
async def reference_graph(
    spec: str = Query(..., description="规范编号, 如 38.300 或 38300"),
):
    """返回指定规范的引用关系图.

    扫描该规范所有 chunk 的文本, 提取其中的 3GPP 规范引用
    (如 "TS 38.413 §8.3.1"), 按被引用规范聚合返回.
    """
    pipeline = get_pipeline()
    store = pipeline._store

    # 归一化 spec 格式: "38300" → "38.300"
    spec = spec.strip()
    if "." not in spec and len(spec) >= 4:
        spec = f"{spec[:2]}.{spec[2:]}"

    store._ensure_connected()
    if store._collection is None:
        return {"spec": spec, "reference_count": 0, "references": [], "error": "集合未初始化"}

    try:
        from src.retriever.milvus_store import _escape_milvus_expr
        results = store._collection.query(
            expr=f'spec_number == "{_escape_milvus_expr(spec)}"',
            output_fields=["text", "spec_number", "doc_id", "release"],
            limit=10000,
        )
    except Exception as e:
        logger.error("查询引用图失败 (spec=%s): %s", spec, e)
        return {"spec": spec, "reference_count": 0, "references": [], "error": str(e)}

    # 从所有 chunk 文本中提取引用并聚合
    ref_map: dict[str, set[str]] = {}  # spec_number → {clause_texts}
    for row in results:
        refs = extract_references(row.get("text", ""))
        for ref in refs:
            if ref.spec_number == spec:
                continue  # 跳过自引用
            if ref.spec_number not in ref_map:
                ref_map[ref.spec_number] = set()
            label = ref.clause or ref.table or ref.raw_text[:60]
            if label:
                ref_map[ref.spec_number].add(label)

    references = sorted(
        [{"spec": sn, "clauses": sorted(clauses)} for sn, clauses in ref_map.items()],
        key=lambda r: r["spec"],
    )

    return {
        "spec": spec,
        "chunk_count": len(results),
        "reference_count": len(references),
        "references": references,
    }
