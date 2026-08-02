/** CommSpec RAG API 调用层 */

const API_URL = "/api/v1"; // 请求经由 Next.js rewrites 反向代理转发至后端，规避跨域与硬编码IP问题

// ── 类型定义 ──

export interface SearchResult {
  chunk_id: number;
  text: string;
  score: number;
  doc_id: string;
  series: number;
  spec_number: string;
  release: string;
  parent_section_id: string;
  parent_title: string;
  chunk_index: number;
}

export interface SourceItem {
  chunk_id: number;
  text: string;
  score: number;
  spec_number: string;
  release: string;
  section_id: string;
}

export interface AskResponse {
  query: string;
  answer: string;
  verified: boolean;
  warnings: string[];
  coverage: number;
  expanded_query?: string;
  sources: SourceItem[];
}

export interface DocumentItem {
  doc_id: string;
  spec_number: string;
  release: string;
  title: string;
  series: number;
  chunk_count: number;
}

export interface DocumentDetail extends DocumentItem {
  version: string;
  source: string;
}

export interface ChunkItem {
  chunk_id: number;
  text: string;
  spec_number: string;
  release: string;
  series: number;
  parent_section_id: string;
  parent_title: string;
  chunk_index: number;
}

export interface SystemStats {
  total_docs: number;
  total_chunks: number;
  releases: Record<string, number>;
  series_distribution: Record<string, number>;
  vector_db: string;
  embedding_dim: number;
  available_series?: string[];
  doc_types?: Record<string, number>;
}

export interface AdminStats {
  total_docs: number;
  total_chunks: number;
  releases: Record<string, number>;
  series_chunk_distribution: Record<string, number>;
  vector_db: string;
  embedding_dim: number;
  bm25_loaded: boolean;
  bm25_count: number;
  manifest_records: number;
  last_ingestion: string | null;
}

export interface ManifestItem {
  key: string;
  spec_number: string;
  release: string;
  latest_version: string;
  file_path: string;
  sha256: string;
  chunk_count: number;
  ingested_at: string;
}

export interface OtherDocumentItem {
  filename: string;
  size_bytes: number;
  modified_at: string;
  kind: string;
}

export interface IngestStatus {
  running: boolean;
  pid: number | null;
  log_tail: string[];
  last_ingestion_at: string | null;
}

export interface SystemInfo {
  python_version: string;
  platform: string;
  uptime_seconds: number;
  memory_used_mb: number;
  memory_total_mb: number;
  memory_percent: number;
  disk_used_gb: number;
  disk_total_gb: number;
  disk_percent: number;
  milvus_connected: boolean;
}

export interface LogEntry {
  lines: string[];
  total_lines: number;
  level: string;
}

export interface ConfigView {
  llm_model: string;
  llm_base_url: string;
  embedding_device: string;
  embedding_provider: string;
  chunk_size: number;
  chunk_overlap: number;
  dense_top_k: number;
  bm25_top_k: number;
  milvus_host: string;
  milvus_port: number;
  online_search_enabled: boolean;
  reranker_enabled_by_default: boolean;
}

export interface FeedbackItem {
  id: number;
  query: string;
  answer: string;
  sources: SourceItem[];
  rating: string;
  comment: string;
  created_at: string;
}

export interface FeedbackStats {
  total: number;
  up: number;
  down: number;
  up_ratio: number;
}

// ── API 调用 ──

async function apiGet<T>(path: string, params?: Record<string, string>): Promise<T> {
  const url = new URL(`${API_URL}${path}`, window.location.origin);
  if (params) {
    Object.entries(params).forEach(([k, v]) => { if (v) url.searchParams.set(k, v); });
  }
  const res = await fetch(url.toString());
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  const json = await res.json();
  // 兼容两种响应格式: 带 APIResponse 包装 (有 success 字段) vs 直接返回数据
  if ("success" in json) {
    if (!json.success && json.error) throw new Error(json.error);
    return json.data as T;
  }
  return json as T;
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  const json = await res.json();
  // 兼容两种响应格式: 带 APIResponse 包装 (有 success 字段) vs 直接返回数据
  if ("success" in json) {
    if (!json.success && json.error) throw new Error(json.error);
    return json.data as T;
  }
  return json as T;
}

// ── 端点函数 ──

/** 语义搜索 */
export async function search(query: string, topK = 10) {
  return apiPost<{ query: string; total: number; results: SearchResult[] }>("/search", { query, top_k: topK });
}

/** RAG 问答 */
export async function ask(query: string, topK = 10, release?: string, series?: string, docType?: string, rerankerEnabled = true) {
  return apiPost<AskResponse>("/ask", { query, top_k: topK, release, series, doc_type: docType, reranker_enabled: rerankerEnabled });
}

/** SSE 流式问答 — 返回 fetch Response 供 ReadableStream 消费 */
export function askStream(
  query: string,
  topK = 10,
  release?: string,
  series?: string,
  docType?: string,
  rerankerEnabled = true,
  history?: { role: string; content: string }[],
  signal?: AbortSignal,
) {
  return fetch(`${API_URL}/ask/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      query,
      top_k: topK,
      release,
      series,
      doc_type: docType,
      reranker_enabled: rerankerEnabled,
      history: history || [],
    }),
    signal,
  });
}

/** 文档列表 */
export async function getDocuments(offset = 0, limit = 50, series?: string, release?: string) {
  const params: Record<string, string> = {
    offset: String(offset),
    limit: String(limit),
  };
  if (series) params.series = series;
  if (release) params.release = release;
  return apiGet<DocumentItem[]>("/documents", params);
}

/** 文档详情 */
export async function getDocument(docId: string) {
  return apiGet<DocumentDetail>(`/documents/${encodeURIComponent(docId)}`);
}

/** 文档 chunks */
export async function getDocumentChunks(docId: string, offset = 0, limit = 50) {
  return apiGet<ChunkItem[]>(`/documents/${encodeURIComponent(docId)}/chunks`, {
    offset: String(offset),
    limit: String(limit),
  });
}

/** 系统统计 */
export async function getStats() {
  return apiGet<SystemStats>("/stats");
}

/** 批量检索 */
export async function searchBatch(queries: { query: string; top_k?: number }[]) {
  return apiPost<{ query: string; total: number; results: SearchResult[] }[]>(
    "/search/batch",
    { queries: queries.map(q => ({ query: q.query, top_k: q.top_k || 5 })) }
  );
}

/** 健康检查 */
export async function healthCheck() {
  return apiGet<{ status: string; vector_db: string; chunk_count: number }>("/health");
}

// ── 管理 API ──

/** 管理员统计（含 manifest + BM25） */
export async function getAdminStats() {
  return apiGet<AdminStats>("/admin/stats");
}

/** 触发摄入 */
export async function triggerIngestion(
  mode: "incremental" | "full" = "incremental",
  source: "marked" | "original" | "all" = "marked",
) {
  return apiPost<{ accepted: boolean; message: string; mode: string; pid: number | null }>(
    `/admin/ingest/trigger?mode=${mode}&source=${source}`, {}
  );
}

/** 上传文档 (自动归类: markdown→marked/, 其他→original/, 非3GPP/O-RAN→other/) */
export async function uploadDocument(
  file: File,
  category?: "marked" | "original" | "other",
  release?: string,
) {
  const form = new FormData();
  form.append("file", file);
  const params = new URLSearchParams();
  if (category) params.set("category", category);
  if (release) params.set("release", release);
  const qs = params.toString();
  const res = await fetch(`${API_URL}/admin/documents/upload${qs ? `?${qs}` : ""}`, {
    method: "POST",
    body: form,
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error((data as { detail?: string }).detail || `API ${res.status}`);
  return data as {
    data: {
      filename: string;
      category: string;
      detected_kind: string;
      target_path: string;
      size_bytes: number;
      duplicate: boolean;
    };
  };
}

/** other/ 目录中的非 3GPP/O-RAN 文档列表 (不参与摄入) */
export async function getOtherDocuments() {
  return apiGet<OtherDocumentItem[]>("/admin/documents/other");
}

/** 摄入状态 */
export async function getIngestStatus(lines = 50) {
  return apiGet<IngestStatus>("/admin/ingest/status", { lines: String(lines) });
}

/** 完整 manifest */
export async function getManifest() {
  return apiGet<ManifestItem[]>("/admin/manifest");
}

/** 删除 manifest 记录 */
export async function deleteManifestRecord(key: string) {
  const res = await fetch(`${API_URL}/admin/manifest/${encodeURIComponent(key)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  const json = await res.json();
  if (!json.success && json.error) throw new Error(json.error);
  return json.data as { deleted_key: string; spec_number: string; release: string };
}

/** 系统日志 */
export async function getSystemLogs(level = "ALL", lines = 100) {
  return apiGet<LogEntry>("/admin/logs", {
    level,
    lines: String(lines),
  });
}

/** 系统运行信息 */
export async function getSystemInfo() {
  return apiGet<SystemInfo>("/admin/system");
}

/** 查看配置 */
export async function getConfig() {
  return apiGet<ConfigView>("/admin/config");
}

/** 删除文档 */
export async function deleteDocument(docId: string) {
  const res = await fetch(`${API_URL}/documents/${encodeURIComponent(docId)}`, {
    method: "DELETE",
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${res.statusText}`);
  return res.json();
}

// ── 反馈 API ──

/** 提交 👍👎 反馈 */
export async function submitFeedback(params: {
  query: string;
  answer: string;
  sources: SourceItem[];
  rating: "up" | "down";
  comment?: string;
}) {
  return apiPost<{ id: number; status: string }>("/feedback", params);
}

/** 反馈统计 */
export async function getFeedbackStats() {
  return apiGet<FeedbackStats>("/feedback/stats");
}

/** 反馈列表 */
export async function getFeedbackList(rating?: string, limit = 50, offset = 0) {
  const params: Record<string, string> = { limit: String(limit), offset: String(offset) };
  if (rating) params.rating = rating;
  return apiGet<FeedbackItem[]>("/feedback", params);
}
