"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { getDocuments, deleteDocument, getManifest, uploadDocument, getOtherDocuments, type DocumentItem, type ManifestItem, type OtherDocumentItem } from "@/lib/api";

const SERIES = ["21", "22", "23", "24", "36", "38"];
const RELEASES = ["R18", "R19"];

export default function AdminDocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [series, setSeries] = useState("");
  const [release, setRelease] = useState("");
  const [search, setSearch] = useState("");
  const [deleting, setDeleting] = useState<string | null>(null);
  const [showUpload, setShowUpload] = useState(false);

  const [manifestMap, setManifestMap] = useState<Map<string, string>>(new Map());
  const [otherDocs, setOtherDocs] = useState<OtherDocumentItem[]>([]);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, manifest] = await Promise.all([
        getDocuments(0, 500, series || undefined, release || undefined),
        getManifest().catch(() => []),
      ]);
      // 构建 spec_number → ingested_at 映射
      const mmap = new Map<string, string>();
      for (const m of manifest) {
        if (!mmap.has(m.spec_number) || m.ingested_at > (mmap.get(m.spec_number) || "")) {
          mmap.set(m.spec_number, m.ingested_at);
        }
      }
      setManifestMap(mmap);
      if (search) {
        setDocs(data.filter(d => d.spec_number?.toLowerCase().includes(search.toLowerCase()) || d.doc_id?.toLowerCase().includes(search.toLowerCase())));
      } else {
        setDocs(data);
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [series, release, search]);

  useEffect(() => { fetch(); }, [fetch]);

  // 加载 other/ 目录中的非 3GPP/O-RAN 文档 (独立请求, 失败不阻塞主列表)
  useEffect(() => {
    getOtherDocuments().then(setOtherDocs).catch(() => setOtherDocs([]));
  }, []);

  const handleDelete = async (docId: string) => {
    if (!confirm(`确认删除文档 ${docId} 的所有 chunks？此操作不可撤销。`)) return;
    setDeleting(docId);
    try {
      await deleteDocument(docId);
      setDocs(prev => prev.filter(d => d.doc_id !== docId));
    } catch (e) {
      setError(e instanceof Error ? e.message : "删除失败");
    } finally {
      setDeleting(null);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">文档管理</h1>
        <div className="flex gap-2">
          <button
            onClick={() => setShowUpload(true)}
            className="px-4 py-2 text-sm bg-blue-600 dark:bg-blue-500 text-white rounded-lg hover:bg-blue-700 dark:hover:bg-blue-600 transition-colors"
          >
            上传文档
          </button>
          <button onClick={fetch} className="px-4 py-2 text-sm bg-gray-100 dark:bg-gray-800 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
            {loading ? "刷新中..." : "刷新"}
          </button>
        </div>
      </div>

      {/* 筛选 + 搜索 */}
      <div className="flex flex-wrap gap-3">
        <select value={release} onChange={(e) => setRelease(e.target.value)} className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300">
          <option value="">所有 Release</option>
          {RELEASES.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <select value={series} onChange={(e) => setSeries(e.target.value)} className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300">
          <option value="">所有 Series</option>
          {SERIES.map((s) => <option key={s} value={s}>{s} 系列</option>)}
        </select>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索规范编号..."
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300 w-48"
        />
        <span className="text-sm text-gray-500 dark:text-gray-400 self-center">{docs.length} 篇</span>
      </div>

      {error && <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm">{error}</div>}

      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 overflow-hidden shadow-sm transition-colors">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 text-left text-gray-500 dark:text-gray-400">
            <tr>
              <th className="px-4 py-3 w-32">规范编号</th>
              <th className="px-4 py-3">Release</th>
              <th className="px-4 py-3">Series</th>
              <th className="px-4 py-3 text-right">Chunks</th>
              <th className="px-4 py-3 hidden md:table-cell">文档 ID</th>
              <th className="px-4 py-3 w-28 hidden md:table-cell">摄入状态</th>
              <th className="px-4 py-3 text-right">操作</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
            {docs.map((doc) => (
              <tr key={doc.doc_id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                <td className="px-4 py-2.5 font-mono text-blue-700 dark:text-blue-400 font-medium">{doc.spec_number || "—"}</td>
                <td className="px-4 py-2.5"><span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300">{doc.release}</span></td>
                <td className="px-4 py-2.5 text-gray-500 dark:text-gray-400">{doc.series || "—"}</td>
                <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">{doc.chunk_count}</td>
                <td className="px-4 py-2.5 hidden md:table-cell text-xs text-gray-400 dark:text-gray-500 font-mono max-w-[180px] truncate">{doc.doc_id}</td>
                <td className="px-4 py-2.5 hidden md:table-cell">
                  {doc.chunk_count > 0 ? (
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300">
                      已摄入{manifestMap.has(doc.spec_number || "") ? ` ${new Date(manifestMap.get(doc.spec_number || "")!).toLocaleDateString("zh-CN")}` : ""}
                    </span>
                  ) : (
                    <span className="px-2 py-0.5 rounded text-xs font-medium bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400">
                      —
                    </span>
                  )}
                </td>
                <td className="px-4 py-2.5 text-right">
                  <div className="flex gap-2 justify-end">
                    <Link href={`/admin/documents/${encodeURIComponent(doc.doc_id)}`} className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 text-xs font-medium">
                      查看
                    </Link>
                    <button
                      onClick={() => handleDelete(doc.doc_id)}
                      disabled={deleting === doc.doc_id}
                      className="text-red-500 dark:text-red-400 hover:text-red-700 dark:hover:text-red-300 text-xs font-medium disabled:opacity-40"
                    >
                      {deleting === doc.doc_id ? "删除中..." : "删除"}
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {docs.length === 0 && !loading && (
              <tr><td colSpan={7} className="px-4 py-10 text-center text-gray-400 dark:text-gray-500">暂无数据</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {showUpload && (
        <UploadModal
          onClose={() => setShowUpload(false)}
          onUploaded={() => { setShowUpload(false); fetch(); }}
        />
      )}

      {/* 其他文档: 非 3GPP/O-RAN, 不参与摄入 */}
      {otherDocs.length > 0 && (
        <div className="rounded-xl border border-yellow-200 dark:border-yellow-700/60 bg-yellow-50/60 dark:bg-yellow-900/10 p-4 space-y-2 transition-colors">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-sm font-bold text-yellow-900 dark:text-yellow-200">其他文档 (other/)</h2>
            <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-yellow-200 dark:bg-yellow-900/40 text-yellow-800 dark:text-yellow-300">
              未识别 · 不摄入
            </span>
            <span className="text-xs text-yellow-700 dark:text-yellow-400">{otherDocs.length} 个文件</span>
          </div>
          <p className="text-xs text-yellow-700 dark:text-yellow-500 leading-relaxed">
            以下文档既不属于 3GPP 也不属于 O-RAN，已归档到 other/ 目录，仅作留存，不会嵌入向量库。
          </p>
          <ul className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
            {otherDocs.map((d) => (
              <li key={d.filename} className="flex items-center justify-between gap-2 text-xs font-mono text-gray-700 dark:text-gray-300 bg-white/60 dark:bg-gray-900/40 rounded px-2.5 py-1.5">
                <span className="truncate" title={d.filename}>{d.filename}</span>
                <span className="shrink-0 text-gray-400 dark:text-gray-500">{(d.size_bytes / 1024).toFixed(1)} KB</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

// ── 上传弹窗 ──

const CATEGORY_LABEL: Record<string, string> = {
  marked: "marked/ — Markdown 协议数据集 (嵌入默认源)",
  original: "original/ — 原始 DOCX (pandoc 处理)",
  other: "other/ — 其他文档 (默认不摄入)",
};

const KIND_LABEL: Record<string, string> = {
  "3gpp": "3GPP 协议",
  oran: "O-RAN 协议",
  unknown: "未识别 (归入 other/)",
};

function UploadModal({ onClose, onUploaded }: { onClose: () => void; onUploaded: () => void }) {
  const [file, setFile] = useState<File | null>(null);
  const [category, setCategory] = useState("auto");
  const [release, setRelease] = useState("");
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<{
    filename: string;
    category: string;
    detected_kind: string;
    target_path: string;
    size_bytes: number;
    duplicate: boolean;
  } | null>(null);

  const handleUpload = async () => {
    if (!file) return;
    setUploading(true);
    setError(null);
    setResult(null);
    try {
      const res = await uploadDocument(
        file,
        category === "auto" ? undefined : (category as "marked" | "original" | "other"),
        release || undefined,
      );
      setResult(res.data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "上传失败");
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4" onClick={onClose}>
      <div className="w-full max-w-lg rounded-2xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-xl p-6 space-y-4" onClick={(e) => e.stopPropagation()}>
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-bold text-gray-900 dark:text-gray-100">上传文档</h2>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 text-xl leading-none">×</button>
        </div>

        <p className="text-xs text-gray-500 dark:text-gray-400 leading-relaxed">
          上传仅保存到源目录，不会立即嵌入。规则：Markdown → marked/（3GPP/O-RAN）；DOCX 等其他格式 → original/；非 3GPP/O-RAN → other/。上传完成后请到「摄入管理」页触发增量摄入。
        </p>

        <label className="block">
          <span className="text-sm text-gray-600 dark:text-gray-400 mb-1 block">选择文件</span>
          <input
            type="file"
            onChange={(e) => { setFile(e.target.files?.[0] || null); setResult(null); setError(null); }}
            className="w-full text-sm text-gray-700 dark:text-gray-300 file:mr-3 file:px-3 file:py-1.5 file:rounded-lg file:border-0 file:bg-blue-50 dark:file:bg-blue-900/30 file:text-blue-700 dark:file:text-blue-300 file:text-sm file:font-medium hover:file:bg-blue-100 dark:hover:file:bg-blue-900/50 cursor-pointer"
          />
        </label>

        {file && (
          <div className="text-xs text-gray-500 dark:text-gray-400 font-mono truncate">
            {file.name} · {(file.size / 1024 / 1024).toFixed(2)} MB
          </div>
        )}

        <div className="grid grid-cols-2 gap-3">
          <label className="block">
            <span className="text-sm text-gray-600 dark:text-gray-400 mb-1 block">目标目录</span>
            <select
              value={category}
              onChange={(e) => setCategory(e.target.value)}
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300"
            >
              <option value="auto">自动归类 (推荐)</option>
              <option value="marked">marked/ (Markdown 数据集)</option>
              <option value="original">original/ (原始文档)</option>
              <option value="other">other/ (其他)</option>
            </select>
          </label>
          <label className="block">
            <span className="text-sm text-gray-600 dark:text-gray-400 mb-1 block">3GPP Release (可选)</span>
            <input
              type="text"
              value={release}
              onChange={(e) => setRelease(e.target.value)}
              placeholder="如 R18，默认自动检测"
              className="w-full px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300"
            />
          </label>
        </div>

        {error && <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm">{error}</div>}

        {result && (
          <div className="p-3 rounded-lg bg-green-50 dark:bg-green-900/20 border border-green-100 dark:border-green-800 space-y-1.5">
            <div className="text-sm font-medium text-green-800 dark:text-green-300">
              上传成功{result.duplicate ? "（已覆盖同名文件）" : ""}
            </div>
            <div className="text-xs text-green-700 dark:text-green-400 space-y-1 font-mono">
              <div>归类: {CATEGORY_LABEL[result.category] || result.category}</div>
              <div>识别: {KIND_LABEL[result.detected_kind] || result.detected_kind}</div>
              <div>路径: {result.target_path}</div>
            </div>
            <div className="text-xs text-green-700 dark:text-green-400">
              已保存到源目录。请前往「摄入管理」页触发增量摄入以嵌入向量库。
            </div>
          </div>
        )}

        <div className="flex gap-2 justify-end pt-2">
          <button
            onClick={onClose}
            disabled={uploading}
            className="px-4 py-2 text-sm rounded-lg border border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-40 transition-colors"
          >
            关闭
          </button>
          <button
            onClick={handleUpload}
            disabled={!file || uploading}
            className="px-4 py-2 text-sm rounded-lg bg-blue-600 dark:bg-blue-500 text-white font-medium hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-40 transition-colors"
          >
            {uploading ? "上传中..." : "上传"}
          </button>
        </div>
      </div>
    </div>
  );
}
