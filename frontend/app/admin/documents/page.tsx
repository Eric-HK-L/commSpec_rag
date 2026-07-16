"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { getDocuments, deleteDocument, getManifest, type DocumentItem, type ManifestItem } from "@/lib/api";

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

  const [manifestMap, setManifestMap] = useState<Map<string, string>>(new Map());

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [data, manifest] = await Promise.all([
        getDocuments(0, 200, series || undefined, release || undefined),
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
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">📄 文档管理</h1>
        <button onClick={fetch} className="px-4 py-2 text-sm bg-gray-100 dark:bg-gray-800 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
          {loading ? "刷新中..." : "刷新"}
        </button>
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
                      ✅ 已摄入{manifestMap.has(doc.spec_number || "") ? ` ${new Date(manifestMap.get(doc.spec_number || "")!).toLocaleDateString("zh-CN")}` : ""}
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
    </div>
  );
}
