"use client";

import { useState, useEffect, useCallback } from "react";
import Link from "next/link";
import { getDocuments, getStats, type DocumentItem, type SystemStats } from "@/lib/api";

export default function DocumentsPage() {
  const [docs, setDocs] = useState<DocumentItem[]>([]);
  const [stats, setStats] = useState<SystemStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [seriesFilter, setSeriesFilter] = useState("");
  const [releaseFilter, setReleaseFilter] = useState("");

  // 从 API 数据中提取可用的 Release 和 Series (非硬编码)
  const releases = stats ? Object.keys(stats.releases) : [];
  const series = stats ? Object.keys(stats.series_distribution) : [];

  const fetchDocs = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [docsData, statsData] = await Promise.all([
        getDocuments(0, 200, seriesFilter || undefined, releaseFilter || undefined),
        getStats(),
      ]);
      setDocs(docsData);
      setStats(statsData);
    } catch (err) {
      setError(err instanceof Error ? err.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [seriesFilter, releaseFilter]);

  useEffect(() => { fetchDocs(); }, [fetchDocs]);

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      {/* 导航 */}
      <div className="flex items-center gap-4">
        <Link
          href="/"
          className="inline-flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
        >
          <span className="text-lg leading-none">←</span> 返回首页
        </Link>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">3GPP 规范文档</h1>
      </div>

      {/* 统计栏 */}
      {stats && (
        <div className="flex flex-wrap gap-4">
          <StatBadge label="总文档" value={stats.total_docs} />
          <StatBadge label="总 Chunks" value={stats.total_chunks} />
          {Object.entries(stats.releases).map(([rel, cnt]) => (
            <StatBadge key={rel} label={rel} value={cnt} color="blue" />
          ))}
        </div>
      )}

      {/* 筛选 */}
      <div className="flex gap-3">
        <select
          value={releaseFilter}
          onChange={(e) => setReleaseFilter(e.target.value)}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300"
        >
          <option value="">所有 Release</option>
          {releases.map((r) => <option key={r} value={r}>{r}</option>)}
        </select>
        <select
          value={seriesFilter}
          onChange={(e) => setSeriesFilter(e.target.value)}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300"
        >
          <option value="">所有 Series</option>
          {series.map((s) => <option key={s} value={s}>{s} 系列</option>)}
        </select>
        <button
          onClick={fetchDocs}
          disabled={loading}
          className="px-4 py-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
        >
          {loading ? "加载中..." : "刷新"}
        </button>
      </div>

      {/* 错误 */}
      {error && (
        <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm border border-red-200 dark:border-red-800">{error}</div>
      )}

      {/* 文档列表 */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 overflow-hidden shadow-sm transition-colors">
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 text-left text-gray-500 dark:text-gray-400 font-medium">
            <tr>
              <th className="px-4 py-3">规范编号</th>
              <th className="px-4 py-3">Release</th>
              <th className="px-4 py-3">Series</th>
              <th className="px-4 py-3 text-right">Chunks</th>
              <th className="px-4 py-3">文档 ID</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
            {docs.map((doc) => (
              <tr key={doc.doc_id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                <td className="px-4 py-2.5 font-mono text-blue-700 dark:text-blue-400 font-medium">
                  {doc.spec_number || "—"}
                </td>
                <td className="px-4 py-2.5">
                  <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300">
                    {doc.release}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-gray-500 dark:text-gray-400">{doc.series || "—"}</td>
                <td className="px-4 py-2.5 text-right font-mono text-gray-700 dark:text-gray-300">{doc.chunk_count}</td>
                <td className="px-4 py-2.5 text-xs text-gray-400 dark:text-gray-500 font-mono truncate max-w-[200px]">
                  {doc.doc_id}
                </td>
              </tr>
            ))}
            {docs.length === 0 && !loading && (
              <tr>
                <td colSpan={5} className="px-4 py-10 text-center text-gray-400 dark:text-gray-500">
                  暂无数据 — 请先执行文档摄入
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function StatBadge({ label, value, color = "gray" }: { label: string; value: number; color?: string }) {
  const colors: Record<string, string> = {
    gray: "bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300",
    blue: "bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300",
    green: "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300",
  };
  return (
    <div className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium ${colors[color] || colors.gray}`}>
      <span className="opacity-70">{label}</span>
      <span className="font-bold">{value}</span>
    </div>
  );
}
