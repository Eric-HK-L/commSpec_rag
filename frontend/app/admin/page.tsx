"use client";

import { useState, useEffect } from "react";
import { getAdminStats, type AdminStats } from "@/lib/api";

export default function AdminDashboard() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getAdminStats()
      .then(setStats)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-gray-500 dark:text-gray-400">加载中...</div>;
  if (error) return <div className="p-8 text-red-600 dark:text-red-400">加载失败: {error}</div>;
  if (!stats) return null;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-8">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">📊 知识库仪表盘</h1>

      {/* 核心指标 */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card title="总规范数" value={stats.total_docs} />
        <Card title="总 Chunks" value={stats.total_chunks} color="blue" />
        <Card title="Manifest 记录" value={stats.manifest_records} color="green" />
        <Card title="BM25 索引" value={stats.bm25_loaded ? `${stats.bm25_count} 条` : "未就绪"} color={stats.bm25_loaded ? "green" : "red"} />
      </div>

      {/* Release 分布 */}
      <Section title="Release 分布">
        <div className="flex flex-wrap gap-4">
          {Object.entries(stats.releases).length === 0 && <span className="text-gray-400 dark:text-gray-500">暂无数据</span>}
          {Object.entries(stats.releases).map(([rel, cnt]) => (
            <div key={rel} className="px-4 py-3 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800">
              <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">{cnt}</div>
              <div className="text-xs text-blue-500 dark:text-blue-400 mt-0.5">{rel}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* Series 分布 */}
      <Section title="Series Chunk 分布">
        <div className="space-y-2 max-w-md">
          {Object.entries(stats.series_chunk_distribution).length === 0 && <span className="text-gray-400 dark:text-gray-500">暂无数据</span>}
          {Object.entries(stats.series_chunk_distribution)
            .sort(([, a]: [string, number], [, b]: [string, number]) => b - a)
            .map(([series, chunks]) => {
            const max = Math.max(...Object.values(stats.series_chunk_distribution), 1);
            return (
              <div key={series} className="flex items-center gap-3">
                <span className="w-10 text-sm font-mono text-gray-600 dark:text-gray-400 text-right">{series}</span>
                <div className="flex-1 h-5 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 dark:bg-blue-400 rounded-full transition-all"
                    style={{ width: `${(chunks / max) * 100}%` }}
                  />
                </div>
                <span className="text-sm text-gray-500 dark:text-gray-400 w-16 text-right">{chunks.toLocaleString()}</span>
              </div>
            );
          })}
        </div>
      </Section>

      {/* 元数据 */}
      <Section title="系统信息">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <MetaItem label="向量库" value={stats.vector_db} />
          <MetaItem label="嵌入维度" value={String(stats.embedding_dim)} />
          <MetaItem label="BM25 状态" value={stats.bm25_loaded ? "✅ 已加载" : "⚠️ 未加载"} />
          <MetaItem label="最近摄入" value={stats.last_ingestion ? new Date(stats.last_ingestion).toLocaleString("zh-CN") : "—"} />
        </div>
      </Section>
    </div>
  );
}

function Card({ title, value, color = "gray" }: { title: string; value: string | number; color?: string }) {
  const bg: Record<string, string> = {
    gray: "bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700",
    blue: "bg-blue-50 dark:bg-blue-900/20 border-blue-100 dark:border-blue-800",
    green: "bg-green-50 dark:bg-green-900/20 border-green-100 dark:border-green-800",
    red: "bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800",
  };
  const tc: Record<string, string> = {
    gray: "text-gray-700 dark:text-gray-300",
    blue: "text-blue-700 dark:text-blue-300",
    green: "text-green-700 dark:text-green-300",
    red: "text-red-700 dark:text-red-300",
  };
  return (
    <div className={`p-4 rounded-xl border ${bg[color]}`}>
      <div className={`text-2xl font-bold ${tc[color]}`}>{value}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{title}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="p-5 rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm transition-colors">
      <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-4">{title}</h2>
      {children}
    </div>
  );
}

function MetaItem({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-100 dark:border-gray-700">
      <div className="text-xs text-gray-400 dark:text-gray-500">{label}</div>
      <div className="font-medium text-gray-700 dark:text-gray-300 text-sm mt-0.5 truncate">{value}</div>
    </div>
  );
}
