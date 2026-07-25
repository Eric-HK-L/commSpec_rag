"use client";

import { useState, useEffect, useCallback } from "react";
import {
  getAdminStats, getFeedbackStats, getFeedbackList, getSystemInfo,
  triggerIngestion,
  type AdminStats, type FeedbackStats, type FeedbackItem, type SystemInfo,
} from "@/lib/api";

export default function AdminDashboard() {
  const [stats, setStats] = useState<AdminStats | null>(null);
  const [feedback, setFeedback] = useState<FeedbackStats | null>(null);
  const [recentFeedback, setRecentFeedback] = useState<FeedbackItem[]>([]);
  const [sysInfo, setSysInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    setLoading(true);
    try {
      const [s, f, si, rf] = await Promise.all([
        getAdminStats(),
        getFeedbackStats().catch(() => null),
        getSystemInfo().catch(() => null),
        getFeedbackList(undefined, 5, 0).catch(() => []),
      ]);
      setStats(s);
      setFeedback(f);
      setSysInfo(si);
      setRecentFeedback(rf);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetch(); }, [fetch]);

  if (loading) return <div className="p-8 text-gray-500 dark:text-gray-400">加载中...</div>;
  if (error) return <div className="p-8 text-red-600 dark:text-red-400">加载失败: {error}</div>;
  if (!stats) return null;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">📊 知识库仪表盘</h1>
        <button onClick={fetch} className="px-3 py-1.5 text-xs bg-gray-100 dark:bg-gray-800 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
          刷新
        </button>
      </div>

      {/* ── 核心指标 ── */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Card title="总规范数" value={String(stats.total_docs)} />
        <Card title="总 Chunks" value={Number(stats.total_chunks).toLocaleString()} color="blue" />
        <Card title="BM25 索引" value={stats.bm25_loaded ? `${stats.bm25_count} 条` : "未就绪"} color={stats.bm25_loaded ? "green" : "red"} />
        <Card
          title="反馈好评率"
          value={feedback ? `${(feedback.up_ratio * 100).toFixed(0)}%` : "—"}
          color={feedback && feedback.up_ratio >= 0.6 ? "green" : "gray"}
        />
        <Card title="Manifest" value={String(stats.manifest_records)} color="green" />
      </div>

      {/* ── 系统健康 ── */}
      {sysInfo && (
        <Section title="系统健康">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <MiniBadge
              label="内存"
              value={`${(sysInfo.memory_used_mb / 1024).toFixed(2)} / ${(sysInfo.memory_total_mb / 1024).toFixed(2)} GB`}
              color={sysInfo.memory_percent > 80 ? "red" : sysInfo.memory_percent > 60 ? "yellow" : "green"}
            />
            <MiniBadge label="磁盘" value={`${sysInfo.disk_used_gb.toFixed(2)} / ${sysInfo.disk_total_gb.toFixed(2)} GB`} color="gray" />
            <MiniBadge label="Milvus" value={sysInfo.milvus_connected ? "✅ 已连接" : "❌ 断开"} color={sysInfo.milvus_connected ? "green" : "red"} />
            <MiniBadge label="Python" value={sysInfo.python_version} color="gray" />
          </div>
        </Section>
      )}

      {/* ── 快捷操作 ── */}
      <Section title="快捷操作">
        <div className="flex flex-wrap gap-3">
          <QuickBtn label="📥 增量摄入" onClick={async () => {
            try {
              const res = await triggerIngestion("incremental");
              alert(res.message);
            } catch (e) { alert(`失败: ${e}`); }
          }} />
          <QuickBtn label="🔄 全量重建" onClick={async () => {
            if (!confirm("全量重建将清空现有数据后重新导入，确认？")) return;
            try {
              const res = await triggerIngestion("full");
              alert(res.message);
            } catch (e) { alert(`失败: ${e}`); }
          }} color="red" />
          <QuickBtn label="📋 查看日志" href="/admin/logs" />
          <QuickBtn label="🖥 系统健康" href="/admin/system" />
        </div>
      </Section>

      {/* ── 最近活动 ── */}
      <Section title="最近活动">
        <div className="space-y-2 text-sm">
          {/* 摄入记录 */}
          {stats.last_ingestion && (
            <div className="flex items-center gap-3 py-1.5">
              <span className="text-xs text-gray-400 dark:text-gray-500 w-32 shrink-0">
                {new Date(stats.last_ingestion).toLocaleString("zh-CN")}
              </span>
              <span className="text-xs px-2 py-0.5 rounded bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium">摄入</span>
              <span className="text-gray-500 dark:text-gray-400">文档摄入完成</span>
            </div>
          )}
          {/* 反馈记录 */}
          {recentFeedback.map((item) => (
            <div key={item.id} className="flex items-center gap-3 py-1.5">
              <span className="text-xs text-gray-400 dark:text-gray-500 w-32 shrink-0">
                {new Date(item.created_at).toLocaleString("zh-CN")}
              </span>
              <span className={`text-xs px-2 py-0.5 rounded font-medium ${item.rating === "up" ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300" : "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-300"}`}>
                {item.rating === "up" ? "👍" : "👎"}
              </span>
              <span className="text-gray-700 dark:text-gray-300 truncate max-w-md">{item.query}</span>
            </div>
          ))}
          {!stats.last_ingestion && recentFeedback.length === 0 && (
            <span className="text-gray-400 dark:text-gray-500 text-sm">暂无活动记录</span>
          )}
        </div>
      </Section>

      {/* ── Release 分布 ── */}
      <Section title="Release 分布">
        <div className="flex flex-wrap gap-4">
          {Object.entries(stats.releases).length === 0 && <span className="text-gray-400 dark:text-gray-500 text-sm">暂无数据</span>}
          {Object.entries(stats.releases).map(([rel, cnt]) => (
            <div key={rel} className="px-4 py-3 rounded-xl bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800">
              <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">{cnt}</div>
              <div className="text-xs text-blue-500 dark:text-blue-400 mt-0.5">{rel}</div>
            </div>
          ))}
        </div>
      </Section>

      {/* ── Series Chunk 分布 ── */}
      <Section title="Series Chunk 分布">
        <div className="space-y-2 max-w-md">
          {Object.entries(stats.series_chunk_distribution).length === 0 && <span className="text-gray-400 dark:text-gray-500 text-sm">暂无数据</span>}
          {Object.entries(stats.series_chunk_distribution)
            .sort(([, a], [, b]) => Number(b) - Number(a))
            .map(([series, chunks]) => {
            const max = Math.max(...Object.values(stats.series_chunk_distribution).map(Number), 1);
            return (
              <div key={series} className="flex items-center gap-3">
                <span className="w-10 text-sm font-mono text-gray-600 dark:text-gray-400 text-right">{series}</span>
                <div className="flex-1 h-5 bg-gray-100 dark:bg-gray-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-blue-500 dark:bg-blue-400 rounded-full transition-all"
                    style={{ width: `${(Number(chunks) / max) * 100}%` }}
                  />
                </div>
                <span className="text-sm text-gray-500 dark:text-gray-400 w-16 text-right">{Number(chunks).toLocaleString()}</span>
              </div>
            );
          })}
        </div>
      </Section>

      {/* ── 系统信息 ── */}
      <Section title="系统信息">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
          <MetaItem label="向量库" value={stats.vector_db} />
          <MetaItem label="BM25 状态" value={stats.bm25_loaded ? "✅ 已加载" : "⚠️ 未加载"} />
          <MetaItem label="最近摄入" value={stats.last_ingestion ? new Date(stats.last_ingestion).toLocaleString("zh-CN") : "—"} />
          <MetaItem label="反馈总数" value={feedback ? String(feedback.total) : "—"} />
        </div>
      </Section>
    </div>
  );
}

// ── 组件 ──

function Card({ title, value, color = "gray" }: { title: string; value: string; color?: string }) {
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

function MiniBadge({ label, value, color }: { label: string; value: string; color: string }) {
  const c = { green: "text-green-700 dark:text-green-300", red: "text-red-700 dark:text-red-300", yellow: "text-yellow-700 dark:text-yellow-300", gray: "text-gray-700 dark:text-gray-300" }[color] || "text-gray-700";
  return (
    <div className="p-2 rounded-lg bg-gray-50 dark:bg-gray-800">
      <div className={`font-medium text-sm ${c}`}>{value}</div>
      <div className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">{label}</div>
    </div>
  );
}

function QuickBtn({ label, onClick, href, color = "blue" }: { label: string; onClick?: () => void; href?: string; color?: string }) {
  const cls = color === "red"
    ? "bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800 text-red-700 dark:text-red-300 hover:bg-red-100 dark:hover:bg-red-900/40"
    : "bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800 text-blue-700 dark:text-blue-300 hover:bg-blue-100 dark:hover:bg-blue-900/40";

  if (href) {
    return (
      <a href={href} className={`px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${cls}`}>
        {label}
      </a>
    );
  }
  return (
    <button onClick={onClick} className={`px-4 py-2 rounded-lg border text-sm font-medium transition-colors ${cls}`}>
      {label}
    </button>
  );
}
