"use client";

import { useState, useEffect, useCallback } from "react";
import { getSystemInfo, type SystemInfo } from "@/lib/api";

function fmtUptime(sec: number) {
  const d = Math.floor(sec / 86400);
  const h = Math.floor((sec % 86400) / 3600);
  const m = Math.floor((sec % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`;
  if (h > 0) return `${h}h ${m}m`;
  return `${m}m`;
}

export default function SystemPage() {
  const [info, setInfo] = useState<SystemInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetch = useCallback(async () => {
    try {
      setInfo(await getSystemInfo());
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetch(); }, [fetch]);

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">系统健康</h1>
        <button onClick={fetch} className="px-3 py-1.5 text-xs bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700">
          刷新
        </button>
      </div>

      {loading && <div className="text-gray-400 text-sm">加载中...</div>}
      {error && <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 text-sm">{error}</div>}

      {info && (
        <>
          {/* 核心状态 */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <HealthBadge label="Milvus 连接" ok={info.milvus_connected} okText="已连接" failText="断开" />
            <HealthBadge label="服务运行" ok={info.uptime_seconds > 0} okText={fmtUptime(info.uptime_seconds)} failText="—" />
            <MetricBadge label="Python" value={info.python_version} />
            <MetricBadge label="平台" value={info.platform.includes("Darwin") ? "macOS" : info.platform.split("-")[0]} />
          </div>

          {/* 内存 */}
          <Section title="内存">
            <ProgressBar
              label="已用"
              used={info.memory_used_mb / 1024}
              total={info.memory_total_mb / 1024}
              unit="GB"
              color={info.memory_percent > 80 ? "red" : info.memory_percent > 60 ? "yellow" : "blue"}
            />
            <div className="mt-1 text-xs text-gray-400 dark:text-gray-500">
              {(info.memory_used_mb / 1024).toFixed(2)} / {(info.memory_total_mb / 1024).toFixed(2)} GB ({info.memory_percent}%)
            </div>
          </Section>

          {/* 磁盘 */}
          <Section title="磁盘">
            <ProgressBar
              label="已用"
              used={info.disk_used_gb}
              total={info.disk_total_gb}
              unit="GB"
              color={info.disk_percent > 80 ? "red" : info.disk_percent > 60 ? "yellow" : "blue"}
            />
            <div className="mt-1 text-xs text-gray-400 dark:text-gray-500">
              {info.disk_used_gb.toFixed(2)} / {info.disk_total_gb.toFixed(2)} GB ({info.disk_percent}%)
            </div>
          </Section>
        </>
      )}
    </div>
  );
}

function HealthBadge({ label, ok, okText, failText }: { label: string; ok: boolean; okText: string; failText: string }) {
  return (
    <div className={`p-4 rounded-xl border ${ok ? "bg-green-50 dark:bg-green-900/20 border-green-100 dark:border-green-800" : "bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800"}`}>
      <div className="text-sm font-medium">{ok ? okText : failText}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{label}</div>
    </div>
  );
}

function MetricBadge({ label, value }: { label: string; value: string }) {
  return (
    <div className="p-4 rounded-xl border bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700">
      <div className="text-sm font-mono text-gray-700 dark:text-gray-300 truncate">{value}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{label}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="p-5 rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm">
      <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-3">{title}</h2>
      {children}
    </div>
  );
}

function ProgressBar({ label, used, total, unit, color }: { label: string; used: number; total: number; unit: string; color: string }) {
  const pct = total > 0 ? Math.min((used / total) * 100, 100) : 0;
  const barColor = { blue: "bg-blue-500", yellow: "bg-yellow-500", red: "bg-red-500" }[color] || "bg-blue-500";
  return (
    <div>
      <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mb-1">
        <span>{label}</span>
        <span>{used.toFixed(1)} / {total.toFixed(1)} {unit}</span>
      </div>
      <div className="h-2.5 bg-gray-100 dark:bg-gray-700 rounded-full overflow-hidden">
        <div className={`h-full ${barColor} rounded-full transition-all duration-500`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  );
}
