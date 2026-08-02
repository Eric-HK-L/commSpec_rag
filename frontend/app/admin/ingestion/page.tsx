"use client";

import { useState, useEffect, useCallback } from "react";
import { getIngestStatus, triggerIngestion, type IngestStatus } from "@/lib/api";

export default function IngestionPage() {
  const [status, setStatus] = useState<IngestStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [triggering, setTriggering] = useState(false);
  const [msg, setMsg] = useState<{ type: "success" | "warning" | "error"; text: string } | null>(null);
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [source, setSource] = useState<"marked" | "original" | "all">("marked");

  const fetchStatus = useCallback(async () => {
    try {
      const s = await getIngestStatus(80);
      setStatus(s);
      if (!s.running) setAutoRefresh(false);
    } catch (e) {
      /* ignore */
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  useEffect(() => {
    if (!autoRefresh) return;
    const t = setInterval(fetchStatus, 3000);
    return () => clearInterval(t);
  }, [autoRefresh, fetchStatus]);

  const handleTrigger = async (mode: "incremental" | "full") => {
    setTriggering(true);
    setMsg(null);
    try {
      const res = await triggerIngestion(mode, source);
      setMsg(res.accepted ? { type: "success", text: `任务已启动 (PID: ${res.pid}, 数据源: ${SOURCE_LABEL[source]})` } : { type: "warning", text: res.message });
      if (res.accepted) setAutoRefresh(true);
    } catch (e) {
      setMsg({ type: "error", text: e instanceof Error ? e.message : "触发失败" });
    } finally {
      setTriggering(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto px-4 py-8 space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">摄入管理</h1>

      {loading && !status && (
        <div className="p-8 text-gray-500 dark:text-gray-400">加载中...</div>
      )}

      {/* 数据源选择 */}
      <div className="space-y-2">
        <div className="text-sm font-medium text-gray-700 dark:text-gray-300">数据源</div>
        <div className="flex gap-2">
          {(Object.keys(SOURCE_LABEL) as Array<"marked" | "original" | "all">).map((s) => (
            <button
              key={s}
              onClick={() => setSource(s)}
              className={`px-4 py-2 text-sm rounded-xl border transition-colors ${
                source === s
                  ? "bg-blue-600 dark:bg-blue-500 border-blue-600 dark:border-blue-500 text-white font-medium"
                  : "bg-white dark:bg-gray-900 border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800"
              }`}
            >
              {SOURCE_LABEL[s]}
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          marked/ 为 Markdown 协议数据集（默认嵌入源）；original/ 为原始 DOCX（pandoc 处理）；all 为两者合并（marked 优先）。
        </p>
      </div>

      {/* 触发按钮 */}
      <div className="flex gap-3 items-center">
        <button
          onClick={() => handleTrigger("incremental")}
          disabled={triggering || status?.running}
          className="px-5 py-2.5 rounded-xl bg-blue-600 dark:bg-blue-500 text-white font-medium text-sm hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-40 transition-colors"
        >
          {triggering ? "触发中..." : "增量摄入"}
        </button>
        <button
          onClick={() => handleTrigger("full")}
          disabled={triggering || status?.running}
          className="px-5 py-2.5 rounded-xl bg-red-600 text-white font-medium text-sm hover:bg-red-700 disabled:opacity-40 transition-colors"
        >
          全量重建
        </button>
        <button onClick={fetchStatus} className="px-4 py-2 text-sm bg-gray-100 dark:bg-gray-800 dark:text-gray-300 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
          刷新
        </button>
      </div>

      {msg && (
        <div className={`p-3 rounded-lg text-sm ${
          msg.type === "success" ? "bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-300" :
          msg.type === "warning" ? "bg-yellow-50 dark:bg-yellow-900/20 text-yellow-700 dark:text-yellow-300" :
          "bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-300"
        }`}>
          {msg.text}
        </div>
      )}

      {/* 状态 */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <Badge label="运行状态" value={status.running ? "运行中" : "空闲"} color={status.running ? "green" : "gray"} />
          <Badge label="PID" value={status.pid ? String(status.pid) : "—"} color="gray" />
          <Badge label="日志行数" value={`${status.log_tail.length} 行`} color="gray" />
          <Badge label="上次摄入" value={status.last_ingestion_at ? new Date(status.last_ingestion_at).toLocaleString("zh-CN") : "—"} color="gray" />
        </div>
      )}

      {/* 日志 */}
      {status && (
        <div className="rounded-xl border border-gray-700 dark:border-gray-600 bg-gray-900 dark:bg-black overflow-hidden shadow-sm">
          <div className="px-4 py-2 bg-gray-800 dark:bg-gray-800 text-xs text-gray-400 font-mono flex justify-between">
            <span>摄入日志</span>
            <span>{status.log_tail.length} 行</span>
          </div>
          <pre className="p-4 text-xs text-green-400 font-mono leading-relaxed max-h-[500px] overflow-auto">
            {status.log_tail.length === 0 ? "暂无日志" : status.log_tail.join("\n")}
          </pre>
        </div>
      )}
    </div>
  );
}

const SOURCE_LABEL: Record<"marked" | "original" | "all", string> = {
  marked: "marked/ (Markdown 数据集)",
  original: "original/ (原始 DOCX)",
  all: "全部 (marked 优先)",
};

function Badge({ label, value, color }: { label: string; value: string; color: string }) {
  const c: Record<string, string> = {
    gray: "bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700",
    green: "bg-green-50 dark:bg-green-900/20 border-green-100 dark:border-green-800",
    red: "bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800",
  };
  const cls = c[color] || c.gray;
  return (
    <div className={`p-3 rounded-lg border ${cls}`}>
      <div className="text-xs text-gray-400 dark:text-gray-500">{label}</div>
      <div className="font-medium text-gray-700 dark:text-gray-300 text-sm mt-0.5">{value}</div>
    </div>
  );
}
