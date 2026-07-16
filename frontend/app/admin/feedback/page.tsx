"use client";

import { useState, useEffect, useCallback } from "react";
import { getFeedbackList, getFeedbackStats, type FeedbackItem, type FeedbackStats } from "@/lib/api";

export default function FeedbackPage() {
  const [stats, setStats] = useState<FeedbackStats | null>(null);
  const [items, setItems] = useState<FeedbackItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filter, setFilter] = useState("");
  const [page, setPage] = useState(0);

  const fetch = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [s, list] = await Promise.all([
        getFeedbackStats(),
        getFeedbackList(filter || undefined, 30, page * 30),
      ]);
      setStats(s);
      setItems(list);
    } catch (e) {
      setError(e instanceof Error ? e.message : "加载失败");
    } finally {
      setLoading(false);
    }
  }, [filter, page]);

  useEffect(() => { fetch(); }, [fetch]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">💬 反馈分析</h1>

      {/* 统计卡片 */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <StatCard label="总反馈" value={String(stats.total)} color="gray" />
          <StatCard label="👍 好评" value={String(stats.up)} color="green" />
          <StatCard label="👎 差评" value={String(stats.down)} color="red" />
          <StatCard label="好评率" value={`${(stats.up_ratio * 100).toFixed(0)}%`} color={stats.up_ratio >= 0.6 ? "green" : "red"} />
        </div>
      )}

      {/* 筛选 */}
      <div className="flex gap-3 items-center">
        <select
          value={filter}
          onChange={(e) => { setFilter(e.target.value); setPage(0); }}
          className="px-3 py-2 rounded-lg border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm"
        >
          <option value="">全部评分</option>
          <option value="up">👍 好评</option>
          <option value="down">👎 差评</option>
        </select>
        <button onClick={fetch} className="px-3 py-2 text-sm bg-gray-100 dark:bg-gray-800 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors">
          刷新
        </button>
      </div>

      {error && <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 text-sm">{error}</div>}

      {/* 列表 */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 dark:bg-gray-800 text-left text-gray-500 dark:text-gray-400">
              <tr>
                <th className="px-4 py-3 w-16">评分</th>
                <th className="px-4 py-3">查询</th>
                <th className="px-4 py-3 hidden md:table-cell w-96">回答</th>
                <th className="px-4 py-3 w-40">时间</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
              {items.map((item) => (
                <tr key={item.id} className="hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
                  <td className="px-4 py-2.5 text-center text-lg">
                    {item.rating === "up" ? "👍" : "👎"}
                  </td>
                  <td className="px-4 py-2.5 text-gray-800 dark:text-gray-200 max-w-[200px] truncate font-medium">
                    {item.query}
                  </td>
                  <td className="px-4 py-2.5 hidden md:table-cell text-gray-500 dark:text-gray-400 max-w-[300px] truncate">
                    {item.answer.slice(0, 120)}...
                  </td>
                  <td className="px-4 py-2.5 text-xs text-gray-400 dark:text-gray-500 whitespace-nowrap">
                    {new Date(item.created_at).toLocaleString("zh-CN")}
                  </td>
                </tr>
              ))}
              {items.length === 0 && !loading && (
                <tr><td colSpan={4} className="px-4 py-10 text-center text-gray-400">暂无反馈数据</td></tr>
              )}
            </tbody>
          </table>
        </div>

        {/* 分页 */}
        {stats && stats.total > 30 && (
          <div className="px-4 py-3 border-t border-gray-100 dark:border-gray-700 flex items-center justify-between">
            <span className="text-xs text-gray-400">共 {stats.total} 条</span>
            <div className="flex gap-2">
              <button
                onClick={() => setPage(Math.max(0, page - 1))}
                disabled={page === 0}
                className="px-3 py-1 text-xs rounded bg-gray-100 dark:bg-gray-800 disabled:opacity-30"
              >
                上一页
              </button>
              <span className="text-xs text-gray-500 self-center">第 {page + 1} 页</span>
              <button
                onClick={() => setPage(page + 1)}
                disabled={(page + 1) * 30 >= stats.total}
                className="px-3 py-1 text-xs rounded bg-gray-100 dark:bg-gray-800 disabled:opacity-30"
              >
                下一页
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function StatCard({ label, value, color }: { label: string; value: string; color: string }) {
  const bg: Record<string, string> = {
    gray: "bg-gray-50 dark:bg-gray-900 border-gray-200 dark:border-gray-700",
    green: "bg-green-50 dark:bg-green-900/20 border-green-100 dark:border-green-800",
    red: "bg-red-50 dark:bg-red-900/20 border-red-100 dark:border-red-800",
  };
  const tc: Record<string, string> = {
    gray: "text-gray-700 dark:text-gray-300",
    green: "text-green-700 dark:text-green-300",
    red: "text-red-700 dark:text-red-300",
  };
  return (
    <div className={`p-4 rounded-xl border ${bg[color]}`}>
      <div className={`text-2xl font-bold ${tc[color]}`}>{value}</div>
      <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">{label}</div>
    </div>
  );
}
