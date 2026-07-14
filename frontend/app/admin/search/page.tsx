"use client";

import { useState } from "react";
import { search, ask, searchBatch, type SearchResult, type AskResponse } from "@/lib/api";

const PRESET_QUERIES = [
  "NR PRACH preamble 格式",
  "NR PUSCH DMRS 配置",
  "NR SSB 时频结构",
  "NR BWP 配置方式",
];

export default function SearchTestPage() {
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(10);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [answer, setAnswer] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState<{ search: number; ask: number } | null>(null);

  // 批量对比模式
  const [batchMode, setBatchMode] = useState(false);
  const [batchResults, setBatchResults] = useState<{ query: string; total: number; results: SearchResult[] }[]>([]);
  const [batchLoading, setBatchLoading] = useState(false);

  const handleTest = async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q) return;

    setLoading(true);
    setError(null);
    setResults([]);
    setAnswer(null);
    setElapsed(null);

    try {
      const t0 = performance.now();
      const [searchRes, askRes] = await Promise.all([
        search(q, topK),
        ask(q, topK),
      ]);
      const t1 = performance.now();
      const t2 = performance.now();

      setResults(searchRes.results);
      setAnswer(askRes);
      setElapsed({ search: t1 - t0, ask: t2 - t0 });
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败");
    } finally {
      setLoading(false);
    }
  };

  const handleBatch = async () => {
    setBatchLoading(true);
    setBatchResults([]);
    try {
      const res = await searchBatch(
        PRESET_QUERIES.map((q) => ({ query: q, top_k: 5 }))
      );
      setBatchResults(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : "批量检索失败");
    } finally {
      setBatchLoading(false);
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-4 py-8 space-y-6">
      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">🔍 搜索测试台</h1>

      <form onSubmit={handleTest} className="flex gap-3 items-center">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="输入测试查询..."
          className="flex-1 px-4 py-3 rounded-xl border border-gray-300 dark:border-gray-600
                     bg-white dark:bg-gray-900 shadow-sm
                     focus:outline-none focus:ring-2 focus:ring-blue-500
                     text-base text-gray-900 dark:text-gray-100
                     placeholder:text-gray-400 dark:placeholder:text-gray-500"
          disabled={loading}
        />
        <select
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          className="px-3 py-3 rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-sm text-gray-700 dark:text-gray-300"
        >
          {[5, 10, 20, 50].map((k) => (
            <option key={k} value={k}>Top-{k}</option>
          ))}
        </select>
        <button
          type="submit"
          disabled={!query.trim() || loading}
          className="px-8 py-3 rounded-xl bg-blue-600 dark:bg-blue-500 text-white font-medium hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-40 transition-colors"
        >
          {loading ? "测试中..." : "搜索"}
        </button>
      </form>

      {/* 耗时 */}
      {elapsed && (
        <div className="flex gap-4 text-xs text-gray-500 dark:text-gray-400">
          <span>🔍 检索: {(elapsed.search).toFixed(0)}ms</span>
          <span>🤖 LLM: {(elapsed.ask).toFixed(0)}ms</span>
        </div>
      )}

      {/* 批量对比模式 */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
        <div className="flex items-center gap-3 mb-3">
          <button
            onClick={() => { setBatchMode(!batchMode); if (!batchMode) handleBatch(); }}
            className="px-4 py-2 rounded-lg text-sm font-medium
                       bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300
                       hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          >
            {batchMode ? "收起批量对比" : "📊 批量对比 (NR L1)"}
          </button>
          {batchLoading && <span className="text-sm text-gray-400 dark:text-gray-500">加载中...</span>}
        </div>

        {batchMode && batchResults.length > 0 && (
          <div className="grid gap-3 sm:grid-cols-2">
            {batchResults.map((br, i) => (
              <div key={i} className="p-3 rounded-lg bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-sm">
                <div className="font-medium text-gray-800 dark:text-gray-200 mb-2">{br.query}</div>
                <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">{br.total} 条结果</div>
                {br.results.slice(0, 3).map((r, j) => (
                  <div key={j} className="text-xs text-gray-600 dark:text-gray-400 truncate">
                    <span className="font-mono text-blue-600 dark:text-blue-400">{r.spec_number}</span>
                    {" "}{r.text.slice(0, 80)}...
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>

      {error && <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm">{error}</div>}

      <div className="grid gap-6 lg:grid-cols-2">
        {/* 检索结果 */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase">检索结果 ({results.length})</h2>
          {results.map((r, i) => (
            <div key={r.chunk_id} className="p-3 rounded-lg bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm text-sm">
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-xs text-blue-600 dark:text-blue-400">
                  #{i + 1} {r.spec_number} {r.parent_section_id ? `§${r.parent_section_id}` : ""}
                </span>
                <span className="text-xs font-mono text-gray-400 dark:text-gray-500">{(r.score * 100).toFixed(1)}%</span>
              </div>
              <p className="text-gray-700 dark:text-gray-300 text-xs leading-relaxed line-clamp-3">{r.text.slice(0, 250)}</p>
            </div>
          ))}
          {results.length === 0 && !loading && (
            <div className="p-8 text-center text-gray-400 dark:text-gray-500 text-sm">输入查询后显示检索结果</div>
          )}
        </div>

        {/* LLM 回答 */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase">
            LLM 回答 {answer && <span className="font-normal text-gray-400 dark:text-gray-500">(已验证: {answer.verified ? "✅" : "⚠️"})</span>}
          </h2>
          {answer ? (
            <>
              <div className="p-5 rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm">
                <div className="prose prose-sm dark:prose-invert max-w-none leading-relaxed whitespace-pre-wrap">
                  {answer.answer}
                </div>
              </div>

              {answer.warnings.length > 0 && (
                <div className="p-3 rounded-lg bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 text-sm text-yellow-700 dark:text-yellow-300">
                  {answer.warnings.map((w, i) => <div key={i}>⚠️ {w}</div>)}
                </div>
              )}

              <details className="text-sm">
                <summary className="text-gray-500 dark:text-gray-400 cursor-pointer hover:text-blue-600 dark:hover:text-blue-400">
                  溯源依据 ({answer.sources.length}) · 覆盖率 {answer.coverage}%
                </summary>
                <div className="mt-2 space-y-2">
                  {answer.sources.map((s, i) => (
                    <div key={i} className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 text-xs">
                      <div className="font-semibold text-blue-700 dark:text-blue-300 mb-1">
                        {s.spec_number} · {s.section_id || "—"} · {(s.score * 100).toFixed(0)}%
                      </div>
                      <p className="text-gray-600 dark:text-gray-400">{s.text}</p>
                    </div>
                  ))}
                </div>
              </details>
            </>
          ) : (
            <div className="p-8 text-center text-gray-400 dark:text-gray-500 text-sm">
              {loading ? "请求中..." : "输入查询后显示 LLM 回答"}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
