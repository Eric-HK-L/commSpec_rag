"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import { search, askStream, submitFeedback, getStats, type SearchResult, type SourceItem } from "@/lib/api";
import { useConversationHistory } from "@/lib/useConversationHistory";

// ── SSE 事件解析 ──
interface StreamState {
  answer: string;
  sources: SourceItem[];
  done: boolean;
  error: string | null;
}

function parseSSELine(line: string): Partial<StreamState> | null {
  if (!line.startsWith("data: ")) return null;
  try {
    const evt = JSON.parse(line.slice(6));
    switch (evt.type) {
      case "sources":
        return { sources: evt.data || [] };
      case "chunk":
        return { answer: evt.content || "" };
      case "done":
        return { done: true };
      case "error":
        return { error: evt.detail || "未知错误" };
    }
  } catch {
    /* ignore malformed */
  }
  return null;
}

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<SourceItem[]>([]);
  const [warnings, setWarnings] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const savedRef = useRef(false);  // 防止重复保存
  const currentQueryRef = useRef("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [feedbackRating, setFeedbackRating] = useState<"up" | "down" | null>(null);
  const [availableReleases, setAvailableReleases] = useState<string[]>([]);

  useEffect(() => {
    getStats().then(s => setAvailableReleases(Object.keys(s.releases))).catch(() => {});
  }, []);

  const { history, addEntry, removeEntry, clearHistory } = useConversationHistory();

  // SSE 完成后保存对话
  useEffect(() => {
    if (!loading && !streaming && answer && currentQueryRef.current && !savedRef.current) {
      addEntry({
        query: currentQueryRef.current,
        answer,
        sources: sources.map((s) => ({
          spec_number: s.spec_number,
          section_id: s.section_id || "",
          text: s.text,
          score: s.score,
        })),
      });
      savedRef.current = true;
    }
    if (loading) {
      savedRef.current = false;
    }
  }, [loading, streaming, answer, sources, addEntry]);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    const q = query.trim();
    if (!q || loading) return;

    setLoading(true);
    setStreaming(false);
    setAnswer("");
    setSources([]);
    setWarnings([]);
    setError(null);
    setResults([]);
    setFeedbackRating(null);
    currentQueryRef.current = q;
    savedRef.current = false;

    try {
      // 1. 检索
      const searchRes = await search(q, 10);
      setResults(searchRes.results);

      // 2. SSE 流式问答
      setStreaming(true);
      const controller = new AbortController();
      abortRef.current = controller;

      const res = await askStream(q, 10);
      if (!res.ok) throw new Error(`SSE ${res.status}`);
      if (!res.body) throw new Error("无响应流");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const parsed = parseSSELine(line);
          if (!parsed) continue;
          if (parsed.answer) setAnswer((prev) => prev + parsed.answer);
          if (parsed.sources) setSources(parsed.sources);
          if (parsed.done) { setStreaming(false); setLoading(false); }
          if (parsed.error) { setError(parsed.error); setStreaming(false); setLoading(false); }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "请求失败");
    } finally {
      setLoading(false);
      setStreaming(false);
    }
  }, [query, loading]);

  const handleStop = () => {
    abortRef.current?.abort();
    setLoading(false);
    setStreaming(false);
  };

  const handleFeedback = async (rating: "up" | "down") => {
    if (feedbackRating) return; // 防止重复提交
    setFeedbackRating(rating);
    try {
      await submitFeedback({
        query: currentQueryRef.current || query,
        answer,
        sources,
        rating,
      });
    } catch (err) {
      console.error("反馈提交失败:", err);
      setFeedbackRating(null); // 失败时恢复
    }
  };

  return (
    <div className="flex-1 flex">
      {/* 对话历史侧边栏 */}
      {historyOpen && (
        <aside className="w-64 flex-shrink-0 border-r border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-950 overflow-y-auto transition-colors">
          <div className="p-3 flex items-center justify-between border-b border-gray-200 dark:border-gray-700">
            <span className="text-sm font-semibold text-gray-600 dark:text-gray-300">对话历史</span>
            <button
              onClick={() => setHistoryOpen(false)}
              className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 transition-colors"
            >
              ✕
            </button>
          </div>
          {history.length === 0 ? (
            <p className="p-4 text-xs text-gray-400 dark:text-gray-500 text-center">暂无历史记录</p>
          ) : (
            <>
              <div className="p-3 space-y-2">
                {history.map((entry) => (
                  <div
                    key={entry.id}
                    className="group p-2 rounded-lg bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 cursor-pointer
                               hover:border-blue-300 dark:hover:border-blue-700 transition-colors"
                    onClick={() => {
                      setQuery(entry.query);
                      setHistoryOpen(false);
                    }}
                  >
                    <p className="text-xs text-gray-700 dark:text-gray-300 line-clamp-2 font-medium">{entry.query}</p>
                    <div className="flex items-center justify-between mt-1">
                      <span className="text-[10px] text-gray-400 dark:text-gray-500">
                        {entry.sources.length} 源 · {new Date(entry.timestamp).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit" })}
                      </span>
                      <button
                        onClick={(e) => { e.stopPropagation(); removeEntry(entry.id); }}
                        className="opacity-0 group-hover:opacity-100 text-[10px] text-red-400 hover:text-red-600 transition-all"
                      >
                        删除
                      </button>
                    </div>
                  </div>
                ))}
              </div>
              {history.length > 0 && (
                <div className="p-2 border-t border-gray-200 dark:border-gray-700">
                  <button
                    onClick={clearHistory}
                    className="w-full text-xs text-gray-400 hover:text-red-500 transition-colors py-1"
                  >
                    清空全部历史
                  </button>
                </div>
              )}
            </>
          )}
        </aside>
      )}

      <div className="flex-1">
      {/* 历史按钮 (有结果时) */}
      {!historyOpen && history.length > 0 && (
        <button
          onClick={() => setHistoryOpen(true)}
          className="absolute top-20 left-4 px-3 py-2 rounded-lg text-xs text-gray-400 dark:text-gray-500
                     hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors z-10"
          title="对话历史"
        >
          📋 历史 ({history.length})
        </button>
      )}
      {/* 无结果时: Hero 区 */}
      {!answer && results.length === 0 && !loading && (
        <div className="max-w-3xl mx-auto px-4 pt-24 pb-12 text-center">
          {/* 历史按钮 */}
          <button
            onClick={() => setHistoryOpen(!historyOpen)}
            className="absolute top-20 left-4 px-3 py-2 rounded-lg text-xs text-gray-400 dark:text-gray-500
                       hover:text-blue-600 dark:hover:text-blue-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
            title="对话历史"
          >
            📋 历史{history.length > 0 && ` (${history.length})`}
          </button>
          <h1 className="text-4xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight mb-3">
            3GPP 通信规范智能问答
          </h1>
          <p className="text-xs text-gray-400 dark:text-gray-500 mb-8">
            基于 RAG 技术，精准检索 {" "}
            {availableReleases.length > 0
              ? availableReleases.join("、") + " "
              : "3GPP "}
            全部规范文档，获取带溯源的即时答案
          </p>

          {/* 搜索框 */}
          <form onSubmit={handleSubmit} className="flex gap-3 max-w-2xl mx-auto mb-8">
            <div className="flex-1 relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="输入问题，如：NR RLC 与 LTE RLC 的区别是什么？"
                className="w-full px-5 py-3.5 rounded-2xl border border-gray-300 dark:border-gray-600
                           bg-white dark:bg-gray-900 shadow-lg shadow-gray-200/50 dark:shadow-black/30
                           focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                           text-base text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500
                           transition-shadow"
                disabled={loading}
              />
            </div>
            <button
              type="submit"
              disabled={!query.trim()}
              className="px-8 py-3.5 rounded-2xl bg-blue-600 dark:bg-blue-500 text-white font-semibold
                         shadow-lg shadow-blue-600/25 dark:shadow-blue-500/25
                         hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-30 disabled:cursor-not-allowed
                         disabled:shadow-none transition-all active:scale-95"
            >
              搜索
            </button>
          </form>

          {/* 示例查询 */}
          <div className="flex flex-wrap justify-center gap-2 max-w-2xl mx-auto">
            <span className="text-xs text-gray-400 dark:text-gray-500 self-center mr-1">试试：</span>
            {[
              "NR PRACH preamble 格式有哪些？",
              "NR PUSCH 的 DMRS 配置方式",
              "NR SSB 的时频结构",
              "NR BWP 的配置方式与作用",
            ].map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => { setQuery(q); }}
                className="px-3 py-1.5 rounded-full text-xs bg-gray-100 dark:bg-gray-800
                           text-gray-600 dark:text-gray-300
                           hover:bg-blue-50 dark:hover:bg-blue-900/30 hover:text-blue-700 dark:hover:text-blue-300
                           border border-gray-200 dark:border-gray-700 hover:border-blue-200 dark:hover:border-blue-700
                           transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* 有结果时: 紧凑搜索栏 + 结果区 */}
      {(answer || results.length > 0 || loading) && (
        <div className="max-w-6xl mx-auto px-4 py-6 space-y-6">
          {/* 紧凑搜索栏 */}
          <form onSubmit={handleSubmit} className="flex gap-3">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="输入 3GPP 问题继续探索..."
              className="flex-1 px-4 py-2.5 rounded-xl border border-gray-300 dark:border-gray-600
                         bg-white dark:bg-gray-900 shadow-sm
                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                         text-base text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500"
              disabled={loading}
            />
            {loading ? (
              <button
                type="button"
                onClick={handleStop}
                className="px-6 py-2.5 rounded-xl bg-red-500 text-white font-medium hover:bg-red-600 transition-colors"
              >
                停止
              </button>
            ) : (
              <button
                type="submit"
                disabled={!query.trim()}
                className="px-6 py-2.5 rounded-xl bg-blue-600 dark:bg-blue-500 text-white font-medium
                           hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                搜索
              </button>
            )}
          </form>

          {/* 错误 */}
          {error && (
            <div className="p-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm border border-red-200 dark:border-red-800">
              {error}
            </div>
          )}

          {/* 检索结果 + 回答 双栏 */}
          <div className={`grid gap-6 ${answer || sources.length ? "lg:grid-cols-[1fr_2fr]" : "grid-cols-1"}`}>
            {/* 左栏: 检索结果 */}
            {results.length > 0 && (
              <div className="space-y-3">
                <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                  检索结果 ({results.length})
                </h2>
                {results.map((r, i) => (
                  <div key={r.chunk_id} className="p-3 rounded-lg bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm text-sm">
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-mono text-xs text-blue-600 dark:text-blue-400">
                        {r.spec_number} {r.parent_section_id ? `§${r.parent_section_id}` : ""}
                      </span>
                      <span className="text-xs text-gray-400 dark:text-gray-500">{(r.score * 100).toFixed(0)}%</span>
                    </div>
                    <p className="text-gray-700 dark:text-gray-300 line-clamp-3 leading-relaxed">{r.text.slice(0, 200)}</p>
                  </div>
                ))}
              </div>
            )}

            {/* 右栏: LLM 回答 */}
            {(answer || streaming) && (
              <div className="space-y-3">
                <h2 className="text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                  回答 {streaming && <span className="animate-pulse text-blue-500">●</span>}
                </h2>
                <div className="p-5 rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm">
                  <div className="prose prose-sm dark:prose-invert max-w-none leading-relaxed whitespace-pre-wrap">
                    {answer || (streaming ? "思考中..." : "")}
                  </div>

                  {/* 反馈按钮 */}
                  {answer && !streaming && (
                    <div className="mt-4 pt-3 border-t border-gray-100 dark:border-gray-800 flex items-center gap-2">
                      <span className="text-xs text-gray-400 dark:text-gray-500 mr-1">评价:</span>
                      <button
                        onClick={() => handleFeedback("up")}
                        disabled={feedbackRating !== null}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          feedbackRating === "up"
                            ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-300 dark:border-green-700"
                            : feedbackRating === null
                              ? "bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700 hover:bg-green-50 dark:hover:bg-green-900/20 hover:text-green-600 dark:hover:text-green-400 hover:border-green-200 dark:hover:border-green-800"
                              : "bg-gray-50 dark:bg-gray-800 text-gray-300 dark:text-gray-600 border border-gray-200 dark:border-gray-700 cursor-not-allowed"
                        }`}
                        title="回答有帮助"
                      >
                        👍 有用
                      </button>
                      <button
                        onClick={() => handleFeedback("down")}
                        disabled={feedbackRating !== null}
                        className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                          feedbackRating === "down"
                            ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border border-red-300 dark:border-red-700"
                            : feedbackRating === null
                              ? "bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 dark:hover:text-red-400 hover:border-red-200 dark:hover:border-red-800"
                              : "bg-gray-50 dark:bg-gray-800 text-gray-300 dark:text-gray-600 border border-gray-200 dark:border-gray-700 cursor-not-allowed"
                        }`}
                        title="回答有误"
                      >
                        👎 没用
                      </button>
                      {feedbackRating && (
                        <span className="text-xs text-gray-400 dark:text-gray-500 ml-1">反馈已提交</span>
                      )}
                    </div>
                  )}
                </div>

                {/* 溯源卡片 */}
                {sources.length > 0 && (
                  <details className="group">
                    <summary className="text-sm text-gray-500 dark:text-gray-400 cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors">
                      溯源依据 ({sources.length} 条)
                    </summary>
                    <div className="mt-2 space-y-2">
                      {sources.map((s, i) => (
                        <div key={i} className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 text-xs">
                          <div className="font-semibold text-blue-700 dark:text-blue-300 mb-1">
                            {s.spec_number} · {s.section_id || "—"} · {(s.score * 100).toFixed(0)}%
                          </div>
                          <p className="text-gray-600 dark:text-gray-400 line-clamp-3">{s.text}</p>
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        </div>
      )}
      </div>
    </div>
  );
}
