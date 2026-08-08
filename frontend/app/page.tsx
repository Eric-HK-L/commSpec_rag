"use client";

import { useState, useRef, useCallback, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  search,
  askStream,
  submitFeedback,
  getStats,
  type SearchResult,
  type SourceItem,
} from "@/lib/api";
import { useConversationHistory } from "@/lib/useConversationHistory";

// ── 类型 ──
interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: SourceItem[];
  warnings?: string[];
  verified?: boolean;
}

// ── SSE 事件解析 ──
interface StreamState {
  answer: string;
  sources: SourceItem[];
  done: boolean;
  error: string | null;
  warnings: string[];
  verified: boolean;
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
        return {
          done: true,
          answer: evt.answer || "",
          warnings: evt.warnings || [],
          verified: evt.verified ?? false,
        };
      case "error":
        return { error: evt.detail || "未知错误" };
    }
  } catch {
    /* ignore malformed */
  }
  return null;
}

// ── Markdown 渲染组件 ──
function Markdown({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      components={{
        h1: ({ children }) => (
          <h1 className="text-xl font-bold mt-6 mb-3 text-gray-900 dark:text-gray-100">
            {children}
          </h1>
        ),
        h2: ({ children }) => (
          <h2 className="text-lg font-semibold mt-5 mb-2 text-gray-900 dark:text-gray-100">
            {children}
          </h2>
        ),
        h3: ({ children }) => (
          <h3 className="text-base font-semibold mt-4 mb-1.5 text-gray-800 dark:text-gray-200">
            {children}
          </h3>
        ),
        p: ({ children }) => <p className="my-2 leading-relaxed">{children}</p>,
        ul: ({ children }) => (
          <ul className="list-disc pl-5 my-2 space-y-0.5">{children}</ul>
        ),
        ol: ({ children }) => (
          <ol className="list-decimal pl-5 my-2 space-y-0.5">{children}</ol>
        ),
        li: ({ children }) => <li className="leading-relaxed">{children}</li>,
        strong: ({ children }) => (
          <strong className="font-semibold text-gray-900 dark:text-gray-100">
            {children}
          </strong>
        ),
        code: ({ children, className }) => {
          const isInline = !className;
          return isInline ? (
            <code className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-sm font-mono text-pink-600 dark:text-pink-400">
              {children}
            </code>
          ) : (
            <code className="block p-3 my-2 rounded-lg bg-gray-100 dark:bg-gray-800 text-sm font-mono overflow-x-auto">
              {children}
            </code>
          );
        },
        pre: ({ children }) => <pre className="my-2">{children}</pre>,
        blockquote: ({ children }) => (
          <blockquote className="border-l-3 border-blue-400 dark:border-blue-500 pl-4 my-2 italic text-gray-600 dark:text-gray-400">
            {children}
          </blockquote>
        ),
        table: ({ children }) => (
          <div className="overflow-x-auto my-3">
            <table className="min-w-full border-collapse border border-gray-200 dark:border-gray-700 text-sm">
              {children}
            </table>
          </div>
        ),
        th: ({ children }) => (
          <th className="border border-gray-200 dark:border-gray-700 px-3 py-1.5 bg-gray-50 dark:bg-gray-800 font-medium text-left">
            {children}
          </th>
        ),
        td: ({ children }) => (
          <td className="border border-gray-200 dark:border-gray-700 px-3 py-1.5">
            {children}
          </td>
        ),
        hr: () => <hr className="my-4 border-gray-200 dark:border-gray-700" />,
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

// ── 示例查询 ──
const EXAMPLE_QUERIES = [
  "NR PRACH preamble 格式有哪些？",
  "NR PUSCH 的 DMRS 配置方式",
  "NR SSB 的时频结构",
  "NR BWP 的配置方式与作用",
];

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [currentAnswer, setCurrentAnswer] = useState("");
  const [currentSources, setCurrentSources] = useState<SourceItem[]>([]);
  const [currentWarnings, setCurrentWarnings] = useState<string[]>([]);
  const [currentVerified, setCurrentVerified] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const currentQueryRef = useRef("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [feedbackRating, setFeedbackRating] = useState<"up" | "down" | null>(
    null
  );
  const [availableReleases, setAvailableReleases] = useState<string[]>([]);
  const [availableSeries, setAvailableSeries] = useState<string[]>([]);
  const [availableDocTypes, setAvailableDocTypes] = useState<string[]>([]);
  const [selectedRelease, setSelectedRelease] = useState<string>("");
  const [selectedSeries, setSelectedSeries] = useState<string>("");
  const [selectedDocType, setSelectedDocType] = useState<string>("");
  const [rerankerEnabled, setRerankerEnabled] = useState(true);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    getStats()
      .then((s) => {
        setAvailableReleases(Object.keys(s.releases));
        setAvailableSeries(
          s.available_series || Object.keys(s.series_distribution)
        );
        setAvailableDocTypes(Object.keys(s.doc_types || {}));
      })
      .catch(() => {});
  }, []);

  const { history, addEntry, removeEntry, clearHistory } =
    useConversationHistory();

  // 自动滚到底部
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [currentAnswer, messages]);

  const handleSubmit = useCallback(
    async (e: React.FormEvent, presetQuery?: string) => {
      e.preventDefault();
      const q = (presetQuery ?? query).trim();
      if (!q || loading) return;

      // 添加用户消息
      const userMsg: Message = { role: "user", content: q };
      setMessages((prev) => [...prev, userMsg]);
      // 如果用的是 presetQuery（示例按钮），同步清空输入框
      if (presetQuery) setQuery("");
      else setQuery("");
      setLoading(true);
      setStreaming(false);
      setCurrentAnswer("");
      setCurrentSources([]);
      setCurrentWarnings([]);
      setCurrentVerified(false);
      setError(null);
      setFeedbackRating(null);
      currentQueryRef.current = q;

      try {
        // SSE 流式问答
        setStreaming(true);
        const controller = new AbortController();
        abortRef.current = controller;

        // 构建多轮对话历史 (最近 8 轮，去重 user+assistant 配对)
        const history = messages.map((m) => ({
          role: m.role,
          content: m.content,
        }));
        const res = await askStream(
          q,
          10,
          selectedRelease || undefined,
          selectedSeries || undefined,
          selectedDocType || undefined,
          rerankerEnabled,
          history,
          controller.signal
        );
        if (!res.ok) throw new Error(`SSE ${res.status}`);
        if (!res.body) throw new Error("无响应流");

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";
        let fullAnswer = "";
        let sources: SourceItem[] = [];
        let warnings: string[] = [];
        let verified = false;

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split("\n");
          buffer = lines.pop() || "";

          for (const line of lines) {
            const parsed = parseSSELine(line);
            if (!parsed) continue;
            if (parsed.answer) {
              fullAnswer += parsed.answer;
              setCurrentAnswer(fullAnswer);
            }
            if (parsed.sources) {
              sources = parsed.sources;
              setCurrentSources(sources);
            }
            if (parsed.warnings) {
              warnings = parsed.warnings;
              setCurrentWarnings(warnings);
            }
            if (parsed.verified !== undefined) verified = parsed.verified;
            if (parsed.done) {
              // 语言兜底时后端会重推完整回答, 用最终 answer 替换已渲染内容
              if (parsed.answer && parsed.answer !== fullAnswer) {
                fullAnswer = parsed.answer;
                setCurrentAnswer(fullAnswer);
              }
              setStreaming(false);
              setLoading(false);
            }
            if (parsed.error) {
              setError(parsed.error);
              setStreaming(false);
              setLoading(false);
            }
          }
        }

        // 完成：添加助手消息
        if (fullAnswer) {
          const assistantMsg: Message = {
            role: "assistant",
            content: fullAnswer,
            sources: sources.length > 0 ? sources : undefined,
            warnings: warnings.length > 0 ? warnings : undefined,
            verified,
          };
          setMessages((prev) => [...prev, assistantMsg]);
          // 保存到对话历史 (流式完成后直接写入，避免依赖竞态)
          addEntry({
            query: currentQueryRef.current,
            answer: fullAnswer,
            sources: sources.map((s) => ({
              spec_number: s.spec_number,
              section_id: s.section_id || "",
              text: s.text,
              score: s.score,
            })),
          });
          setCurrentAnswer("");
          setCurrentSources([]);
          setCurrentWarnings([]);
          setCurrentVerified(false);
        }
      } catch (err: any) {
        // 用户主动停止 (AbortError) 不显示错误
        if (err?.name !== "AbortError") {
          setError(err instanceof Error ? err.message : "请求失败");
        }
        setLoading(false);
        setStreaming(false);
      }
    },
    [query, loading, rerankerEnabled, addEntry]
  );

  const handleStop = () => {
    abortRef.current?.abort();
    // 剪切到一半的内容也保留
    if (currentAnswer) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: currentAnswer,
          sources: currentSources.length > 0 ? currentSources : undefined,
          warnings: currentWarnings.length > 0 ? currentWarnings : undefined,
          verified: currentVerified,
        },
      ]);
    }
    setCurrentAnswer("");
    setLoading(false);
    setStreaming(false);
    // 停止后自动聚焦输入框，避免用户误以为界面卡死
    setTimeout(() => inputRef.current?.focus(), 0);
  };

  const handleFeedback = async (rating: "up" | "down") => {
    if (feedbackRating) return;
    setFeedbackRating(rating);
    // 从最后一条助手消息中拿 answer
    const lastAssistant = [...messages]
      .reverse()
      .find((m) => m.role === "assistant");
    try {
      await submitFeedback({
        query: currentQueryRef.current || query,
        answer: lastAssistant?.content || currentAnswer,
        sources: lastAssistant?.sources || currentSources,
        rating,
      });
    } catch (err) {
      console.error("反馈提交失败:", err);
      setFeedbackRating(null);
    }
  };

  const handleHistoryClick = (entryQuery: string) => {
    setQuery(entryQuery);
    setHistoryOpen(false);
    inputRef.current?.focus();
  };

  const handleNewChat = () => {
    abortRef.current?.abort();
    setMessages([]);
    setCurrentAnswer("");
    setCurrentSources([]);
    setCurrentWarnings([]);
    setCurrentVerified(false);
    setError(null);
    setFeedbackRating(null);
    setQuery("");
    currentQueryRef.current = "";
    inputRef.current?.focus();
  };

  const hasConversation = messages.length > 0 || loading || !!currentAnswer;

  return (
    <div className="flex-1 flex h-full relative">
      {/* ── 左侧工具竖条 ── */}
      <aside className="shrink-0 w-11 bg-gray-100 dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col items-center pt-3 gap-1 z-10">
        <button
          onClick={() => setHistoryOpen(true)}
          className="press-feedback p-2 rounded-lg text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          title="对话历史"
          aria-label="对话历史"
        >
          <svg
            className="w-5 h-5"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
        {/* 新对话 (进入对话后显示) */}
        {hasConversation && (
          <button
            onClick={handleNewChat}
            className="press-feedback p-2 rounded-lg text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
            title="新对话"
            aria-label="新对话"
          >
            <svg
              className="w-5 h-5"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              aria-hidden="true"
            >
              <path d="M12 4.5v15m7.5-7.5h-15" />
            </svg>
          </button>
        )}
      </aside>

      {/* ── 主区域 (对话内容 + 输入栏) ── */}
      <div className="flex-1 flex flex-col h-full min-w-0">
        {/* ── 对话历史侧边栏 (常驻 DOM，对称进出场) ── */}
        <>
          {/* 遮罩 */}
          <div
            className={`fixed inset-0 z-30 bg-black/20 dark:bg-black/40 transition-opacity duration-300
              ${historyOpen ? "opacity-100" : "opacity-0 pointer-events-none"}`}
            onClick={() => setHistoryOpen(false)}
            aria-hidden="true"
          />
          <aside
            aria-hidden={!historyOpen}
            className={`fixed left-0 top-0 bottom-0 z-[60] w-72 bg-gray-50 dark:bg-gray-950
            border-r border-gray-200 dark:border-gray-700 overflow-y-auto shadow-xl
            transition-transform duration-300 ease-[cubic-bezier(0.32,0.72,0,1)]
            ${historyOpen ? "translate-x-0" : "-translate-x-full"}`}
          >
            <div className="p-4 flex items-center justify-between border-b border-gray-200 dark:border-gray-700">
              <span className="text-sm font-semibold text-gray-600 dark:text-gray-300">
                对话历史
              </span>
              <button
                onClick={() => setHistoryOpen(false)}
                className="p-1.5 rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
                aria-label="关闭"
              >
                <svg
                  className="w-4 h-4"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  aria-hidden="true"
                >
                  <path d="M6 6l12 12M18 6L6 18" />
                </svg>
              </button>
            </div>
            {history.length === 0 ? (
              <p className="p-4 text-xs text-gray-400 dark:text-gray-500 text-center">
                暂无历史记录
              </p>
            ) : (
              <>
                <div className="p-3 space-y-2">
                  {history.map((entry) => (
                    <div
                      key={entry.id}
                      className={`group p-3 rounded-lg bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 cursor-pointer
                        hover:border-blue-300 dark:hover:border-blue-700 transition-colors`}
                      onClick={() => handleHistoryClick(entry.query)}
                    >
                      <p className="text-xs text-gray-700 dark:text-gray-300 line-clamp-2 font-medium">
                        {entry.query}
                      </p>
                      <div className="flex items-center justify-between mt-1.5">
                        <span className="text-[10px] text-gray-400 dark:text-gray-500">
                          {entry.sources.length} 源 ·{" "}
                          {new Date(entry.timestamp).toLocaleTimeString(
                            "zh-CN",
                            { hour: "2-digit", minute: "2-digit" }
                          )}
                        </span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            removeEntry(entry.id);
                          }}
                          className="opacity-0 group-hover:opacity-100 p-1 rounded-md text-gray-500 dark:text-gray-400 hover:text-red-500 dark:hover:text-red-400 hover:bg-gray-100 dark:hover:bg-gray-800 transition-all"
                          title="删除"
                          aria-label="删除"
                        >
                          <svg
                            className="w-3.5 h-3.5"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.8"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                            aria-hidden="true"
                          >
                            <path d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
                          </svg>
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
        </>

        {/* ── 主内容区 (可滚动) ── */}
        <div className="flex-1 overflow-y-auto">
          {!hasConversation ? (
            /* ── 初始状态: Hero ── */
            <div className="max-w-3xl mx-auto px-4 pt-24 pb-8 text-center">
              <h1 className="hero-item text-4xl font-semibold text-gray-900 dark:text-gray-100 tracking-tight mb-3">
                通信规范智能问答
              </h1>
              <p
                className="hero-item text-sm text-gray-400 dark:text-gray-500 mb-10"
                style={{ animationDelay: "80ms" }}
              >
                基于 RAG 技术，精准检索全部规范文档，获取带溯源的即时答案
              </p>

              {/* 示例查询 */}
              <div className="flex flex-wrap justify-center gap-2 max-w-2xl mx-auto">
                <span className="text-xs text-gray-400 dark:text-gray-500 self-center mr-1">
                  试试：
                </span>
                {EXAMPLE_QUERIES.map((q, i) => (
                  <button
                    key={q}
                    type="button"
                    onClick={(e) => {
                      setQuery(q);
                      handleSubmit(e, q);
                    }}
                    style={{ animationDelay: `${160 + i * 40}ms` }}
                    className={`hero-item press-feedback px-3 py-1.5 rounded-full text-xs bg-gray-100 dark:bg-gray-800
                    text-gray-600 dark:text-gray-300
                    hover:bg-blue-50 dark:hover:bg-blue-900/30 hover:text-blue-700 dark:hover:text-blue-300
                    border border-gray-200 dark:border-gray-700 hover:border-blue-200 dark:hover:border-blue-700
                    transition-colors`}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            /* ── 对话消息 ── */
            <div className="max-w-3xl mx-auto px-4 py-6 space-y-6">
              {/* 消息列表 */}
              {messages.map((msg, i) => (
                <div
                  key={i}
                  className={`msg-enter flex gap-3 ${msg.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  {/* 助手头像 */}
                  {msg.role === "assistant" && (
                    <div className="shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-[10px] font-bold mt-0.5">
                      AI
                    </div>
                  )}
                  <div
                    className={`max-w-[85%] ${msg.role === "user" ? "order-1" : ""}`}
                  >
                    {/* 用户消息 */}
                    {msg.role === "user" && (
                      <div className="px-4 py-3 rounded-2xl rounded-br-md bg-blue-600 dark:bg-blue-500 text-white text-sm leading-relaxed shadow-sm">
                        {msg.content}
                      </div>
                    )}

                    {/* 助手消息 */}
                    {msg.role === "assistant" && (
                      <div className="space-y-2">
                        <div className="px-5 py-4 rounded-2xl rounded-bl-md bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm">
                          <div className="text-gray-800 dark:text-gray-200 leading-relaxed text-sm">
                            <Markdown content={msg.content} />
                          </div>

                          {/* 反馈按钮 */}
                          {!streaming && (
                            <div className="mt-4 pt-3 border-t border-gray-100 dark:border-gray-800 flex items-center gap-2">
                              <span className="text-xs text-gray-400 dark:text-gray-500 mr-1">
                                评价:
                              </span>
                              <button
                                onClick={() => handleFeedback("up")}
                                disabled={feedbackRating !== null}
                                className={`press-feedback px-3 py-1.5 rounded-lg text-sm font-medium transition-[color,background-color,border-color,opacity] duration-150 ${
                                  feedbackRating === "up"
                                    ? "bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-400 border border-green-300 dark:border-green-700"
                                    : "bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700 hover:bg-green-50 dark:hover:bg-green-900/20 hover:text-green-600 dark:hover:text-green-400"
                                } ${feedbackRating !== null && feedbackRating !== "up" ? "opacity-30 cursor-not-allowed" : ""}`}
                                title="回答有帮助"
                              >
                                有用
                              </button>
                              <button
                                onClick={() => handleFeedback("down")}
                                disabled={feedbackRating !== null}
                                className={`press-feedback px-3 py-1.5 rounded-lg text-sm font-medium transition-[color,background-color,border-color,opacity] duration-150 ${
                                  feedbackRating === "down"
                                    ? "bg-red-100 dark:bg-red-900/30 text-red-700 dark:text-red-400 border border-red-300 dark:border-red-700"
                                    : "bg-gray-50 dark:bg-gray-800 text-gray-500 dark:text-gray-400 border border-gray-200 dark:border-gray-700 hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 dark:hover:text-red-400"
                                } ${feedbackRating !== null && feedbackRating !== "down" ? "opacity-30 cursor-not-allowed" : ""}`}
                                title="回答有误"
                              >
                                没用
                              </button>
                              {feedbackRating && (
                                <span className="text-xs text-gray-400 dark:text-gray-500 ml-1">
                                  已提交
                                </span>
                              )}
                            </div>
                          )}
                        </div>

                        {/* 警告 */}
                        {msg.warnings && msg.warnings.length > 0 && (
                          <div className="space-y-1">
                            {msg.warnings.map((w, j) => (
                              <div
                                key={j}
                                className="flex items-start gap-2 text-xs text-amber-700 dark:text-amber-400 bg-amber-50 dark:bg-amber-900/20 px-3 py-2 rounded-lg border border-amber-200 dark:border-amber-800"
                              >
                                <span>{w}</span>
                              </div>
                            ))}
                          </div>
                        )}

                        {/* 溯源 (始终可折叠，默认隐藏) */}
                        {msg.sources && msg.sources.length > 0 && (
                          <details className="group">
                            <summary className="text-xs text-gray-400 dark:text-gray-500 cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors select-none">
                              引用依据 ({msg.sources.length} 条)
                            </summary>
                            <div className="details-content mt-2 space-y-2">
                              {msg.sources.map((s, j) => (
                                <div
                                  key={j}
                                  className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 text-xs"
                                >
                                  <div className="font-semibold text-blue-700 dark:text-blue-300 mb-1">
                                    {s.spec_number} · {s.section_id || "—"} ·{" "}
                                    {(s.score * 100).toFixed(0)}%
                                  </div>
                                  <p className="text-gray-600 dark:text-gray-400 line-clamp-3">
                                    {s.text}
                                  </p>
                                </div>
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    )}
                  </div>

                  {/* 用户头像 */}
                  {msg.role === "user" && (
                    <div className="shrink-0 w-7 h-7 rounded-full bg-gray-300 dark:bg-gray-600 flex items-center justify-center text-white text-[10px] font-bold mt-0.5">
                      U
                    </div>
                  )}
                </div>
              ))}

              {/* 加载等待态（还没有内容回来时） */}
              {loading && !streaming && !currentAnswer && (
                <div className="msg-enter flex justify-start gap-3">
                  <div className="shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-[10px] font-bold">
                    AI
                  </div>
                  <div className="px-5 py-4 rounded-2xl rounded-bl-md bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm">
                    <div className="flex gap-1.5">
                      <span
                        className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce"
                        style={{ animationDelay: "0ms" }}
                      />
                      <span
                        className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce"
                        style={{ animationDelay: "150ms" }}
                      />
                      <span
                        className="w-2 h-2 rounded-full bg-gray-400 dark:bg-gray-500 animate-bounce"
                        style={{ animationDelay: "300ms" }}
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* 流式回答中 */}
              {streaming && currentAnswer && (
                <div className="msg-enter flex justify-start gap-3">
                  <div className="shrink-0 w-7 h-7 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center text-white text-[10px] font-bold">
                    AI
                  </div>
                  <div className="max-w-[85%] space-y-2">
                    <div className="px-5 py-4 rounded-2xl rounded-bl-md bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm">
                      <div className="text-gray-800 dark:text-gray-200 leading-relaxed text-sm">
                        <Markdown content={currentAnswer} />
                        <span className="inline-block w-2 h-4 ml-0.5 bg-blue-500 dark:bg-blue-400 animate-pulse rounded-sm align-middle" />
                      </div>
                    </div>

                    {/* 流式阶段引用折叠 */}
                    {currentSources.length > 0 && (
                      <details className="group">
                        <summary className="text-xs text-gray-400 dark:text-gray-500 cursor-pointer hover:text-blue-600 dark:hover:text-blue-400 transition-colors select-none">
                          引用依据 ({currentSources.length} 条)
                        </summary>
                        <div className="details-content mt-2 space-y-2">
                          {currentSources.map((s, j) => (
                            <div
                              key={j}
                              className="p-3 rounded-lg bg-blue-50 dark:bg-blue-900/20 border border-blue-100 dark:border-blue-800 text-xs"
                            >
                              <div className="font-semibold text-blue-700 dark:text-blue-300 mb-1">
                                {s.spec_number} · {s.section_id || "—"} ·{" "}
                                {(s.score * 100).toFixed(0)}%
                              </div>
                              <p className="text-gray-600 dark:text-gray-400 line-clamp-3">
                                {s.text}
                              </p>
                            </div>
                          ))}
                        </div>
                      </details>
                    )}
                  </div>
                </div>
              )}

              {/* 错误 */}
              {error && (
                <div className="msg-enter p-4 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-700 dark:text-red-400 text-sm border border-red-200 dark:border-red-800">
                  {error}
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>
          )}
        </div>

        {/* ── 底部输入栏 ── */}
        <div className="shrink-0 bg-white/80 dark:bg-gray-950/80 backdrop-blur border-t border-gray-200 dark:border-gray-800 px-4 py-3">
          <div className="max-w-3xl mx-auto">
            {/* ── Row 1: 过滤选择器 (左) + 精排开关 (右, 与输入框右对齐) ── */}
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              {/* 文档类型 */}
              {availableDocTypes.length > 0 && (
                <select
                  value={selectedDocType}
                  onChange={(e) => {
                    const v = e.target.value;
                    setSelectedDocType(v);
                    if (v === "oran") {
                      setSelectedRelease("");
                      setSelectedSeries("");
                    }
                  }}
                  className="px-2.5 py-1.5 rounded-lg text-xs bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-transparent cursor-pointer"
                >
                  <option value="">全部类型</option>
                  {availableDocTypes.map((dt) => (
                    <option key={dt} value={dt}>
                      {dt === "3gpp" ? "3GPP" : dt === "oran" ? "ORAN" : dt}
                    </option>
                  ))}
                </select>
              )}

              {/* Release — ORAN 选中时隐藏 */}
              {selectedDocType !== "oran" && availableReleases.length > 0 && (
                <select
                  value={selectedRelease}
                  onChange={(e) => setSelectedRelease(e.target.value)}
                  className="px-2.5 py-1.5 rounded-lg text-xs bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-transparent cursor-pointer"
                >
                  <option value="">全部 Release</option>
                  {availableReleases.sort().map((r) => (
                    <option key={r} value={r}>
                      {r}
                    </option>
                  ))}
                </select>
              )}

              {/* Series — ORAN 选中时隐藏 */}
              {selectedDocType !== "oran" && availableSeries.length > 0 && (
                <select
                  value={selectedSeries}
                  onChange={(e) => setSelectedSeries(e.target.value)}
                  className="px-2.5 py-1.5 rounded-lg text-xs bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 text-gray-600 dark:text-gray-300 focus:outline-none focus:ring-1 focus:ring-blue-500 focus:border-transparent cursor-pointer"
                >
                  <option value="">全部 Series</option>
                  {availableSeries
                    .sort((a, b) => parseInt(a) - parseInt(b))
                    .map((s) => (
                      <option key={s} value={s}>
                        Series {s}
                      </option>
                    ))}
                </select>
              )}

              {/* 清除过滤 */}
              {(selectedDocType || selectedRelease || selectedSeries) && (
                <button
                  onClick={() => {
                    setSelectedDocType("");
                    setSelectedRelease("");
                    setSelectedSeries("");
                  }}
                  className="press-feedback p-1.5 rounded-2xl text-gray-400 dark:text-gray-500 hover:bg-gray-200/60 dark:hover:bg-gray-700 hover:text-gray-600 dark:hover:text-gray-300 transition-colors"
                  title="清除全部过滤"
                  aria-label="清除全部过滤"
                >
                  <svg
                    className="w-4 h-4"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    aria-hidden="true"
                  >
                    <path d="M6 6l12 12M18 6L6 18" />
                  </svg>
                </button>
              )}

              {/* 精排开关 — ml-auto 推到右侧，与输入框右对齐 */}
              <button
                type="button"
                onClick={() => setRerankerEnabled(!rerankerEnabled)}
                disabled={loading}
                className={`ml-auto shrink-0 w-12 h-8 flex items-center justify-center rounded-xl transition-colors ${
                  rerankerEnabled
                    ? "text-gray-600 dark:text-gray-300 bg-gray-200/50 dark:bg-gray-700/50"
                    : "text-gray-400 dark:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                } ${loading ? "opacity-50 cursor-not-allowed" : ""}`}
                title={
                  rerankerEnabled
                    ? "精排已启用：结果更准但稍慢"
                    : "精排已关闭：速度优先"
                }
                aria-label={rerankerEnabled ? "精排已启用" : "精排已关闭"}
              >
                <svg
                  className="w-5 h-5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path d="M10.5 6h9.75M10.5 6a1.5 1.5 0 11-3 0m3 0a1.5 1.5 0 10-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 01-3 0m3 0a1.5 1.5 0 00-3 0m-9.75 0h9.75" />
                </svg>
              </button>
            </div>

            {/* ── Row 2: 输入框 + 发送 ── */}
            <form
              onSubmit={(e) => handleSubmit(e)}
              className="flex items-center gap-2"
            >
              <div className="flex-1 relative flex">
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  onKeyDown={(e) => {
                    if (
                      e.key === "Enter" &&
                      !e.shiftKey &&
                      !e.nativeEvent.isComposing
                    ) {
                      e.preventDefault();
                      handleSubmit(e);
                    }
                  }}
                  placeholder={
                    hasConversation
                      ? "继续提问..."
                      : "输入问题，如：NR PUSCH 的 DMRS 配置方式有哪些？"
                  }
                  rows={1}
                  className="w-full h-12 px-4 py-3 rounded-2xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-sm text-gray-900 dark:text-gray-100 placeholder:text-gray-400 dark:placeholder:text-gray-500 resize-none transition-shadow"
                  disabled={loading}
                  ref={inputRef}
                />
              </div>

              {/* 发送/停止按钮 — 停止态用红色方块图标，宽度与发送按钮一致 */}
              {loading ? (
                <button
                  type="button"
                  onClick={handleStop}
                  className="press-feedback shrink-0 h-12 w-12 flex items-center justify-center rounded-2xl border border-transparent bg-red-500 text-white hover:bg-red-600 transition-colors"
                  title="停止"
                  aria-label="停止生成"
                >
                  <svg
                    className="w-5 h-5"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    aria-hidden="true"
                  >
                    <rect x="5.5" y="5.5" width="13" height="13" rx="2" />
                  </svg>
                </button>
              ) : (
                <button
                  type="submit"
                  disabled={!query.trim()}
                  className="press-feedback shrink-0 h-12 w-12 flex items-center justify-center rounded-2xl border border-transparent bg-blue-600 dark:bg-blue-500 text-white hover:bg-blue-700 dark:hover:bg-blue-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                  title="发送"
                  aria-label="发送"
                >
                  <svg
                    className="w-5 h-5"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.8"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M6 12L3.269 3.126A59.768 59.768 0 0121.485 12 59.77 59.77 0 013.27 20.876L5.999 12zm0 0h7.5" />
                  </svg>
                </button>
              )}
            </form>
          </div>
        </div>
      </div>
    </div>
  );
}
