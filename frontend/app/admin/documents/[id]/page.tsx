"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import Link from "next/link";
import { getDocumentChunks, getDocument, type ChunkItem, type DocumentDetail } from "@/lib/api";

export default function DocumentDetailPage() {
  const { id } = useParams<{ id: string }>();
  const docId = decodeURIComponent(id);
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [doc, setDoc] = useState<DocumentDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([
      getDocument(docId),
      getDocumentChunks(docId, 0, 200),
    ])
      .then(([d, c]) => { setDoc(d); setChunks(c); })
      .catch((e) => setError(e instanceof Error ? e.message : "加载失败"))
      .finally(() => setLoading(false));
  }, [docId]);

  if (loading) return <div className="p-8 text-gray-500 dark:text-gray-400">加载中...</div>;
  if (error) return <div className="p-8 text-red-600 dark:text-red-400">加载失败: {error}</div>;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      {/* 返回按钮 */}
      <Link
        href="/admin/documents"
        className="inline-flex items-center gap-1.5 text-sm text-gray-500 dark:text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 transition-colors"
      >
        <span className="text-lg leading-none">←</span> 返回文档列表
      </Link>

      <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">📄 文档详情</h1>

      {/* 元数据 */}
      {doc && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 p-5 rounded-xl bg-white dark:bg-gray-900 border border-gray-200 dark:border-gray-700 shadow-sm transition-colors">
          <Meta label="规范编号" value={doc.spec_number || "—"} mono />
          <Meta label="Release" value={doc.release} />
          <Meta label="Series" value={String(doc.series || "—")} />
          <Meta label="Chunks" value={String(doc.chunk_count)} />
          <Meta label="版本" value={doc.version || "—"} />
          <Meta label="来源" value={doc.source || "—"} />
          <Meta label="文档 ID" value={doc.doc_id} mono />
          <Meta label="标题" value={doc.title || "—"} />
        </div>
      )}

      {/* Chunks 列表 */}
      <div className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-900 overflow-hidden shadow-sm transition-colors">
        <div className="px-4 py-3 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 text-sm font-medium text-gray-500 dark:text-gray-400">
          Chunks ({chunks.length})
        </div>
        <div className="divide-y divide-gray-100 dark:divide-gray-700">
          {chunks.map((c) => (
            <div key={c.chunk_index} className="p-4 hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors">
              <div className="flex items-center gap-3 mb-2">
                <span className="text-xs font-mono bg-gray-100 dark:bg-gray-800 px-2 py-0.5 rounded text-gray-600 dark:text-gray-400">
                  #{c.chunk_index}
                </span>
                {c.parent_section_id && (
                  <span className="text-xs font-mono text-blue-600 dark:text-blue-400">§{c.parent_section_id}</span>
                )}
                {c.parent_title && (
                  <span className="text-xs text-gray-400 dark:text-gray-500 truncate">{c.parent_title}</span>
                )}
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 leading-relaxed whitespace-pre-wrap line-clamp-4">
                {c.text}
              </p>
            </div>
          ))}
          {chunks.length === 0 && (
            <div className="p-8 text-center text-gray-400 dark:text-gray-500">暂无 Chunk</div>
          )}
        </div>
      </div>
    </div>
  );
}

function Meta({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="p-3 rounded-lg bg-gray-50 dark:bg-gray-800 border border-gray-100 dark:border-gray-700">
      <div className="text-xs text-gray-400 dark:text-gray-500">{label}</div>
      <div className={`font-medium text-gray-700 dark:text-gray-300 text-sm mt-0.5 truncate ${mono ? "font-mono" : ""}`}>{value}</div>
    </div>
  );
}
