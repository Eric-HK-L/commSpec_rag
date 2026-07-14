"use client";

import { useState, useEffect, useCallback } from "react";

export interface ConversationEntry {
  id: string;
  query: string;
  answer: string;
  sources: { spec_number: string; section_id: string; text: string; score: number }[];
  timestamp: number;
}

const STORAGE_KEY = "3gpp_rag_history";
const MAX_ENTRIES = 50;

function loadHistory(): ConversationEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(entries: ConversationEntry[]) {
  if (typeof window === "undefined") return;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
  } catch {
    /* quota exceeded, silently ignore */
  }
}

export function useConversationHistory() {
  const [history, setHistory] = useState<ConversationEntry[]>([]);

  // 初始化加载
  useEffect(() => {
    setHistory(loadHistory());
  }, []);

  // 添加新对话
  const addEntry = useCallback((entry: Omit<ConversationEntry, "id" | "timestamp">) => {
    const newEntry: ConversationEntry = {
      ...entry,
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      timestamp: Date.now(),
    };
    setHistory((prev) => {
      const updated = [newEntry, ...prev];
      saveHistory(updated);
      return updated;
    });
  }, []);

  // 删除单条
  const removeEntry = useCallback((id: string) => {
    setHistory((prev) => {
      const updated = prev.filter((e) => e.id !== id);
      saveHistory(updated);
      return updated;
    });
  }, []);

  // 清空全部
  const clearHistory = useCallback(() => {
    setHistory([]);
    localStorage.removeItem(STORAGE_KEY);
  }, []);

  return { history, addEntry, removeEntry, clearHistory };
}
