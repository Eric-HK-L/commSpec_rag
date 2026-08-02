"use client";

import { useState, useEffect, useCallback } from "react";

export interface ConversationEntry {
  id: string;
  query: string;
  answer: string;
  sources: { spec_number: string; section_id: string; text: string; score: number }[];
  timestamp: number;
}

const STORAGE_KEY = "commspec_rag_history";
const LEGACY_STORAGE_KEY = "3gpp_rag_history"; // 旧命名，一次性迁移
const MAX_ENTRIES = 50;

// 将旧命名 key 下的历史数据迁移到新 key（仅执行一次）
function migrateLegacyHistory() {
  if (typeof window === "undefined") return;
  try {
    if (localStorage.getItem(STORAGE_KEY)) return;
    const legacy = localStorage.getItem(LEGACY_STORAGE_KEY);
    if (legacy) {
      localStorage.setItem(STORAGE_KEY, legacy);
      localStorage.removeItem(LEGACY_STORAGE_KEY);
    }
  } catch {
    /* ignore */
  }
}

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

  // 初始化加载（含旧命名数据迁移）
  useEffect(() => {
    migrateLegacyHistory();
    setHistory(loadHistory());
  }, []);

  // 添加新对话
  const addEntry = useCallback((entry: Omit<ConversationEntry, "id" | "timestamp">) => {
    const newEntry: ConversationEntry = {
      ...entry,
      id: crypto.randomUUID(),
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
