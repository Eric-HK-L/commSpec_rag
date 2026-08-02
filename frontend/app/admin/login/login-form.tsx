"use client";

import { useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";

const VALID_USERNAME = "admin";
const VALID_PASSWORD = "linux123";
const AUTH_COOKIE = "admin_auth";
const AUTH_VALUE = "commspec_admin_authenticated";

export function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const from = searchParams.get("from") || "/admin";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    if (username === VALID_USERNAME && password === VALID_PASSWORD) {
      const expires = new Date();
      expires.setDate(expires.getDate() + 1);
      document.cookie = `${AUTH_COOKIE}=${AUTH_VALUE}; path=/; expires=${expires.toUTCString()}; SameSite=Lax`;
      router.push(from);
    } else {
      setError("账号或密码错误");
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-950 px-4">
      <div className="w-full max-w-sm">
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 mb-3">
            <span className="w-9 h-9 rounded-lg bg-blue-600 dark:bg-blue-500 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                <path d="M3 8V4m3 4V4m3 4V4M2 10h8" />
              </svg>
            </span>
            <span className="font-extrabold text-xl text-gray-800 dark:text-gray-200">CommSpec</span>
            <span className="text-xs font-bold px-1.5 py-0.5 rounded bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400 uppercase tracking-wider">RAG</span>
          </div>
          <h1 className="text-lg font-semibold text-gray-700 dark:text-gray-300">管理员登录</h1>
        </div>

        <form onSubmit={handleSubmit} className="bg-white dark:bg-gray-900 rounded-2xl shadow-lg border border-gray-200 dark:border-gray-700 p-6 space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1.5">账号</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="admin"
              autoComplete="username"
              className="w-full px-4 py-2.5 rounded-xl border border-gray-300 dark:border-gray-600
                         bg-white dark:bg-gray-800
                         focus:outline-none focus:ring-2 focus:ring-blue-500
                         text-base text-gray-900 dark:text-gray-100
                         placeholder:text-gray-400 dark:placeholder:text-gray-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-600 dark:text-gray-400 mb-1.5">密码</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="••••••••"
              autoComplete="current-password"
              className="w-full px-4 py-2.5 rounded-xl border border-gray-300 dark:border-gray-600
                         bg-white dark:bg-gray-800
                         focus:outline-none focus:ring-2 focus:ring-blue-500
                         text-base text-gray-900 dark:text-gray-100
                         placeholder:text-gray-400 dark:placeholder:text-gray-500"
            />
          </div>

          {error && (
            <div className="p-3 rounded-lg bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400 text-sm text-center">
              {error}
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !username || !password}
            className="w-full py-2.5 rounded-xl bg-blue-600 dark:bg-blue-500 text-white font-semibold
                       hover:bg-blue-700 dark:hover:bg-blue-600
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            {loading ? "验证中..." : "登 录"}
          </button>
        </form>
      </div>
    </div>
  );
}
