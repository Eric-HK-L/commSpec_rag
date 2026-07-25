"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";

const NAV = [
  { href: "/admin", label: "📊 仪表盘" },
  { href: "/admin/documents", label: "📄 文档管理" },
  { href: "/admin/ingestion", label: "⚙️ 摄入管理" },
  { href: "/admin/search", label: "🔍 搜索测试" },
  { href: "/admin/feedback", label: "💬 反馈分析" },
  { href: "/admin/logs", label: "📋 系统日志" },
  { href: "/admin/system", label: "🖥 系统健康" },
];

export default function AdminLayout({ children }: { children: React.ReactNode }) {
  const pathname = usePathname();
  const router = useRouter();

  // 登录页不使用管理布局
  if (pathname === "/admin/login") {
    return <>{children}</>;
  }

  const handleLogout = () => {
    document.cookie = "admin_auth=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT";
    router.push("/admin/login");
  };

  return (
    <div className="flex h-full">
      {/* 侧边栏 */}
      <aside className="w-52 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-700 flex-shrink-0 flex flex-col transition-colors">
        <div className="px-4 py-4 border-b border-gray-100 dark:border-gray-800">
          <span className="text-xs font-semibold text-gray-400 dark:text-gray-500 uppercase tracking-wider">管理控制台</span>
        </div>
        <nav className="p-2 space-y-0.5 flex-1">
          {NAV.map((item) => {
            const active = pathname === item.href || (item.href !== "/admin" && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                className={`block px-3 py-2 rounded-lg text-sm transition-colors ${
                  active
                    ? "bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 font-medium"
                    : "text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-800 hover:text-gray-900 dark:hover:text-gray-100"
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* 退出 */}
        <div className="p-2 border-t border-gray-100 dark:border-gray-800">
          <button
            onClick={handleLogout}
            className="w-full px-3 py-2 rounded-lg text-sm text-gray-500 dark:text-gray-400
                       hover:bg-red-50 dark:hover:bg-red-900/20 hover:text-red-600 dark:hover:text-red-400
                       transition-colors text-left"
          >
            🚪 退出登录
          </button>
        </div>
      </aside>

      {/* 主内容 */}
      <main className="flex-1 overflow-auto">{children}</main>
    </div>
  );
}
