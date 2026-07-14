import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";
import { ThemeProvider } from "@/lib/theme";
import { ThemeToggle } from "@/components/ThemeToggle";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "3GPP RAG — 通信规范智能问答",
  description: "面向 3GPP 通信标准文档的检索增强生成系统",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`} suppressHydrationWarning>
      <body className="min-h-full flex flex-col">
        <ThemeProvider>
          <nav className="sticky top-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur border-b border-gray-200 dark:border-gray-800 px-6 py-3 flex items-center justify-between transition-colors">
            <Link href="/" className="flex items-center gap-2.5 group select-none">
              <span className="flex-shrink-0 w-8 h-8 rounded-lg bg-blue-600 dark:bg-blue-500
                               flex items-center justify-center
                               group-hover:shadow-md group-hover:shadow-blue-600/25
                               dark:group-hover:shadow-blue-500/25 transition-all">
                <svg className="w-5 h-5 text-white" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
                  <path d="M3 8V4m3 4V4m3 4V4M2 10h8" />
                </svg>
              </span>
              <span className="font-extrabold text-lg text-gray-800 dark:text-gray-200 tracking-tight">
                3GPP
              </span>
              <span className="text-xs font-bold px-1.5 py-0.5 rounded
                               bg-blue-100 dark:bg-blue-900/50
                               text-blue-600 dark:text-blue-400
                               uppercase tracking-wider">
                RAG
              </span>
            </Link>
            <ThemeToggle />
          </nav>
          <main className="flex-1">{children}</main>
        </ThemeProvider>
      </body>
    </html>
  );
}
