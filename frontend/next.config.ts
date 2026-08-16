import type { NextConfig } from "next";
import * as os from "os";

// 自动获取本机所有非内网IPv4地址，加入 allowedDevOrigins
// 解决远程访问时 Next.js 16 开发模式拒绝 HMR WebSocket 导致 React Hydration 失败、按钮失效的问题
function getAllLocalIPs(): string[] {
  const ips: string[] = [];
  const interfaces = os.networkInterfaces();
  for (const name of Object.keys(interfaces)) {
    for (const iface of interfaces[name] || []) {
      if (iface.family === "IPv4" && !iface.internal) {
        ips.push(iface.address);
      }
    }
  }
  return ips;
}

const autoIPs = getAllLocalIPs();
const extraOrigins = process.env.DEV_ORIGIN
  ? process.env.DEV_ORIGIN.split(",")
      .map((s) => s.trim())
      .filter(Boolean)
  : [];
const devOrigins = [...autoIPs, ...extraOrigins];

const BACKEND_URL = process.env.BACKEND_URL || "http://localhost:8000";

const nextConfig: NextConfig = {
  output: "standalone",
  devIndicators: false,
  allowedDevOrigins: devOrigins,
  // 显式指定应用根目录 — 仓库根目录有第二个 package-lock.json,
  // 不设置会导致 Turbopack 把 workspace root 误判为仓库根, 解析 next 包失败
  turbopack: {
    root: process.cwd(),
  },
  // SSE 流式问答最长可 60+ 秒，防止 Next.js 代理超时断开 (socket hang up)
  httpAgentOptions: { keepAlive: true },
  experimental: {
    proxyTimeout: 180_000,
  },
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${BACKEND_URL}/api/v1/:path*`,
      },
      {
        source: "/images/:path*",
        destination: `${BACKEND_URL}/images/:path*`,
      },
    ];
  },
};

export default nextConfig;
