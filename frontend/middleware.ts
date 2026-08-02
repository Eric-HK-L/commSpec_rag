import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";

const AUTH_COOKIE = "admin_auth";
const AUTH_VALUE = "commspec_admin_authenticated";

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;

  // 只保护 /admin 路径，放行登录页本身
  if (pathname.startsWith("/admin") && !pathname.startsWith("/admin/login")) {
    const cookie = request.cookies.get(AUTH_COOKIE);
    if (cookie?.value !== AUTH_VALUE) {
      const loginUrl = new URL("/admin/login", request.url);
      loginUrl.searchParams.set("from", pathname);
      return NextResponse.redirect(loginUrl);
    }
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/admin/:path*"],
};
