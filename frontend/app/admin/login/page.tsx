import { Suspense } from "react";
import { LoginForm } from "./login-form";

export default function AdminLoginPage() {
  return (
    <Suspense fallback={<div className="min-h-screen flex items-center justify-center text-gray-400">加载中...</div>}>
      <LoginForm />
    </Suspense>
  );
}
