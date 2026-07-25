#!/usr/bin/env bash
# ── CommSpec RAG 一键停止脚本（根目录） ──
# 用法: ./stop.sh
# 停止前后端进程 + Docker 容器，恢复到初开机状态
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_DIR="$SCRIPT_DIR/logs"
BACKEND_PID="$PID_DIR/backend.pid"
FRONTEND_PID="$PID_DIR/frontend.pid"

echo "========================================"
echo "  CommSpec RAG 一键停止"
echo "========================================"
echo ""

# ── 1. 停止后端 ──
echo "[1/3] 停止后端 API 服务..."
if [ -f "$BACKEND_PID" ]; then
  BACKEND=$(cat "$BACKEND_PID")
  if kill -0 "$BACKEND" 2>/dev/null; then
    kill "$BACKEND" 2>/dev/null || true
    # 等待进程退出
    for i in $(seq 1 10); do
      if ! kill -0 "$BACKEND" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    # 若仍未退出，强制终止
    if kill -0 "$BACKEND" 2>/dev/null; then
      kill -9 "$BACKEND" 2>/dev/null || true
    fi
    echo "  ✅ 后端已停止 (PID: $BACKEND)"
  else
    echo "  ⚠️  后端 PID 文件存在但进程已不存在，按端口清理..."
    lsof -ti :8000 2>/dev/null | xargs kill 2>/dev/null || true
    echo "  ✅ 端口 8000 已清理"
  fi
  rm -f "$BACKEND_PID"
else
  echo "  ℹ️  未找到后端 PID 文件，尝试按端口清理..."
  # 兜底：按端口 8000 找进程并终止
  lsof -ti :8000 2>/dev/null | xargs kill 2>/dev/null || true
  echo "  ✅ 端口 8000 已清理"
fi

# ── 2. 停止前端 ──
echo ""
echo "[2/3] 停止前端 (Next.js)..."
if [ -f "$FRONTEND_PID" ]; then
  FRONTEND=$(cat "$FRONTEND_PID")
  if kill -0 "$FRONTEND" 2>/dev/null; then
    kill "$FRONTEND" 2>/dev/null || true
    for i in $(seq 1 10); do
      if ! kill -0 "$FRONTEND" 2>/dev/null; then
        break
      fi
      sleep 1
    done
    if kill -0 "$FRONTEND" 2>/dev/null; then
      kill -9 "$FRONTEND" 2>/dev/null || true
    fi
    echo "  ✅ 前端已停止 (PID: $FRONTEND)"
  else
    echo "  ⚠️  前端 PID 文件存在但进程已不存在，按端口清理..."
    lsof -ti :3000 2>/dev/null | xargs kill 2>/dev/null || true
    echo "  ✅ 端口 3000 已清理"
  fi
  rm -f "$FRONTEND_PID"
else
  echo "  ℹ️  未找到前端 PID 文件，尝试按端口清理..."
  lsof -ti :3000 2>/dev/null | xargs kill 2>/dev/null || true
  echo "  ✅ 端口 3000 已清理"
fi

# ── 3. 停止 Docker 容器 ──
echo ""
echo "[3/3] 停止 Docker 容器..."
if command -v docker &>/dev/null && docker info &>/dev/null 2>&1; then
  docker compose -f deploy/docker-compose.yml down --remove-orphans 2>/dev/null || true
  echo "  ✅ Docker 容器已停止"
else
  echo "  ⚠️  Docker 未运行或不可用，跳过"
fi

echo ""
echo "========================================"
echo "  全部服务已停止"
echo "========================================"
echo ""
echo "  💡 数据卷保留，下次启动可复用。如需清除数据:"
echo "     docker volume rm spec_rag_etcd_data spec_rag_minio_data spec_rag_milvus_data"
echo ""
