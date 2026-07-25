#!/usr/bin/env bash
# ── CommSpec RAG 一键启动脚本（根目录） ──
# 用法: ./start.sh
# 从电脑初开机状态，一键启动前后端 + Docker 基础设施
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PID_DIR="$SCRIPT_DIR/logs"
BACKEND_PID="$PID_DIR/backend.pid"
FRONTEND_PID="$PID_DIR/frontend.pid"
BACKEND_LOG="$PID_DIR/backend.log"
FRONTEND_LOG="$PID_DIR/frontend.log"

mkdir -p "$PID_DIR"

echo "========================================"
echo "  CommSpec RAG 一键启动"
echo "========================================"
echo ""

# ── 1. 检查 Docker ──
echo "[1/5] 检查 Docker..."
if ! command -v docker &>/dev/null; then
  echo "❌ 未找到 Docker，请先安装 Docker Desktop"
  exit 1
fi
if ! docker info &>/dev/null; then
  echo "❌ Docker 未运行，请先启动 Docker Desktop"
  exit 1
fi
echo "  ✅ Docker 已就绪"

# ── 2. 创建数据卷 + 启动 Milvus 基础设施 ──
echo ""
echo "[2/5] 启动 Milvus 基础设施 (etcd + minio + milvus)..."

for vol in spec_rag_etcd_data spec_rag_minio_data spec_rag_milvus_data; do
  if ! docker volume inspect "$vol" &>/dev/null; then
    docker volume create "$vol" > /dev/null
  fi
done

docker compose -f deploy/docker-compose.yml up -d etcd minio milvus

echo "  ⏳ 等待 Milvus 就绪..."
for i in $(seq 1 45); do
  if docker compose -f deploy/docker-compose.yml exec -T milvus curl -sf http://localhost:9091/healthz &>/dev/null; then
    echo "  ✅ Milvus 已就绪"
    break
  fi
  sleep 2
done

# ── 3. 启动后端 ──
echo ""
echo "[3/5] 启动后端 API 服务..."

if [ -f "$BACKEND_PID" ]; then
  OLD_PID=$(cat "$BACKEND_PID")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "  ⚠️  后端已在运行 (PID: $OLD_PID)，跳过"
  else
    rm -f "$BACKEND_PID"
  fi
fi

if [ ! -f "$BACKEND_PID" ]; then
  if [ ! -f ".venv/bin/python" ]; then
    echo "❌ 未找到虚拟环境 .venv/，请先执行: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
    exit 1
  fi
  nohup .venv/bin/python -m src.main > "$BACKEND_LOG" 2>&1 &
  echo $! > "$BACKEND_PID"
  echo "  ✅ 后端已启动 (PID: $(cat $BACKEND_PID))"
fi

echo "  ⏳ 等待 API 就绪 (http://localhost:8000)..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:8000/api/v1/health &>/dev/null; then
    echo "  ✅ API 已就绪"
    break
  fi
  sleep 2
done

# ── 4. 启动前端 ──
echo ""
echo "[4/5] 启动前端 (Next.js 开发服务器)..."

if [ -f "$FRONTEND_PID" ]; then
  OLD_PID=$(cat "$FRONTEND_PID")
  if kill -0 "$OLD_PID" 2>/dev/null; then
    echo "  ⚠️  前端已在运行 (PID: $OLD_PID)，跳过"
  else
    rm -f "$FRONTEND_PID"
  fi
fi

if [ ! -f "$FRONTEND_PID" ]; then
  if [ ! -d "frontend/node_modules" ]; then
    echo "  📦 安装前端依赖..."
    cd frontend && npm install && cd "$SCRIPT_DIR"
  fi
  cd frontend
  nohup npm run dev > "$FRONTEND_LOG" 2>&1 &
  echo $! > "$FRONTEND_PID"
  cd "$SCRIPT_DIR"
  echo "  ✅ 前端已启动 (PID: $(cat $FRONTEND_PID))"
fi

echo "  ⏳ 等待前端就绪 (http://localhost:3000)..."
for i in $(seq 1 20); do
  if curl -sf http://localhost:3000 &>/dev/null; then
    echo "  ✅ 前端已就绪"
    break
  fi
  sleep 2
done

# ── 5. 完成 ──
echo ""
echo "[5/5] 启动完成！"
echo ""
echo "========================================"
echo "  全部服务已启动"
echo "========================================"
echo "  📡 API:       http://localhost:8000"
echo "  📋 Swagger:   http://localhost:8000/docs"
echo "  🖥️  前端:      http://localhost:3000"
echo "  ⚙️  管理后台:  http://localhost:3000/admin"
echo ""
echo "  日志文件:"
echo "    $BACKEND_LOG"
echo "    $FRONTEND_LOG"
echo ""
echo "  停止服务: ./stop.sh"
echo "========================================"
