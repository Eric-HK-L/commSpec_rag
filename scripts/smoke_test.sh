#!/usr/bin/env bash
# =============================================================================
# CommSpec RAG 全栈冒烟测试
# 覆盖: 后端健康检查 / 检索(含过滤) / 问答 / 管理后台认证 / 前端页面
#
# 用法:
#   ./scripts/smoke_test.sh [BASE_URL=http://localhost:8000] [FRONTEND_URL=http://localhost:3000]
#
# 退出码: 0=全部通过  1=存在失败项
# =============================================================================

set -uo pipefail

BASE_URL="${1:-http://localhost:8000}"
FRONTEND_URL="${2:-http://localhost:3000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$ROOT_DIR/.env"
CURL_BIN="$(command -v curl || printf 'curl')"

# 从 .env 读取 ADMIN_PASSWORD (若已配置)
ADMIN_PASSWORD=""
if [ -f "$ENV_FILE" ] && grep -Eq '^ADMIN_PASSWORD=.+' "$ENV_FILE"; then
  ADMIN_PASSWORD="$(grep -E '^ADMIN_PASSWORD=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"'"'"' ')"
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
COOKIE_JAR="$TMP/cookies.txt"

PASS=0; FAIL=0; SKIP=0
FAILED_ITEMS=()

log()  { printf '\033[36m== %s ==\033[0m\n' "$*"; }
ok()   { PASS=$((PASS+1)); printf '  \033[32m[PASS]\033[0m %s\n' "$1"; }
bad()  { FAIL=$((FAIL+1)); printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; FAILED_ITEMS+=("$1"); }
skip() { SKIP=$((SKIP+1)); printf '  \033[33m[SKIP]\033[0m %s\n' "$1"; }

# 从 JSON 文件按点路径提取字段 (失败时退出码非 0)
json_field() {
  python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))
for k in sys.argv[2].split("."):
    if isinstance(data, list) and k.isdigit():
        data = data[int(k)]
    else:
        data = data[k]
if isinstance(data, bool):
    print("true" if data else "false")
elif data is None:
    print("")
else:
    print(data)
' "$1" "$2" 2>/dev/null
}

# 发起请求并保存响应体到文件, 同时输出 HTTP 状态码
req() {
  local out="$1"; shift
  local timeout="${CURL_TIMEOUT:-30}"
  "$CURL_BIN" -sS --max-time "$timeout" -o "$out" -w '%{http_code}' "$@"
}

echo "目标后端: $BASE_URL"
echo "目标前端: $FRONTEND_URL"
echo "管理后台: $([ -n "$ADMIN_PASSWORD" ] && echo '已配置 ADMIN_PASSWORD' || echo '未配置 ADMIN_PASSWORD (登录项按 403 校验)')"
echo

# ---------------------------------------------------------------------------
log "1. 后端健康检查 GET /api/v1/health"
# ---------------------------------------------------------------------------
CODE="$(req "$TMP/health.json" "$BASE_URL/api/v1/health")"
STATUS="$(json_field "$TMP/health.json" status || echo '')"
CHUNKS="$(json_field "$TMP/health.json" chunk_count || echo 0)"
if [ "$CODE" = "200" ] && [ "$STATUS" = "ready" ] && [ "$CHUNKS" -gt 0 ]; then
  ok "健康检查通过 (chunk_count=$CHUNKS)"
else
  bad "健康检查异常: HTTP=$CODE status='$STATUS' chunk_count=$CHUNKS"
fi

# ---------------------------------------------------------------------------
log "2. 检索接口 POST /api/v1/search (无过滤)"
# ---------------------------------------------------------------------------
CODE="$(CURL_TIMEOUT=60 req "$TMP/search.json" -X POST "$BASE_URL/api/v1/search" \
  -H 'Content-Type: application/json' \
  -d '{"query":"PDU Session Resource Setup 流程","top_k":10}')"
SUCCESS="$(json_field "$TMP/search.json" success || echo '')"
TOTAL="$(json_field "$TMP/search.json" data.total || echo 0)"
if [ "$CODE" = "200" ] && [ "$SUCCESS" = "true" ] && [ "$TOTAL" -gt 0 ]; then
  ok "无过滤检索返回 $TOTAL 条结果"
else
  bad "无过滤检索异常: HTTP=$CODE success='$SUCCESS' total=$TOTAL"
fi

# ---------------------------------------------------------------------------
log "3. 检索过滤验证 POST /api/v1/search (release+series, 校验 BM25 过滤修复)"
# ---------------------------------------------------------------------------
RELEASE="$(json_field "$TMP/search.json" data.results.0.release || echo '')"
SERIES="$(json_field "$TMP/search.json" data.results.0.series || echo '')"
# 自适应: 优先用实际数据元数据, 否则回退默认值
[ -z "$RELEASE" ] && RELEASE="R18"
[ -z "$SERIES" ] && SERIES="38"
CODE="$(CURL_TIMEOUT=60 req "$TMP/search_filtered.json" -X POST "$BASE_URL/api/v1/search" \
  -H 'Content-Type: application/json' \
  -d "{\"query\":\"PDU Session Resource Setup 流程\",\"top_k\":20,\"filters\":{\"release\":\"$RELEASE\",\"series\":\"$SERIES\"}}")"
FTOTAL="$(json_field "$TMP/search_filtered.json" data.total || echo 0)"
if [ "$CODE" != "200" ]; then
  bad "过滤检索 HTTP=$CODE"
elif [ "$FTOTAL" -eq 0 ]; then
  bad "过滤检索无结果 (release=$RELEASE series=$SERIES) — 请检查数据元数据"
else
  python3 -c '
import json, sys
data = json.load(open(sys.argv[1]))["data"]["results"]
rel, ser = sys.argv[2], int(sys.argv[3])
print(sum(1 for r in data if str(r.get("release", "")) != rel or int(r.get("series") or 0) != ser))
' "$TMP/search_filtered.json" "$RELEASE" "$SERIES" > "$TMP/violations"
  VIOLATIONS="$(cat "$TMP/violations")"
  if [ "$VIOLATIONS" -eq 0 ]; then
    ok "过滤检索 $FTOTAL 条全部满足 release=$RELEASE series=$SERIES"
  else
    bad "过滤检索存在 $VIOLATIONS 条不匹配 (release=$RELEASE series=$SERIES)"
  fi
fi

# ---------------------------------------------------------------------------
log "4. 问答接口 POST /api/v1/ask (真实 LLM, 最长 180s)"
# ---------------------------------------------------------------------------
CODE="$(CURL_TIMEOUT=180 req "$TMP/ask.json" -X POST "$BASE_URL/api/v1/ask" \
  -H 'Content-Type: application/json' \
  -d '{"query":"38.413 中 PDU Session Resource Setup 流程是什么？","top_k":10}')"
ANSWER="$(json_field "$TMP/ask.json" answer || echo '')"
VERIFIED="$(json_field "$TMP/ask.json" verified || echo '')"
if [ "$CODE" = "200" ] && [ -n "$ANSWER" ]; then
  SRC_CNT="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1])).get("sources", [])))' "$TMP/ask.json" 2>/dev/null || echo 0)"
  ok "问答返回有效答案 (${#ANSWER} 字, verified=$VERIFIED, sources=$SRC_CNT)"
else
  bad "问答异常: HTTP=$CODE answer 为空 (${#ANSWER} 字)"
fi

# ---------------------------------------------------------------------------
log "5. 管理后台认证"
# ---------------------------------------------------------------------------
if [ -z "$ADMIN_PASSWORD" ]; then
  CODE="$(req "$TMP/login.json" -X POST "$BASE_URL/api/v1/admin/login" \
    -H 'Content-Type: application/json' \
    -d '{"username":"admin","password":"wrong"}')"
  if [ "$CODE" = "403" ]; then
    ok "未配置 ADMIN_PASSWORD 时登录按预期返回 403"
  else
    bad "未配置 ADMIN_PASSWORD 时登录应返回 403, 实际 $CODE"
  fi
else
  CODE="$(req "$TMP/config_anon.json" "$BASE_URL/api/v1/admin/config")"
  if [ "$CODE" = "401" ]; then
    ok "未登录访问 /admin/config 返回 401"
  else
    bad "未登录访问 /admin/config 应返回 401, 实际 $CODE"
  fi
  CODE="$(req "$TMP/login.json" -c "$COOKIE_JAR" -X POST "$BASE_URL/api/v1/admin/login" \
    -H 'Content-Type: application/json' \
    -d "{\"username\":\"admin\",\"password\":\"$ADMIN_PASSWORD\"}")"
  LUSER="$(json_field "$TMP/login.json" data.username || echo '')"
  if [ "$CODE" = "200" ] && [ "$LUSER" = "admin" ]; then
    ok "管理员登录成功"
  else
    bad "管理员登录失败: HTTP=$CODE (用户名默认 admin, 可用 ADMIN_USERNAME 覆盖)"
  fi
  CODE="$(req "$TMP/config_auth.json" -b "$COOKIE_JAR" "$BASE_URL/api/v1/admin/config")"
  LCONF="$(json_field "$TMP/config_auth.json" data.llm_model || echo '')"
  if [ "$CODE" = "200" ] && [ -n "$LCONF" ]; then
    ok "登录后访问 /admin/config 成功 (llm_model=$LCONF)"
  else
    bad "登录后访问 /admin/config 异常: HTTP=$CODE"
  fi
fi

# ---------------------------------------------------------------------------
log "6. 前端页面"
# ---------------------------------------------------------------------------
CODE="$(req "$TMP/front.html" "$FRONTEND_URL/")"
if [ "$CODE" = "200" ]; then
  ok "前端首页 GET / 200"
else
  bad "前端首页 GET / 返回 $CODE"
fi

CODE="$(req "$TMP/admin.html" "$FRONTEND_URL/admin")"
if [ "$CODE" = "200" ] || [ "$CODE" = "307" ]; then
  ok "管理后台页 GET /admin $CODE"
else
  bad "管理后台页 GET /admin 返回 $CODE (期望 200 或 307)"
fi

# ---------------------------------------------------------------------------
echo
echo "========== 冒烟测试结果 =========="
echo "  通过: $PASS  失败: $FAIL  跳过: $SKIP"
if [ "$FAIL" -gt 0 ]; then
  printf '\033[31m  失败项:\033[0m\n'
  for item in "${FAILED_ITEMS[@]}"; do
    printf '    - %s\n' "$item"
  done
  exit 1
fi
echo "  全部通过"
exit 0
