---
title: "附录E: 安全架构"
tags: [security, architecture, hardening]
---

# 附录 E — 安全架构

## E.1 安全概览

本系统采用纵深防御策略，从网络层、应用层、数据层三个维度保障安全。

```
┌─────────────────────────────────────────────────┐
│                 网络层安全                        │
│  · 防火墙端口最小化                               │
│  · Nginx 反向代理 + TLS                          │
│  · 内网隔离 (Milvus / Admin)                     │
├─────────────────────────────────────────────────┤
│                 应用层安全                        │
│  · API Key 认证                                  │
│  · CORS 白名单                                   │
│  · 输入校验 (Pydantic)                           │
│  · 速率限制 (预留)                                │
├─────────────────────────────────────────────────┤
│                 数据层安全                        │
│  · 环境变量隔离 (.env)                           │
│  · LLM API Key 不落盘日志                        │
│  · 用户反馈数据本地存储                           │
│  · Milvus 无内置认证（依赖网络隔离）              │
└─────────────────────────────────────────────────┘
```

## E.2 认证与授权

### API Key 认证

系统支持可选的 API Key 认证，通过 `src/api/auth.py` 的中间件实现：

```bash
# 启用认证（在 .env 中设置）
API_KEY_ENABLED=true
API_KEY=your-secret-api-key-here

# 客户端请求需带 Header
curl -H "X-API-Key: your-secret-api-key-here" http://localhost:8000/api/v1/search?q=test
```

**当前默认状态**: 认证已实现但默认宽松（未配置 API Key 时放行所有请求）。生产环境必须启用。

### CORS 配置

```python
# src/main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,  # 生产环境限制为具体域名
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)
```

```bash
# .env 生产配置
CORS_ORIGINS=["https://rag.your-company.com"]
```

## E.3 密钥管理

### 密钥存储

| 密钥类型 | 存储位置 | 保护措施 |
|----------|----------|----------|
| LLM_API_KEY | `.env` | 不提交 Git (.gitignore) |
| GOOGLE_API_KEY | `.env` | 不提交 Git |
| API_KEY (系统认证) | `.env` | 不提交 Git |
| HuggingFace Token | 环境变量 `HF_TOKEN` | 可选，仅私有模型需要 |

### 密钥泄露防范

```bash
# 1. 确保 .env 在 .gitignore 中
grep "\.env" .gitignore
# 应输出: .env 或 .env.local

# 2. 扫描 Git 历史中是否有密钥泄露
git log --all --full-history -- '*.env'

# 3. 如果密钥已泄露，立即轮换
# - OpenAI: https://platform.openai.com/api-keys
# - Google: https://console.cloud.google.com/apis/credentials

# 4. 确保日志不记录敏感信息
grep -i "api.key\|api_key\|secret" logs/app.log
```

## E.4 网络安全

### 端口暴露策略

| 端口 | 服务 | 默认绑定 | 生产建议 |
|------|------|----------|----------|
| 3000 | Frontend | `0.0.0.0:3000` | Nginx 反代 → `127.0.0.1:3000` |
| 8000 | API | `0.0.0.0:8000` | Nginx 反代 → `127.0.0.1:8000` |
| 19530 | Milvus gRPC | `0.0.0.0:19530` | `127.0.0.1:19530` |
| 9091 | Milvus Health | `0.0.0.0:9091` | `127.0.0.1:9091` |

### Docker 网络安全

```yaml
# docker-compose.yml 生产加固版
services:
  api:
    ports:
      - "127.0.0.1:8000:8000"  # 仅本地
    networks:
      - internal

  milvus:
    ports:
      - "127.0.0.1:19530:19530"  # 仅本地
    networks:
      - internal

networks:
  internal:
    driver: bridge
    internal: true  # 禁止出站网络（如无需外网 LLM）
```

## E.5 数据安全

### 敏感数据处理

| 数据类型 | 存储位置 | 敏感性 | 保护措施 |
|----------|----------|--------|----------|
| 用户查询 | 日志文件 | 中 | 日志轮转 + 访问控制 |
| LLM 响应 | 日志文件 | 中 | 日志轮转 + 访问控制 |
| 用户反馈 | `data/cache/feedback.jsonl` | 中 | 文件权限 600 |
| 3GPP 文档 | `data/documents/` | 低 | 公开规范 |
| 嵌入向量 | Milvus | 低 | 需 Milvus 连接才能读取 |
| LLM API Key | `.env` | 高 | 文件权限 600 + .gitignore |

### 日志安全

```python
# 生产环境建议对日志中的敏感字段脱敏
# 当前实现: 日志记录请求参数，建议审查是否包含敏感信息
LOG_LEVEL=INFO  # DEBUG 可能输出 API Key 等敏感信息
```

### Milvus 安全

Milvus 社区版不支持内置认证，依赖以下措施：

1. **网络隔离**: Milvus 端口仅绑定 `127.0.0.1`
2. **Docker 内部网络**: 仅 API 容器可访问 Milvus
3. **防火墙**: 阻止外部对 19530 端口的访问

## E.6 依赖安全

### 定期检查

```bash
# Python 依赖安全检查
pip install safety
safety check -r requirements.txt

# Node.js 依赖安全检查
cd frontend && npm audit

# Docker 镜像扫描
docker scan milvusdb/milvus:v2.4.10
```

### 依赖更新策略

- **安全补丁**: 立即更新
- **小版本**: 每月评估
- **大版本**: 充分测试后更新

## E.7 安全基线检查清单

部署前应完成的检查：

- [ ] `.env` 文件权限为 600 (`chmod 600 .env`)
- [ ] `.env` 已添加到 `.gitignore`
- [ ] Milvus 端口不对外暴露（`127.0.0.1:19530`）
- [ ] CORS 限制为生产域名（非 `*`）
- [ ] 生产环境 `LOG_LEVEL=INFO`（非 DEBUG）
- [ ] Nginx 已配置 HTTPS (TLS 1.2+)
- [ ] Admin 路径已限制内网访问
- [ ] 日志轮转已配置
- [ ] `LLM_API_KEY` 非默认值 `sk-your-key-here`
- [ ] 定期备份已配置 (cron)
- [ ] API Key 认证已启用（生产环境）
- [ ] 防火墙规则已配置（仅开放 443/3000 端口）

## E.8 应急响应

### 密钥泄露应急

```bash
# 1. 立即在服务提供商处吊销旧 Key
# 2. 生成新 Key
# 3. 更新 .env
# 4. 重启服务
docker compose -f deploy/docker-compose.yml restart api
# 5. 检查日志中是否有异常调用
grep "llm\|LLM" logs/app.log | tail -100
```

### 异常访问检测

```bash
# 检查最近的 API 请求
grep "GET\|POST" logs/app.log | tail -50 | awk '{print $1, $2, $NF}'

# 检查失败的认证请求
grep "401\|403\|Unauthorized" logs/app.log | tail -20

# IP 访问频次统计
grep -oP '\d+\.\d+\.\d+\.\d+' logs/app.log | sort | uniq -c | sort -rn | head -10
```

---

> **相关文档**: [[09-部署与运维]] [[01c-运维操作手册]] [[02-系统架构设计#2.7 网络拓扑与端口映射]]
