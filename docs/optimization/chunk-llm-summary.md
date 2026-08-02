Chunk 级 LLM 语义摘要增强方案 — 针对低语义密度 chunk（纯数字表格/公式块）离线摄入期用本地小型 LLM 生成自然语言摘要，存入 Milvus 字段，提升 Dense 检索召回与排序。

# 1 背景
## 1.1 问题描述
3GPP 规范经 pandoc 转换 + splitter 分块后，部分 chunk 语义密度极低：
- 纯数字表格 chunk：如 38.211 §6.3.3 PRACH preamble 表，被拆成 `1151 15 30 48...` 这类纯数字行，丢失表头和章节上下文
- 公式块 chunk：孤立 LaTeX 公式无文字说明
- 缩写/参数列表 chunk：无上下文的枚举值

这些 chunk 经 BGE-M3 嵌入后向量信息极低，Dense 检索 distance 偏低（实测 0.6063 vs 正常 0.6442），导致权威规范排名靠后。

## 1.2 现有方案不足
`docs/plans/table-chunk-optimization.md` 已规划方案 A（规则版摘要：表头注入 + 章节上下文 + 规则摘要），但：
- 规则摘要质量有限，无法处理复杂表格语义
- 规则覆盖不全，纯公式/缩写列表无对应规则

## 1.3 API 约束
当前 `LLM_PROVIDER=openai` 的远端 API（如 Gauss trial）RPM 仅 3，对 ~10 万 chunk 全量摘要几乎不可用（理论需 28000 分钟 ≈ 19 天）。必须本地部署小型 LLM，绕开 RPM 限制。

# 2 目标
## 2.1 核心目标
- 对低语义密度 chunk 生成 LLM 自然语言摘要，存入 Milvus `summary` 字段
- Dense 检索时优先编码 `summary + text` 的拼接文本，提升语义匹配
- 仅对需要摘要的 chunk 触发（预计 5-10%），控制成本

## 2.2 验证标准
以 38.211 PRACH 查询为基准（`scripts/diagnostics/diagnose_prach.py`）：
- 38.211 Dense top-20 排名 ≤ #3（当前 #7）
- 38.211 Dense distance ≥ 0.65（当前 0.6063）
- 38.211 RRF top-10 排名 ≤ #2
- 全量 eval set@10 提升但不回退（`tests/eval/run_eval.py`）

# 3 模型选型
## 3.1 候选模型对比
| 模型 | 参数量 | 显存(Q4_K_M GGUF) | 中英表格 | 推理速度 | 备注 |
|------|--------|---------------------|----------|----------|------|
| Qwen2.5-7B-Instruct | 7B | ~5GB | ★★★★★ | 中 | 首选：中英双语强，表格理解同级别最优 |
| Qwen2.5-3B-Instruct | 3B | ~2.5GB | ★★★★ | 快 | 显存紧张备选，速度 2x |
| Phi-3.5-mini-instruct | 3.8B | ~3GB | ★★★ | 快 | 英文技术文档强，中文弱 |
| Llama-3.2-3B-Instruct | 3B | ~2.5GB | ★★★ | 快 | 通用备选，英文为主 |

## 3.2 推荐方案
首选 Qwen2.5-7B-Instruct（GGUF Q4_K_M 量化），理由：
1. 中英双语：3GPP 规范有中英混合查询场景，Qwen 系列双语能力最强
2. 表格理解：7B 在同级别中对 Markdown Grid Table 的语义归纳最优
3. 无缝对接：项目 `llm_client.py` 已支持 OpenAI 兼容 API，本地起 Ollama 暴露 `http://localhost:11434/v1` 即可复用现有客户端，无需改代码
4. 部署成本：7B Q4 在 16GB+ 内存设备可跑（含 ARM 平台），与现有 BGE-M3 嵌入模型可共存

## 3.3 模型文件获取（内网离线）
```bash
# 在有网环境下载 GGUF 文件
# 地址：https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf
# 大小：~4.7GB

# 传输到内网后，放入 Ollama 模型目录或用 Modelfile 导入
# Modelfile 示例：
# FROM ./qwen2.5-7b-instruct-q4_k_m.gguf
# PARAMETER num_ctx 8192
# PARAMETER temperature 0.0

# ollama create qwen2.5:7b -f Modelfile
```

# 4 部署方案
## 4.1 Ollama 本地部署
复用 `.env.example` 已有的 Ollama 配置模式：
```env
# .env 新增（摄入期专用 LLM，与线上 LLM_PROVIDER 解耦）
SUMMARY_LLM_PROVIDER=openai
SUMMARY_LLM_BASE_URL=http://localhost:11434/v1
SUMMARY_LLM_API_KEY=ollama
SUMMARY_LLM_MODEL=qwen2.5:7b
SUMMARY_LLM_TEMPERATURE=0.0
SUMMARY_LLM_MAX_TOKENS=512
SUMMARY_LLM_TIMEOUT=60.0
```

## 4.2 复用 llm_client.py
`llm_client.py` 的 OpenAI 后端已支持任意 OpenAI 兼容 endpoint。摘要生成器只需实例化一个独立的 `LLMClient`（指向本地 Ollama），不污染线上 LLM 配置。

# 5 触发策略
## 5.1 低语义密度检测
在 `splitter.py` 切块后、`orchestrator._split_all()` 入库前，对每个 chunk 做启发式判断：

| 判定条件 | 说明 |
|---------|------|
| 数字占比 > 60% 且英文字符 < 20% | 纯数字表格行组 |
| 无英文字母单词 ≥ 4 字符 | 缩写/符号堆叠 |
| `_contains_table() == True` 且 chunk 文本 < 200 字符 | 小表格碎片 |
| `content_type == "parameter_table"` 且 `section_title` 为空 | 丢失标题的参数表 |

满足任一条件则触发 LLM 摘要。预计命中 5-10% chunk。

## 5.2 摘要 Prompt
```
You are a 3GPP specification expert. Summarize the following table/data chunk into ONE concise natural language sentence (max 80 tokens) that captures:
- What the table/configures (e.g., "PRACH preamble format configuration")
- Key parameters and their meaning

Output ONLY the summary sentence, no preamble.

Chunk metadata:
- Spec: {spec_number}
- Section: {section_number} {section_title}
- Parent: {parent_title}

Chunk content:
{chunk_text}
```

## 5.3 摘要存储
- 摘要存入 Milvus 新字段 `summary`（VARCHAR, max_length=512）
- 嵌入拼接策略：`embed_text = summary + "\n" + text`（摘要在前，BGE-M3 对前部更敏感）
- 原始 `text` 字段保留不变（供精确匹配和前端展示）

# 6 Milvus Schema 变更
## 6.1 新增字段
`src/retriever/milvus_store.py` 的 `create_collection()` 新增：
```python
FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=512),
```

## 6.2 Chunk 数据结构变更
`src/retriever/vector_store.py` 的 Chunk dataclass 新增：
```python
summary: str = ""  # LLM 生成的语义摘要（低语义密度 chunk 才有）
```

## 6.3 插入逻辑
`_insert_batch()` 新增 `summary` 列写入。`RetrievalResult` 同步新增 `summary` 字段。

## 6.4 检索时使用
`search.py` 的 `RetrievalResult.to_context_str()` 在 body 前注入摘要：
```
[TS 38.211 | §6.3.3 | Physical random-access channel (R18)]
[摘要] PRACH preamble format configuration table: format=0 maps to samples=1151, cp=15...
<原始 chunk 文本>
```

# 7 摄入流程变更
## 7.1 orchestrator 新增步骤
`src/ingestion/orchestrator.py` 的 `run_full_pipeline()` 在 split 之后、embed 之前插入：
```
Step 4.5: LLM 摘要生成（仅低语义 chunk）
├─ 遍历 chunks，启发式判定低语义密度
├─ 批量调用本地 LLM 生成摘要
├─ 摘要写入 chunk.summary
└─ 嵌入时拼接 summary + text
```
- 摘要结果落盘到 `data/interim/{key}_summaries.json`，支持 `--skip-summary` 跳过
- 复用 `data/checkpoint/` 机制，单 chunk 摘要失败可重试

# 8 实施步骤
## 8.1 Phase 1: 部署验证（1 天）
1. 内网部署 Ollama + Qwen2.5-7B-Instruct GGUF
2. 用 `.env` 配置 `SUMMARY_LLM_*` 环境变量
3. 手动对 38.211 PRACH chunk 跑 10 条摘要，人工评估质量

## 8.2 Phase 2: 代码实现（2-3 天）
1. `vector_store.py`：Chunk 新增 `summary` 字段
2. `milvus_store.py`：schema 新增 `summary`，`_insert_batch` 写入
3. `splitter.py`：新增 `is_low_density_chunk()` 启发式函数
4. 新增 `src/ingestion/summarizer.py`：LLM 摘要生成器（复用 llm_client）
5. `orchestrator.py`：在 split 与 embed 之间插入摘要步骤
6. `search.py`：`RetrievalResult` + `to_context_str` 支持 summary

## 8.3 Phase 3: 增量摄入与验证（1 天）
1. 重建 Milvus collection（drop + create，因 schema 变更）
2. 先对 38.211 + 38.213 重跑摄入，验证 `diagnose_prach.py`
3. 达标后全量重跑（预计 ~10 万 chunk × 5-10% × 2s/chunk ≈ 3-6 小时）

## 8.4 Phase 4: 全量评估（半天）
1. 运行 `tests/eval/run_eval.py` 对比 recall@10
2. 抽查 20 条低语义 chunk 的摘要质量
3. 更新 `docs/technical-whitepaper/07-文档摄入管线.md`

# 9 风险与回滚
## 9.1 风险
| 风险 | 缓解 |
|------|------|
| LLM 摘要失真（漏关键参数） | `temperature=0.0` + 人工抽检 + 保留原始 text |
| 本地 LLM 推理慢拖累摄入 | 仅 5-10% chunk 触发 + 断点续传 |
| 摘要引入幻觉术语 | prompt 约束“仅基于 chunk 内容”，不引入外部知识 |
| Milvus schema 变更需重建 collection | 接受全量重摄入成本 |

## 9.2 回滚
- `summary` 字段为空时，检索降级为纯 text 嵌入（向前兼容）
- 关闭摘要：设 `SUMMARY_LLM_PROVIDER=`（空）则跳过摘要步骤

# 10 与现有方案的关系
| 方案 | 状态 | 关系 |
|------|------|------|
| `table-chunk-optimization.md` 方案 A（规则摘要） | 推荐先做 | 本方案的前置基线，规则能覆盖的不用 LLM |
| 本方案（LLM 摘要） | 规则方案 A 验证不足后启动 | 对规则无法覆盖的 chunk 兜底 |
| Cross-Encoder Reranker | 已部署 | 互补：LLM 摘要解决初筛召回，reranker 解决精排 |

执行顺序：先实施方案 A（规则）→ 验证 → 不足部分用本方案 LLM 摘要补齐。

# 11 标签
`#optimization #retrieval-quality #chunk-summary #llm #ollama #qwen #milvus-schema`