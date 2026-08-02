# CHANGELOG — 项目更新记录

> 本文件按会话/迭代记录项目的功能变更、修复与优化内容。
> 与 `technical-whitepaper/`（系统现状描述）互补：本文件记录"改了什么"，白皮书描述"现在是什么"。

---

## 2026-08-02 会话总结：问答性能优化 + 前端美化 + 多标准支持

本次会话共 **66 项变更**（54 处修改、4 处移动/重命名、8 个新增文件，+1703/-390 行），
核心工作围绕四条主线：前端 UI 美化、去 3GPP-only 化改造、marked 数据集全量摄入、问答链路性能优化。

### 1. 前端 UI 美化（用户端 + 管理员端）

对用户对话页、文档页与管理员页面统一进行视觉规范调整：

| 项目 | 变更内容 |
|------|----------|
| Emoji | 用户端 + 管理员端界面中所有 emoji 全部移除 |
| 历史记录入口 | 不再显示独立图标，改为左侧浅灰色竖条（边栏），历史折叠为竖条内三道杠 SVG 图标 |
| 文档过滤器清除 | 文字"清除"改为浅灰无边框 X 图标，四角弧度 `2xl`，颜色略深于左侧竖条 |
| 精排按钮 | 去除文字与原有形状，改为"三道杠 + 调节器"SVG 图标，浅灰色、弧度 `xl` |
| 发送按钮 | 改为纸飞机 SVG 图标；运行时切换为停止图标，宽度与蓝色发送按钮一致 |
| 新对话按钮 | 进入对话状态后，在左侧竖条历史下方显示新对话按钮（SVG 图标）；初始界面隐藏 |
| 历史删除按钮 | 红色文字"删除"改为深灰色垃圾桶 SVG 图标（无填充、无边框） |

**涉及文件（13 个）：**

- `frontend/app/page.tsx` — 对话主界面（历史竖条、精排/发送/停止/新对话按钮）
- `frontend/app/layout.tsx` — 全局布局
- `frontend/app/documents/page.tsx` — 文档页（过滤器清除 X 图标）
- `frontend/lib/useConversationHistory.ts` — 历史记录管理
- `frontend/app/admin/layout.tsx` — 管理员布局
- `frontend/app/admin/page.tsx` — 管理台首页
- `frontend/app/admin/documents/page.tsx` — 文档管理
- `frontend/app/admin/documents/[id]/page.tsx` — 文档详情
- `frontend/app/admin/feedback/page.tsx` — 反馈管理
- `frontend/app/admin/ingestion/page.tsx` — 摄入管理
- `frontend/app/admin/login/login-form.tsx` — 登录页
- `frontend/app/admin/search/page.tsx` — 检索管理
- `frontend/app/admin/system/page.tsx` — 系统页

### 2. 多标准支持改造（去 3GPP-only）

- 系统已同时支持 3GPP 与 O-RAN 文档，前后端及代码中移除"仅支持 3GPP"类文案，检索界面不再展示 Release 版本列表，服务与认证标识统一改名为 CommSpec。
- 代码命名统一：文档转换实际使用 **pandoc**，将残留的 `docling` 相关命名（`DoclingExtractor` 等）改为 `pandoc`，依赖中移除 `docling`。
- 新增"其他"文档分类：既非 3GPP 也非 O-RAN 的文档在管理员后端标记为 `other`（上传自动归类 + 列表接口 + 摄入 source 参数）。

**涉及文件（12 个）：**

- `src/main.py` — 服务名/API 标题 "3GPP RAG" → "CommSpec RAG"
- `src/cli.py` — 文案 Docling → Pandoc
- `src/api/rest/admin_router.py`（+219 行）— 文档上传接口（自动归类 `marked|original|other`）、other 文档列表/删除接口
- `src/api/mcp/server.py`、`src/api/mcp/tools.py` — MCP 工具层去 3GPP 化
- `src/ingestion/__init__.py`、`src/ingestion/incremental.py` — `DoclingExtractor` → `PandocExtractor`
- `src/ingestion/splitter.py` — 注释去 "3GPP 规范" 限定
- `frontend/lib/api.ts` — 注释改名、`OtherDocumentItem` 接口、上传与 `triggerIngestion(source=marked|original|all)` 参数
- `frontend/middleware.ts` — 管理员认证 cookie 值改名
- `requirements.txt` — 移除 `docling` 依赖

### 3. marked 数据集全量摄入（方案 A）

- **方案 A 实施**：Markdown 图片处理由"直接删除图片标签"改为 **保留图片标题**——`![标题](路径)` 转为 `[图: 标题]` 文本；空 alt 的 `![]()` 与 `<img>` 标签仍删除。4,185 张图片的标题/说明文字可被检索。
- **Milvus 修复**：摄入期间因 `section_path` 等 VARCHAR 字段字节超限（1024）崩溃，删除并重建集合（上限提升至 4096），写入前做字节级截断，避免再次超限。
- **目录结构**：`data/documents` 下新增 `marked/`（默认嵌入数据源）、`original/`（pandoc 处理数据源）、`other/`（其他文档）划分，配置与脚本同步适配。
- **断点续跑**：实现/完善 checkpoint（`data/checkpoint/chunks_checkpoint.pkl`）与分段重跑（v3→v4→resume 日志），中断后可续跑。
- **摄入结果**：marked 数据集全量入库完成（2026-08-01），共 **163 个规格**（R18 × 162、O-RAN × 1）、**55,224 chunks**。
- **other 目录策略**：marked 下的其他类型文档暂不处理，待后续通知再摄入。

**涉及文件（10 个）：**

- `src/ingestion/extractor.py` — 方案 A（图片标题保留）
- `src/ingestion/orchestrator.py` — 摄入编排（marked/original 数据源）
- `src/retriever/milvus_store.py` — 字段上限 4096 + 写入前字节截断
- `src/config/settings.py`（+15 行）— 新增 `documents_marked_dir` / `documents_original_dir` / `documents_other_dir` 属性
- `.env.example` — `DOCUMENTS_DIR=data/documents`（原 R18 子目录）
- `scripts/bulk_ingest.py` — 断点续跑、marked 摄入
- `scripts/download_specs.py` — 默认输出目录改为 `original/`
- `scripts/prepare_offline.py` — 目录适配
- `src/ingestion/release_monitor.py` — 监控目录改为 `original/`
- `tests/test_ingestion/test_extractor_md_clean.py`（新增）、`tests/test_retriever/test_milvus_store.py`（新增）

### 4. 问答性能优化（本轮重点）

在不损失回答质量的前提下缩短单轮问答时间：

- **流式输出（SSE）**：回答改为逐 token 流式返回，首字延迟（TTFB）显著下降。
- **消除回译环节**：原"中文查询 → 翻译为英文 → 检索 → 回译为中文"链路中，回答生成改为按用户语言直接输出，减少一次 LLM 调用与串行等待。
- **合并 LLM 调用**：检索/生成阶段合并可并行的模型请求，减少调用次数与总耗时。
- **空流重试机制**：流式生成异常（空响应）时自动重试，避免白屏/无回答。
- **多跳缺口分析**：`max_tokens` 300 → 1024，提升子查询生成质量。

**涉及文件（5 个）：**

- `src/generator/llm_client.py` — 流式生成支持
- `src/generator/pipeline.py` — 回译消除、LLM 调用合并、空流重试
- `src/generator/i18n.py` — 按用户语言直接输出
- `src/api/rest/router.py` — SSE 流式端点
- `src/retriever/multi_hop.py` — 缺口分析 max_tokens 提升

### 5. 验证器与提示词修复

- **中文回答误报修复**：验证器改为语言感知的"双信号"策略（答案文本与检索上下文分别评估），修复中文回答被误判"答案与检索结果重合度 0%"的问题；支持引用编号检查。
- **提示词规则**：系统提示中禁止回答使用"直接回答"类标题；同时确认 DeepSeek 出现"直接回答"前缀并非丢弃 RAG 结果，而是模型习惯性表述，通过提示词约束消除。

**涉及文件（2 个）：**

- `src/generator/verifier.py` — 语言感知双信号验证
- `src/generator/prompt.py` — 禁止"直接回答"标题规则

### 6. 测试与验证

| 文件 | 状态 | 覆盖内容 |
|------|------|----------|
| `tests/test_generator/test_stream.py` | 新增 | 流式（SSE）响应 |
| `tests/test_api/test_admin_upload.py` | 新增 | 管理员上传 + other 分类 |
| `tests/test_ingestion/test_extractor_md_clean.py` | 新增 | 图片标题保留方案 |
| `tests/test_retriever/test_milvus_store.py` | 新增 | Milvus 存储层 |
| `tests/test_generator/test_verifier.py` | 更新（+65 行） | 语言感知验证 |
| `tests/test_generator/test_prompt.py` | 更新 | 提示词规则 |
| `tests/test_api_integration.py` | 更新（+31 行） | 端到端接口 |
| `tests/test_pipeline_integration.py` | 更新 | 管线集成 |
| `tests/eval/__init__.py`、`tests/eval/run_eval.py`、`tests/eval/test_run_eval.py` | 更新 | 评测运行器 |

### 7. 文档同步与整理

- 技术白皮书章节与最新实现同步（6 个章节文件）。
- docs 目录重组：新建总索引 `docs/README.md` 与变更记录 `docs/CHANGELOG.md`；根目录散落文档移入 `design/`（architecture、hardware-compatibility、ingestion-pipeline-deep-dive）与 `deployment/`（offline-deployment）；带空格文件名规范化；清理 .DS_Store。

**涉及文件（13 个）：**

- `docs/CHANGELOG.md`（新增）、`docs/README.md`（新增）、`docs/design/ingestion-pipeline-deep-dive.md`（新增）、`docs/optimization/chunk-llm-summary.md`（新增）
- `docs/technical-whitepaper/01-项目概览与目标.md`、`docs/technical-whitepaper/01d-故障排查手册.md`、`docs/technical-whitepaper/02-系统架构设计.md`、`docs/technical-whitepaper/03-关键技术选型.md`、`docs/technical-whitepaper/07-文档摄入管线.md`、`docs/technical-whitepaper/附录D-演进路线图.md`
- 移动：`docs/architecture.md` → `docs/design/architecture.md`、`docs/hardware-compatibility.md` → `docs/design/hardware-compatibility.md`、`docs/offline-deployment.md` → `docs/deployment/offline-deployment.md`
- 重命名：`docs/open_source/3gpp-rag-rel18 vs 3gpp-rag-project.md` → `docs/open_source/3gpp-rag-rel18-vs-3gpp-rag-project.md`
- `README.md` — 文档链接与目录树同步

### 8. 问题排查记录

- **前端整体放大**：Chrome 100% 缩放下界面整体变大，排查后重开恢复正常，疑为浏览器渲染缓存问题，未遗留代码改动。

### 遗留事项

- `test_config` 中部分默认值断言与当前配置不一致，需在后续整理。
- 检索结果波动（RRF 排名不稳）与首字延迟（TTFB）仍有优化空间。
- 管理员端上传页面标题含 `[id]` 路径参数渲染问题待验证。

---

## 更早记录（摘要）

- **v0.2.4**（上次提交）：多语言 i18n 管线、多跳推理、交叉引用、Release 感知、在线补充（Google CSE / TSpec-LLM）、答案溯源验证。
- 详细演进请参见 `docs/plans/phase1~phase6` 与 `docs/technical-whitepaper/附录D-演进路线图.md`。
