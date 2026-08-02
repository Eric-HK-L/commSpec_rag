# docs — 文档中心

CommSpec RAG 项目的全部文档导航。建议阅读顺序：**总览 → 白皮书 → 专题文档**。

## 📂 目录结构

```
docs/
├── README.md                  ← 本文件（总索引）
├── CHANGELOG.md               ← 变更记录（按会话/迭代记录"改了什么"）
├── design/                    ← 架构与设计文档
│   ├── architecture.md              系统架构总览
│   ├── hardware-compatibility.md    硬件兼容性说明
│   └── ingestion-pipeline-deep-dive.md  文档摄入管线深入分析
├── deployment/                ← 部署相关
│   └── offline-deployment.md        离线部署指南
├── technical-whitepaper/      ← 技术白皮书（当前系统完整描述，含目录导航 00）
├── plans/                     ← 历史实施计划（phase1~phase6）
├── optimization/              ← 性能优化专项方案
├── open_source/               ← 开源项目调研与对比
└── troubleshooting/           ← 问题排查记录
```

## 📖 快速导航

| 需求 | 入口 |
|------|------|
| 了解系统全貌 | `technical-whitepaper/00-目录与导航.md`（白皮书 MOC） |
| 快速部署上线 | `technical-whitepaper/01b-快速入门与部署指南.md` |
| 离线环境部署 | `deployment/offline-deployment.md` |
| 日常运维巡检 | `technical-whitepaper/01c-运维操作手册.md` |
| 故障排查 | `technical-whitepaper/01d-故障排查手册.md` + `troubleshooting/` |
| 查看近期改动 | `CHANGELOG.md` |
| 架构/摄入管线深入 | `design/` |
| 历史实施过程 | `plans/` |

## 🗂 各目录说明

- **technical-whitepaper/**：19 章完整技术白皮书，覆盖架构、选型、RAG 管线、检索增强、多语言、摄入、性能、部署、管理台、API 与安全，是系统现状的权威描述。
- **design/**：独立成篇的架构设计文档，比白皮书章节更深入，适合研发人员。
- **plans/**：各阶段实施计划（MVP1 → docx 摄入 → MVP3 → 优化 → chunk 升级 → 多轮对话存储），反映项目演进历史。
- **optimization/**：性能优化专项方案（如 chunk 级 LLM 语义摘要）。
- **open_source/**：对本仓库所参考的开源项目（3gpp-rag-rel18 等）的分析与对比。
- **troubleshooting/**：具体问题的排查过程与修复记录。
- **CHANGELOG.md**：随每次迭代更新的变更记录，与白皮书互补。

## ⚠️ 维护约定

1. 新增文档请放入对应子目录，并在本索引补充链接。
2. 每次迭代完成后，在 `CHANGELOG.md` 顶部追加更新记录。
3. 白皮书章节文件使用 `[[双向链接]]` 组织，可用 Obsidian 打开阅读。
