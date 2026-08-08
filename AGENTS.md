# commSpec_rag 项目指引

## PyTorch MPS 内存配置（踩坑要点，重要）

- PyTorch 2.13+ 的 MPS 支持 `PYTORCH_MPS_HIGH_WATERMARK_RATIO` /
  `PYTORCH_MPS_LOW_WATERMARK_RATIO` 水位线控制；默认 HIGH=1.0 允许吃满系统内存，
  会导致 wired 内存爆炸，**两个变量必须同时设置**（LOW 未设时自动 = HIGH×4，须 ≤1.0）。
- 最优实践：batch_size=4 + HIGH=0.5 + LOW=0.3（wired 约 8GB）。
- PyTorch ≥2.13 中 `torch.mps.empty_cache()` 已能实质释放 Metal 命令缓冲区与中间分配，
  可有效缓解 wired 累积，无需等进程退出。

## 运行环境

- Python 虚拟环境：`.venv/`（运行前先 `source .venv/bin/activate`）。
- 后端 FastAPI 监听 :8000，前端 :3000，向量库 Milvus。
- 全栈冒烟测试：`./scripts/smoke_test.sh`（健康检查/检索/过滤/LLM 问答/后台认证/前端
  共 6 项，约 2-4 分钟，退出码 0 = 全部通过）。该脚本纯 localhost 只读查询，无风险。
- LLM：DeepSeek deepseek-v4-flash，密钥从 ~/ds-api-key 读取。

## 提交约定

- git add/commit 可自主执行（提交信息准确概括变更）；git push 须用户明确要求。
