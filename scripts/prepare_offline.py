"""离线依赖制备工具 — 支持多平台并行下载，生成公司内网可用的离线包.

公司目标平台:
  1. Intel x86_64 Linux (无 GPU)     → CPU 嵌入 + 标准 Docker
  2. NVIDIA GB10 (ARM64 + CUDA)      → GPU 嵌入 + ARM Docker

用法:
    python scripts/prepare_offline.py                       # 制备全部 (默认 linux-x86_64 + linux-aarch64)
    python scripts/prepare_offline.py --platform linux-x86_64  # 仅 Intel
    python scripts/prepare_offline.py --platform linux-aarch64 # 仅 GB10
    python scripts/prepare_offline.py --pip-only            # 仅 pip wheels
    python scripts/prepare_offline.py --models-only         # 仅 HuggingFace 模型
    python scripts/prepare_offline.py --docker-only         # 仅 Docker 镜像

输出:
    offline/
    ├── wheels/
    │   ├── linux-x86_64/       ← Intel 服务器 pip 包
    │   └── linux-aarch64/      ← GB10 pip 包 (含 CUDA torch)
    ├── models/                 ← 平台无关, 全平台共享
    │   └── BAAI--bge-m3/
    ├── docker/
    │   ├── linux-amd64/        ← Intel x86 Docker 镜像
    │   └── linux-arm64/        ← GB10 ARM Docker 镜像
    └── manifest.json
"""

from __future__ import annotations

import json
import logging
import platform as _platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OFFLINE_DIR = PROJECT_ROOT / "offline"
WHEELS_DIR = OFFLINE_DIR / "wheels"
MODELS_DIR = OFFLINE_DIR / "models"
DOCKER_DIR = OFFLINE_DIR / "docker"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_offline")

# ── 平台配置 ──

# 公司目标平台定义
TARGET_PLATFORMS = {
    "linux-x86_64": {
        "label": "Intel x86_64 Linux (无 GPU)",
        "pip_platform": "manylinux2014_x86_64",
        "python_version": "3.11",
        "abi": "cp311",
        "docker_arch": "linux/amd64",
        "torch_index": None,  # 标准 PyPI
        "torch_version": "2.5.1",  # CPU only
    },
    "linux-aarch64": {
        "label": "NVIDIA GB10 (ARM64 + CUDA)",
        "pip_platform": "manylinux2014_aarch64",
        "python_version": "3.11",
        "abi": "cp311",
        "docker_arch": "linux/arm64",
        "torch_index": "https://download.pytorch.org/whl/cu128",  # CUDA 12.8 for Blackwell
        "torch_version": "2.5.1+cu128",
    },
}

# 当前机器平台 (用于模型/Docker 导出)
CURRENT_PLATFORM = f"{'macos' if _platform.system() == 'Darwin' else 'linux'}-{_platform.machine().lower()}"


# ── 1. Pip Wheels ──

def download_pip_wheels(platform_id: str):
    """为指定目标平台下载所有 pip wheels."""
    plat = TARGET_PLATFORMS[platform_id]
    wheels_dir = WHEELS_DIR / platform_id
    wheels_dir.mkdir(parents=True, exist_ok=True)

    req_path = PROJECT_ROOT / "requirements.txt"
    if not req_path.exists():
        logger.error("requirements.txt 不存在: %s", req_path)
        return

    logger.info("━━━ %s ━━━", plat["label"])
    logger.info("pip wheels → %s", wheels_dir)

    # 基础下载命令
    base_cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", str(req_path),
        "-d", str(wheels_dir),
        "--platform", plat["pip_platform"],
        "--python-version", plat["python_version"],
        "--implementation", "cp",
        "--abi", plat["abi"],
        "--only-binary", ":all:",
    ]

    # 尝试下载
    try:
        subprocess.run(base_cmd, check=True, cwd=str(PROJECT_ROOT))
    except subprocess.CalledProcessError:
        logger.warning("仅二进制下载失败 → 允许源码包")
        base_cmd.remove("--only-binary")
        base_cmd.remove(":all:")
        subprocess.run(base_cmd, check=True, cwd=str(PROJECT_ROOT))

    # 再下载传递依赖 (含纯 Python 包, 无平台限制)
    deps_cmd = [
        sys.executable, "-m", "pip", "download",
        "-r", str(req_path),
        "-d", str(wheels_dir),
    ]
    subprocess.run(deps_cmd, check=True, cwd=str(PROJECT_ROOT))

    # 特殊: 如果有 torch_index, 单独下载 PyTorch CUDA 版本
    if plat["torch_index"]:
        logger.info("  下载 PyTorch CUDA 版本 (ARM64)...")
        torch_cmd = [
            sys.executable, "-m", "pip", "download",
            f"torch=={plat['torch_version']}",
            "-d", str(wheels_dir),
            "--index-url", plat["torch_index"],
            "--platform", plat["pip_platform"],
            "--python-version", plat["python_version"],
            "--implementation", "cp",
            "--abi", plat["abi"],
            "--only-binary", ":all:",
        ]
        try:
            subprocess.run(torch_cmd, check=True, cwd=str(PROJECT_ROOT))
        except subprocess.CalledProcessError:
            logger.warning("PyTorch CUDA ARM 下载失败 → GB10 上需手动安装 torch")

    wheel_files = list(wheels_dir.glob("*.whl"))
    total_size = sum(f.stat().st_size for f in wheels_dir.iterdir())
    logger.info("  ✅ %d wheels, %.0f MB", len(wheel_files), total_size / 1e6)


# ── 2. HuggingFace 模型 (平台无关) ──

MODEL_NAMES = [
    "BAAI/bge-m3",
]


def copy_hf_cache():
    """从本地 HF 缓存复制模型 (不下载, 最快)."""
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"

    for model_name in MODEL_NAMES:
        safe_name = model_name.replace("/", "--")
        cache_dir = hf_cache / f"models--{safe_name}"
        target_dir = MODELS_DIR / safe_name

        if target_dir.exists():
            logger.info("模型已就绪: %s", model_name)
            continue
        if not cache_dir.exists():
            continue

        snapshots_dir = cache_dir / "snapshots"
        if not snapshots_dir.exists():
            continue

        snapshots = sorted(snapshots_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if not snapshots:
            continue

        target_dir.mkdir(parents=True, exist_ok=True)
        latest = snapshots[0]
        logger.info("复制: HF缓存 → %s", target_dir)
        for item in latest.iterdir():
            dest = target_dir / item.name
            if item.is_dir():
                if not dest.exists():
                    shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)

        size = sum(f.stat().st_size for f in target_dir.rglob("*") if f.is_file())
        logger.info("  ✅ %.0f MB", size / 1e6)


def download_hf_models():
    """从 HuggingFace Hub 下载模型 (兜底)."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        logger.error("需要 huggingface_hub: pip install huggingface_hub")
        return

    for model_name in MODEL_NAMES:
        safe_name = model_name.replace("/", "--")
        model_dir = MODELS_DIR / safe_name
        if model_dir.exists():
            continue

        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info("下载: %s", model_name)
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=str(model_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as e:
            logger.error("下载失败: %s → %s", model_name, e)
            shutil.rmtree(model_dir, ignore_errors=True)


def prepare_models():
    """制备模型文件."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    copy_hf_cache()

    # 补漏
    for model_name in MODEL_NAMES:
        safe_name = model_name.replace("/", "--")
        model_dir = MODELS_DIR / safe_name
        has_weights = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
        if not has_weights:
            logger.info("模型缺失, 从 Hub 下载: %s", model_name)
            download_hf_models()
            break

    for model_name in MODEL_NAMES:
        safe_name = model_name.replace("/", "--")
        model_dir = MODELS_DIR / safe_name
        has_weights = model_dir.exists() and any(model_dir.rglob("*.safetensors"))
        status = "✅" if has_weights else "❌"
        size_mb = sum(f.stat().st_size for f in model_dir.rglob("*") if f.is_file()) / 1e6 if has_weights else 0
        logger.info("  %s %s (%.0f MB)", status, model_name, size_mb)


# ── 3. Docker 镜像 ──

DOCKER_IMAGES = [
    ("milvusdb/milvus:v2.4.10", "milvus-v2.4.10"),
    ("minio/minio:RELEASE.2023-03-20T20-16-18Z", "minio-20230320"),
    ("quay.io/coreos/etcd:v3.5.5", "etcd-v3.5.5"),
]


def export_docker_images():
    """导出 Docker 镜像 — 分别为 x86_64 和 ARM64 下载."""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("Docker 不可用 → 跳过镜像导出")
        for img, _ in DOCKER_IMAGES:
            logger.info("  内网手动: docker pull %s", img)
        return

    for plat_id, plat_cfg in TARGET_PLATFORMS.items():
        docker_arch = plat_cfg["docker_arch"]
        arch_label = docker_arch.replace("/", "-")
        arch_dir = DOCKER_DIR / arch_label
        arch_dir.mkdir(parents=True, exist_ok=True)

        logger.info("━━━ Docker: %s (%s) ━━━", plat_cfg["label"], docker_arch)

        for image, filename in DOCKER_IMAGES:
            tar_path = arch_dir / f"{filename}.tar"
            if tar_path.exists():
                logger.info("  已存在: %s", tar_path.name)
                continue

            # Pull 指定架构
            logger.info("  pull --platform %s %s", docker_arch, image)
            try:
                subprocess.run(
                    ["docker", "pull", "--platform", docker_arch, image],
                    check=True,
                )
            except subprocess.CalledProcessError as e:
                logger.warning("  pull 失败 (%s) → 跳过", e)
                continue

            # 获取实际 image ID (多架构 pull 后需要)
            inspect = subprocess.run(
                ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                capture_output=True, text=True, check=True,
            )
            image_id = inspect.stdout.strip()

            # 导出
            logger.info("  save → %s", tar_path.name)
            subprocess.run(
                ["docker", "save", "-o", str(tar_path), image_id],
                check=True,
            )
            size_mb = tar_path.stat().st_size / 1e6
            logger.info("    ✅ %.0f MB", size_mb)


# ── 4. Manifest ──

def generate_manifest(platforms: list[str]) -> dict:
    """生成离线包清单."""
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "current_machine": CURRENT_PLATFORM,
        "target_platforms": {},
        "models": {},
        "docker": {},
    }

    # pip wheels per platform
    for pid in platforms:
        plat_cfg = TARGET_PLATFORMS[pid]
        wdir = WHEELS_DIR / pid
        wheel_files = sorted(wdir.glob("*.whl")) if wdir.exists() else []
        manifest["target_platforms"][pid] = {
            "label": plat_cfg["label"],
            "python": plat_cfg["python_version"],
            "wheels_dir": str(wdir.relative_to(PROJECT_ROOT)),
            "wheel_count": len(wheel_files),
            "size_mb": round(sum(f.stat().st_size for f in wheel_files) / 1e6, 1),
        }

    # models (shared)
    for model_name in MODEL_NAMES:
        safe_name = model_name.replace("/", "--")
        mdir = MODELS_DIR / safe_name
        if mdir.exists():
            files = list(mdir.rglob("*"))
            manifest["models"][model_name] = {
                "dir": str(mdir.relative_to(PROJECT_ROOT)),
                "size_mb": round(sum(f.stat().st_size for f in files if f.is_file()) / 1e6, 1),
                "files": len([f for f in files if f.is_file()]),
            }

    # docker per arch
    for arch_label in ["linux-amd64", "linux-arm64"]:
        adir = DOCKER_DIR / arch_label
        if adir.exists():
            tars = sorted(adir.glob("*.tar"))
            manifest["docker"][arch_label] = {
                "dir": str(adir.relative_to(PROJECT_ROOT)),
                "images": [t.name for t in tars],
                "size_mb": round(sum(t.stat().st_size for t in tars) / 1e6, 1),
            }

    manifest_path = OFFLINE_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("清单: %s", manifest_path)

    return manifest


def print_summary(manifest: dict):
    """打印中文摘要."""
    total_mb = 0
    print("\n" + "=" * 60)
    print("  离线依赖包制备完成")
    print("=" * 60)

    for pid, info in manifest["target_platforms"].items():
        mb = info["size_mb"]
        total_mb += mb
        print(f"  📦 {pid} ({info['label']}): {info['wheel_count']} 个 wheel, {mb:.0f} MB")

    for name, info in manifest["models"].items():
        mb = info["size_mb"]
        total_mb += mb
        print(f"  🧠 {name}: {mb:.0f} MB")

    for arch, info in manifest["docker"].items():
        mb = info["size_mb"]
        total_mb += mb
        print(f"  🐳 Docker {arch}: {mb:.0f} MB")

    print(f"\n  📁 总大小: ~{total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
    print(f"  📂 目录: {OFFLINE_DIR}")
    print()
    print("  下一步:")
    print("    1. 将 offline/ 目录拷贝到内网机器")
    print("    2. Intel x86: bash scripts/install_offline.sh linux-x86_64")
    print("    3. GB10 ARM:  bash scripts/install_offline.sh linux-aarch64")


# ── 入口 ──

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="制备 CommSpec RAG 离线依赖包 (公司内网部署用)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/prepare_offline.py                            # 默认: Intel + GB10 全部
  python scripts/prepare_offline.py --platform linux-x86_64    # 仅 Intel
  python scripts/prepare_offline.py --pip-only                 # 仅 pip
  python scripts/prepare_offline.py --models-only              # 仅模型
  python scripts/prepare_offline.py --docker-only              # 仅 Docker
        """,
    )
    parser.add_argument(
        "--platform", nargs="+",
        default=["linux-x86_64", "linux-aarch64"],
        choices=["linux-x86_64", "linux-aarch64"],
        help="目标平台 (可多选, 默认全选)",
    )
    parser.add_argument("--pip-only", action="store_true", help="仅 pip wheels")
    parser.add_argument("--models-only", action="store_true", help="仅模型")
    parser.add_argument("--docker-only", action="store_true", help="仅 Docker")
    parser.add_argument("--no-archive", action="store_true", help="不打包")
    args = parser.parse_args()

    OFFLINE_DIR.mkdir(parents=True, exist_ok=True)
    platforms = args.platform

    logger.info("当前机器: %s", CURRENT_PLATFORM)
    logger.info("目标平台: %s", ", ".join(platforms))
    logger.info("输出目录: %s", OFFLINE_DIR)

    all_mode = not (args.pip_only or args.models_only or args.docker_only)

    if all_mode or args.pip_only:
        for pid in platforms:
            download_pip_wheels(pid)

    if all_mode or args.models_only:
        prepare_models()

    if all_mode or args.docker_only:
        export_docker_images()

    manifest = generate_manifest(platforms)
    print_summary(manifest)


if __name__ == "__main__":
    main()
