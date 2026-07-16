"""工具函数 — 跨平台硬件检测与通用工具."""

from __future__ import annotations

import logging
import platform
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


# ── 硬件平台信息 ──

@dataclass
class HardwareInfo:
    """跨平台硬件检测结果.

    用于指导嵌入设备选择、内存管理策略、批处理大小调优。
    """
    os_name: str               # "darwin" | "linux" | "win32"
    os_label: str              # "macOS" | "Linux" | "Windows"
    cpu_arch: str              # "arm64" | "x86_64" | "aarch64"
    cpu_label: str             # "Apple Silicon" | "Intel/AMD x64" | "NVIDIA Grace" | "ARM"
    gpu_type: str              # "nvidia_cuda" | "apple_mps" | "none"
    gpu_memory_mb: int         # CUDA: actual VRAM; MPS: 0 (unified memory not separately reported)
    recommended_device: str    # composite recommendation: "cuda" > "cpu" > "mps"(warned)
    is_unified_memory: bool    # Apple Silicon: unified memory architecture
    mps_known_issues: bool     # MPS: empty_cache 释放不彻底 + 统一内存回收策略差异


# ── 硬件检测缓存 ──
_best_device: str | None = None
_hardware_info: HardwareInfo | None = None


def get_hardware_info() -> HardwareInfo:
    """Detect and cache the current hardware platform information.

    Supported platforms:
    - NVIDIA GPU + CUDA (Linux/Windows x86_64): recommended cuda
    - Apple Silicon MPS (macOS arm64): detected but warned for memory issues, recommended cpu
    - NVIDIA Grace (Linux aarch64 + CUDA): recommended cuda
    - Intel/AMD x64 CPU (any OS): recommended cpu
    - Other ARM CPU (Raspberry Pi / cloud instances): recommended cpu

    Returns:
        HardwareInfo with complete platform metadata. First call detects and caches.
    """
    global _hardware_info
    if _hardware_info is not None:
        return _hardware_info

    # OS detection
    _os = platform.system().lower()  # darwin | linux | win32 (java legacy)
    _os_label = {"darwin": "macOS", "linux": "Linux", "win32": "Windows"}.get(_os, _os)

    # CPU architecture
    _arch = platform.machine().lower()
    if _arch in ("arm64", "aarch64"):
        if _os == "darwin":
            _cpu_label = "Apple Silicon"
        elif "grace" in platform.processor().lower() or "grace" in platform.uname().version.lower():
            _cpu_label = "NVIDIA Grace"
        else:
            _cpu_label = "ARM"
    elif _arch in ("x86_64", "amd64", "i386"):
        _cpu_label = "Intel/AMD x64"
    else:
        _cpu_label = _arch

    # GPU detection
    _gpu_type = "none"
    _gpu_memory_mb = 0
    _is_unified = False
    _mps_issues = False

    try:
        import torch

        # NVIDIA CUDA
        if torch.cuda.is_available():
            _gpu_type = "nvidia_cuda"
            try:
                _gpu_memory_mb = int(
                    torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                )
            except Exception:
                _gpu_memory_mb = 0

        # Apple MPS
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _gpu_type = "apple_mps"
            _is_unified = True
            _mps_issues = True
            _gpu_memory_mb = 0  # unified memory not separately reported

    except ImportError:
        pass  # torch not installed, CPU-only

    # Composite device recommendation
    if _gpu_type == "nvidia_cuda":
        _recommended = "cuda"
    elif _gpu_type == "apple_mps":
        # MPS has known memory leak issues (no empty_cache, unified memory doesn't free)
        # Safe only for small batches (<1000 chunks); large-scale embedding MUST use cpu
        _recommended = "cpu"
    else:
        _recommended = "cpu"

    _hardware_info = HardwareInfo(
        os_name=_os,
        os_label=_os_label,
        cpu_arch=_arch,
        cpu_label=_cpu_label,
        gpu_type=_gpu_type,
        gpu_memory_mb=_gpu_memory_mb,
        recommended_device=_recommended,
        is_unified_memory=_is_unified,
        mps_known_issues=_mps_issues,
    )

    logger.info(
        "硬件检测: %s / %s / GPU=%s%s",
        _os_label, _cpu_label, _gpu_type,
        f" ({_gpu_memory_mb}MB)" if _gpu_memory_mb else "",
    )
    if _mps_issues:
        logger.warning(
            "检测到 Apple MPS GPU — 注意: MPS empty_cache() 只释放 Metal 命令缓冲区,"
            "不保证立即回收统一内存。大批量嵌入推荐 EMBEDDING_DEVICE=cpu。"
        )

    return _hardware_info


def get_best_device() -> str:
    """Auto-detect optimal compute device: CUDA > CPU > MPS.

    Consistent with get_hardware_info().recommended_device:
    - Linux/Windows NVIDIA: returns "cuda"
    - macOS Apple Silicon: returns "cpu" (MPS memory leak risk)
    - No GPU / torch not installed: returns "cpu"

    To force MPS (accepting memory risk), set EMBEDDING_DEVICE=mps in .env.
    """
    global _best_device
    if _best_device is not None:
        return _best_device

    info = get_hardware_info()
    _best_device = info.recommended_device
    return _best_device


def extract_spec_number(doc_id: str, text: str = "") -> str:
    """从文档 ID 或文本首行提取 3GPP 规范编号."""
    digits = ""
    for ch in doc_id:
        if ch.isdigit():
            digits += ch
        elif digits:
            break
    if len(digits) >= 5:
        return f"{digits[0:2]}.{digits[2:5]}"
    first_line = text.split("\n", 1)[0] if text else ""
    m = re.search(r"TS\s*(\d{2}\.\d{3})", first_line, re.IGNORECASE)
    if m:
        return m.group(1)
    return ""


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
