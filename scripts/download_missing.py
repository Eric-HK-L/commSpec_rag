#!/usr/bin/env python3
"""精准下载 36/38-series 缺失的 R18 规范，跳过一致性测试等低价值文件.

策略：
  - 下载 data/ 中不存在的规范 (对比 FTP 完整列表)
  - 跳过 36.1xx 一致性测试系列 (超大文件, 对 RAG 无价值)
  - 优先下载核心 RAN 协议 (36.3xx, 38.3xx/38.4xx)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# 添加项目根目录到 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.download_specs import SpecDownloader


def main():
    output = Path(__file__).resolve().parent.parent / "data" / "raw"
    downloader = SpecDownloader(output_dir=str(output), timeout=120)

    # ── 36-series: 优先核心协议，跳过一致性测试 ──
    # 高优先级 RAN 协议 (36.3xx)
    priority_36 = [
        "36322",  # LTE RLC ⬅ 本次 RAG 失败的根因
        "36323",  # LTE PDCP
        "36302",  # Physical channels requirements
        "36314",  # LTE-Advanced
        "36355",  # LTE Positioning
        "36360",  # Performance requirements
        "36361",  # Service requirements
    ]
    # RAN ASN.1 信令 (36.4xx)
    ran_sig_36 = [f"36{i}" for i in [
        401, 410, 411, 412, 414,
        420, 421, 422, 424, 425,
        440, 441, 442, 443, 444, 445,
        455, 456, 457, 458, 459,
        461, 462, 463, 464, 465,
    ]]
    # SA 服务层
    service_36 = [
        "36201", "36211", "36213", "36214", "36216",
        "36770",
        "36912", "36913", "36921", "36922", "36927",
        "36931", "36932", "36933", "36942", "36976",
    ]

    all_36 = priority_36 + ran_sig_36 + service_36

    print(f"\n{'='*60}")
    print(f"📥 下载 36-series 缺失规范 ({len(all_36)} 篇)")
    print(f"{'='*60}")
    print(f"  核心 RAN: {len(priority_36)} 篇")
    print(f"  ASN.1信令: {len(ran_sig_36)} 篇")
    print(f"  SA服务层: {len(service_36)} 篇")
    print()

    success_36 = 0
    fail_36 = 0
    for i, spec in enumerate(all_36):
        print(f"[{i+1}/{len(all_36)}] TS {spec[:2]}.{spec[2:]}")
        try:
            files = downloader.download(release="R18", spec=spec, dry_run=False)
            if files:
                success_36 += 1
            else:
                fail_36 += 1
                print("  ⚠️ 未找到")
        except Exception as e:
            fail_36 += 1
            print(f"  ❌ 错误: {e}")
        time.sleep(0.5)  # 礼貌性延迟

    print(f"\n36-series: {success_36} 成功, {fail_36} 失败")

    # ── 38-series: 优先接口协议 ──
    priority_38 = [
        "38314",  # L2/L3 Measurements
        "38411",  # N2AP (NG Application Protocol)
        "38412",  # N2F
        "38414",  # F1AP
        "38509",  # IAB
        "38901",  # NR User Plane
    ]
    ran_sig_38 = [
        "38421", "38422", "38424",
        "38460", "38461", "38462", "38463",
        "38471", "38472", "38474",
    ]
    other_38 = [
        "38133", "38151", "38171", "38175",
        "38523", "38533", "38551", "38561",
        "38822", "38850", "38880",
        "38894", "38896", "38897", "38898", "38899",
        "38912", "38913", "38921",
    ]

    all_38 = priority_38 + ran_sig_38 + other_38

    print(f"\n{'='*60}")
    print(f"📥 下载 38-series 缺失规范 ({len(all_38)} 篇)")
    print(f"{'='*60}")
    print(f"  核心接口: {len(priority_38)} 篇")
    print(f"  ASN.1信令: {len(ran_sig_38)} 篇")
    print(f"  其他TR:   {len(other_38)} 篇")
    print()

    success_38 = 0
    fail_38 = 0
    for i, spec in enumerate(all_38):
        print(f"[{i+1}/{len(all_38)}] TS {spec[:2]}.{spec[2:]}")
        try:
            files = downloader.download(release="R18", spec=spec, dry_run=False)
            if files:
                success_38 += 1
            else:
                fail_38 += 1
                print("  ⚠️ 未找到")
        except Exception as e:
            fail_38 += 1
            print(f"  ❌ 错误: {e}")
        time.sleep(0.5)

    print(f"\n38-series: {success_38} 成功, {fail_38} 失败")
    print(f"\n{'='*60}")
    print(f"📊 总计: {success_36 + success_38} 成功, {fail_36 + fail_38} 失败")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
