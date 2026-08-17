#!/usr/bin/env python3
"""人工知识校正 test_set.json 的 expected_sections — 生成 spec→章节 精确对应.

章节级 Recall 需要"每个 spec 的答案章节"精确标注。本脚本用 3GPP 规范知识,
为 40 题生成 {spec: [sections]} 映射, 写入 test_set.json 的 `expected` 字段
(新增, 不破坏现有 expected_specs/expected_sections 兼容)。

用法:
  python scripts/apply_expected_sections.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# question 关键词 → {spec: [sections]} (按 3GPP 规范知识标注)
# 章节号用"答案真实所在章节", 前缀匹配宽松 (如 6.3.3 可匹配 6.3.3.2)
_EXPECTED: dict[str, dict[str, list[str]]] = {
    "PRACH preamble 格式": {"38.211": ["6.3.3"]},
    "SSB（SS/PBCH block）的时频资源结构": {"38.211": ["7.4.3"], "38.213": ["4.1"]},
    "PDSCH 的 DMRS 有哪两种类型": {"38.211": ["7.4.1.1"]},
    "LDPC 码和 Polar 码": {"38.212": ["5.3"]},
    "PUSCH 的 DMRS 配置方式": {"38.211": ["6.4.1"], "38.214": ["6.1"]},
    "CORESET 和搜索空间": {"38.213": ["10.1"], "38.211": ["7.3.2"]},
    "UCI 在 PUSCH 上传输时的复用规则": {"38.213": ["9.3"]},
    "上行功率控制中，PUSCH 的发射功率": {"38.213": ["7.1"]},
    "速率匹配中，PDSCH/PUSCH 如何避开": {"38.212": ["5.4"]},
    "HARQ-ACK codebook 有哪两种类型": {"38.213": ["9.1"]},
    "小区搜索中，UE 如何通过 PSS/SSS": {"38.213": ["4.1"]},
    "SRS 资源集和 SRS 资源的配置关系": {"38.214": ["6.2"], "38.331": ["6.3.2"]},
    "SCell 的激活与去激活流程": {"38.321": ["5.9"], "38.213": ["4.3"]},
    "逻辑信道优先级（LCP）": {"38.321": ["5.4.3"]},
    "DRX 机制": {"38.321": ["5.7"]},
    "上行时间对齐": {"38.321": ["5.2"]},
    "RLC 层有哪三种传输模式": {"38.322": ["4.2"]},
    "PDCP 层的安全保护机制": {"38.323": ["5.8"]},
    "逻辑信道到传输信道的映射关系": {"38.321": ["4.4"]},
    "PDCP 层的重排序和重复检测": {"38.323": ["5.1"]},
    "波束失败恢复的完整流程": {"38.213": ["6"], "38.321": ["5.17"], "38.331": ["6.3.2"]},
    "HARQ 过程：PHY 层的 HARQ-ACK": {"38.213": ["9.1"], "38.321": ["5.3"]},
    "BWP（带宽部分）的配置（RRC）与切换": {"38.213": ["12"], "38.331": ["6.3.2"], "38.321": ["5.15"]},
    "TCI 状态的激活（MAC CE）与指示（DCI）": {"38.214": ["5.1"], "38.321": ["5.10"], "38.331": ["6.3.2"]},
    "随机接入：PRACH preamble": {"38.211": ["6.3.3"], "38.321": ["5.1"]},
    "4-step RACH 与 2-step RACH 的流程对比": {"38.213": ["8"], "38.321": ["5.1"]},
    "CBG（码块组）重传": {"38.213": ["9.1"], "38.214": ["5.3"]},
    "SPS（半持续调度）的配置（RRC）与激活": {"38.321": ["5.8"], "38.331": ["6.3.2"]},
    "RRC 状态（RRC_IDLE / RRC_INACTIVE": {"38.331": ["4.2"], "38.300": ["4.2"]},
    "测量配置：measurement object": {"38.331": ["5.5"]},
    "NR MAC 与 LTE MAC 在 HARQ 进程": {"38.321": ["5.3"], "36.321": ["5.3"]},
    "NR RRC 与 LTE RRC 在系统信息": {"38.331": ["5.2"], "36.331": ["5.2"]},
    "PDCCH 的 DCI 格式": {"38.212": ["7.3"]},
    "PDSCH/PUSCH 的处理时间": {"38.214": ["5.3", "6.4"]},
    "CSI-RS 的资源配置包含哪些参数": {"38.211": ["7.4.1.5"], "38.214": ["5.2"]},
    "PTRS（相位跟踪参考信号）": {"38.211": ["7.4.1.2"], "38.214": ["5.1"]},
    "上行控制信息（UCI）包含哪些类型": {"38.213": ["9.2"]},
    "下行预编码的码本类型": {"38.214": ["5.2"]},
    "双连接（EN-DC）下 MCG 与 SCG 的承载类型": {"38.323": ["4.2"], "38.331": ["4.2"]},
    "PUCCH 格式有哪些": {"38.211": ["6.3.2"], "38.213": ["9.2"]},
}


def main() -> None:
    path = PROJECT_ROOT / "tests" / "eval" / "test_set.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    applied = 0
    unmatched = []
    for item in data:
        q = item["question"]
        # 关键词匹配
        key = next((k for k in _EXPECTED if k in q), None)
        if key is None:
            unmatched.append(q[:40])
            continue
        item["expected"] = _EXPECTED[key]
        # 同步 expected_sections 为所有章节的并集 (保持向后兼容)
        all_secs: list[str] = []
        for secs in _EXPECTED[key].values():
            for s in secs:
                if s not in all_secs:
                    all_secs.append(s)
        item["expected_sections"] = all_secs
        applied += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"校正完成: {applied}/{len(data)} 题, 未匹配 {len(unmatched)} 题")
    for u in unmatched:
        print(f"  ⚠ 未匹配: {u}")


if __name__ == "__main__":
    main()
