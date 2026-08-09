"""3GPP 缩写术语表 — 查询侧缩写展开 (Telco-RAG 论文验证的高性价比召回手段).

内嵌 dict: 缩写 → (英文全称, 中文释义), 后续可迁移为独立 JSON 文件.
展开用于检索侧: "RRC" → "RRC (Radio Resource Control)", 提升缩写类查询召回.
"""

from __future__ import annotations

import re

# 缩写 → (英文全称, 中文释义)
_ABBREVIATIONS: dict[str, tuple[str, str]] = {
    "AMF": ("Access and Mobility Management Function", "接入与移动性管理功能"),
    "CSI": ("Channel State Information", "信道状态信息"),
    "HARQ": ("Hybrid Automatic Repeat reQuest", "混合自动重传请求"),
    "IMS": ("IP Multimedia Subsystem", "IP 多媒体子系统"),
    "MAC": ("Medium Access Control", "媒体接入控制"),
    "MIB": ("Master Information Block", "主信息块"),
    "NGAP": ("NG Application Protocol", "NG 应用协议"),
    "NR": ("New Radio", "新空口"),
    "PDCP": ("Packet Data Convergence Protocol", "分组数据汇聚协议"),
    "PDU": ("Protocol Data Unit", "协议数据单元"),
    "PDSCH": ("Physical Downlink Shared Channel", "物理下行共享信道"),
    "PUSCH": ("Physical Uplink Shared Channel", "物理上行共享信道"),
    "QoS": ("Quality of Service", "服务质量"),
    "RACH": ("Random Access Channel", "随机接入信道"),
    "RLC": ("Radio Link Control", "无线链路控制"),
    "RRC": ("Radio Resource Control", "无线资源控制"),
    "SIB": ("System Information Block", "系统信息块"),
    "SMF": ("Session Management Function", "会话管理功能"),
    "SMS": ("Short Message Service", "短消息业务"),
    "UPF": ("User Plane Function", "用户面功能"),
    "XnAP": ("Xn Application Protocol", "Xn 应用协议"),
}

# 按缩写长度降序预编译 — 长缩写先替换, 避免前缀重叠歧义
_ABBREV_ITEMS: list[tuple[str, str, re.Pattern[str]]] = [
    (
        abbr,
        full_name,
        re.compile(
            # 词边界: 两侧不得为 ASCII 字母/数字/下划线, 避免 "RRC" 误匹配 "RRCR"/"RRC1"
            # 允许 CJK 邻接: "RRC连接" 同样展开; 括号内已展开形式 "(RRC)" 跳过
            rf"(?<![A-Za-z0-9_(]){re.escape(abbr)}(?![A-Za-z0-9_]|\s*\()",
            re.IGNORECASE,
        ),
    )
    for abbr, (full_name, _cn) in sorted(
        _ABBREVIATIONS.items(), key=lambda kv: len(kv[0]), reverse=True
    )
]


def expand_abbreviations(text: str) -> str:
    """将文本中的 3GPP 缩写展开为 "缩写 (英文全称)" 形式, 用于检索.

    规则:
    - 大小写不敏感 ("rrc"/"RRC"/"RrC" 均匹配)
    - 词边界匹配, 避免 "RRC" 误匹配 "RRCR"/"RRC1"
    - 中英混合查询可用 ("RRC连接建立流程" 中的 "RRC" 同样展开)
    - 已展开形式 ("RRC (Radio Resource Control)" / "(RRC)") 不重复展开
    - 无缩写命中时原样返回
    """
    result = text
    for abbr, full_name, pattern in _ABBREV_ITEMS:
        result = pattern.sub(f"{abbr} ({full_name})", result)
    return result
