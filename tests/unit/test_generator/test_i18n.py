"""i18n.py 单元测试 — 语言检测不依赖 LLM."""

from src.generator.i18n import detect_language


class TestDetectLanguage:

    # ── 中文检测 ──
    def test_zh_simple(self):
        assert detect_language("PDU会话建立流程是什么") == "zh"

    def test_zh_mixed(self):
        assert detect_language("5G中AMF与SMF的交互流程") == "zh"

    def test_zh_long(self):
        text = "在5G核心网架构中，PDU会话的建立需要经过多个步骤，包括UE发起请求、AMF选择SMF等"
        assert detect_language(text) == "zh"

    # ── 韩文检测 ──
    def test_ko_simple(self):
        assert detect_language("PDU 세션 설정 절차") == "ko"

    def test_ko_full(self):
        assert detect_language("5G 네트워크에서 PDU 세션을 설정하는 방법") == "ko"

    # ── 英文检测 (默认) ──
    def test_en_simple(self):
        assert detect_language("What is PDU Session Establishment?") == "en"

    def test_en_technical(self):
        assert detect_language("TS 38.413 NGAP PDU Session Resource Setup") == "en"

    def test_en_numbers(self):
        assert detect_language("3GPP 5G 38.300 R18 specification") == "en"

    # ── 边界情况 ──
    def test_empty_string(self):
        assert detect_language("") == "en"

    def test_pure_numbers(self):
        assert detect_language("12345 67890") == "en"

    def test_only_specials(self):
        assert detect_language("§8.3.1 Table 5-2") == "en"

    # ── 中韩混合 → 中文优先 (字符多) ──
    def test_zh_more_than_ko(self):
        # 中文字符多
        text = "PDU会话建立한국어"
        assert detect_language(text) == "zh"

    def test_ko_more_than_zh(self):
        # 韩文字符多
        text = "한국어PDU세션절차中"
        assert detect_language(text) == "ko"
