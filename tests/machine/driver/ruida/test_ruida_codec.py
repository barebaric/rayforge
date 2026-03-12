"""
Tests for RuidaCodec class.
"""

from rayforge.machine.driver.ruida.ruida_codec import RuidaCodec
from rayforge.machine.driver.ruida.ruida_maps import CARD_ID_TO_MAGIC


class TestInit:
    """Test initialization."""

    def test_init_default_magic(self):
        codec = RuidaCodec()
        assert codec.magic == 0x88

    def test_init_custom_magic(self):
        codec = RuidaCodec(magic=0x55)
        assert codec.magic == 0x55

    def test_init_luts_built(self):
        codec = RuidaCodec()
        assert codec._swizzle_lut is not None
        assert codec._unswizzle_lut is not None
        assert len(codec._swizzle_lut) == 256
        assert len(codec._unswizzle_lut) == 256


class TestSetMagic:
    """Test set_magic method."""

    def test_set_magic_returns_true_when_changed(self):
        codec = RuidaCodec(magic=0x88)
        result = codec.set_magic(0x55)
        assert result is True
        assert codec.magic == 0x55

    def test_set_magic_returns_false_when_same(self):
        codec = RuidaCodec(magic=0x88)
        result = codec.set_magic(0x88)
        assert result is False
        assert codec.magic == 0x88

    def test_set_magic_updates_luts(self):
        codec = RuidaCodec(magic=0x88)
        old_swizzle = list(codec._swizzle_lut)
        codec.set_magic(0x55)
        assert list(codec._swizzle_lut) != old_swizzle


class TestSwizzle:
    """Test swizzle and unswizzle methods."""

    def test_swizzle_unswizzle_roundtrip_empty(self):
        codec = RuidaCodec()
        data = b""
        swizzled = codec.swizzle(data)
        unswizzled = codec.unswizzle(swizzled)
        assert unswizzled == data

    def test_swizzle_unswizzle_roundtrip_single_byte(self):
        codec = RuidaCodec()
        for byte_val in [0x00, 0x55, 0x88, 0xFF]:
            data = bytes([byte_val])
            swizzled = codec.swizzle(data)
            unswizzled = codec.unswizzle(swizzled)
            assert unswizzled == data

    def test_swizzle_unswizzle_roundtrip_multi_byte(self):
        codec = RuidaCodec()
        data = b"\x00\x01\x02\x03\x04\x05\x06\x07"
        swizzled = codec.swizzle(data)
        unswizzled = codec.unswizzle(swizzled)
        assert unswizzled == data

    def test_swizzle_different_magic_produces_different_output(self):
        codec88 = RuidaCodec(magic=0x88)
        codec55 = RuidaCodec(magic=0x55)
        data = b"\xda\x00\x05\x7e"
        swizzled88 = codec88.swizzle(data)
        swizzled55 = codec55.swizzle(data)
        assert swizzled88 != swizzled55

    def test_swizzle_is_deterministic(self):
        codec = RuidaCodec()
        data = b"\x01\x02\x03\x04"
        swizzled1 = codec.swizzle(data)
        swizzled2 = codec.swizzle(data)
        assert swizzled1 == swizzled2


class TestDetectMagicFromPayload:
    """Test detect_magic_from_payload method."""

    def test_detect_magic_from_payload_4_bytes(self):
        codec = RuidaCodec()
        original = b"\xda\x00\x05\x7e"
        for expected_magic in [0x00, 0x55, 0x88, 0xAA, 0xFF]:
            temp_codec = RuidaCodec(magic=expected_magic)
            swizzled = temp_codec.swizzle(original)
            detected = codec.detect_magic_from_payload(swizzled)
            assert detected == expected_magic

    def test_detect_magic_from_payload_wrong_length_returns_none(self):
        codec = RuidaCodec()
        assert codec.detect_magic_from_payload(b"") is None
        assert codec.detect_magic_from_payload(b"\x00") is None
        assert codec.detect_magic_from_payload(b"\x00\x01") is None
        assert codec.detect_magic_from_payload(b"\x00\x01\x02") is None
        assert codec.detect_magic_from_payload(b"\x00\x01\x02\x03\x04") is None

    def test_detect_magic_from_payload_unknown_pattern_returns_none(self):
        codec = RuidaCodec()
        unknown_payload = b"\xff\xff\xff\xff"
        assert codec.detect_magic_from_payload(unknown_payload) is None


class TestDetectMagicFromMemRequest:
    """Test detect_magic_from_mem_request method."""

    def test_detect_magic_from_mem_request_known_card(self):
        codec = RuidaCodec()
        for card_id, expected_magic in list(CARD_ID_TO_MAGIC.items())[:5]:
            mem_hi = (card_id >> 8) & 0xFF
            mem_lo = card_id & 0xFF
            unswizzled = bytes([0xDA, 0x00, mem_hi, mem_lo])
            detected = codec.detect_magic_from_mem_request(unswizzled)
            assert detected == expected_magic

    def test_detect_magic_from_mem_request_unknown_card_returns_none(self):
        codec = RuidaCodec()
        unswizzled = b"\xda\x00\xff\xff"
        result = codec.detect_magic_from_mem_request(unswizzled)
        assert result is None

    def test_detect_magic_from_mem_request_wrong_prefix_returns_none(self):
        codec = RuidaCodec()
        unswizzled = b"\xdb\x00\x05\x7e"
        result = codec.detect_magic_from_mem_request(unswizzled)
        assert result is None

    def test_detect_magic_from_mem_request_wrong_subcommand_returns_none(self):
        codec = RuidaCodec()
        unswizzled = b"\xda\x01\x05\x7e"
        result = codec.detect_magic_from_mem_request(unswizzled)
        assert result is None

    def test_detect_magic_from_mem_request_too_short_returns_none(self):
        codec = RuidaCodec()
        assert codec.detect_magic_from_mem_request(b"") is None
        assert codec.detect_magic_from_mem_request(b"\xda") is None
        assert codec.detect_magic_from_mem_request(b"\xda\x00") is None
        assert codec.detect_magic_from_mem_request(b"\xda\x00\x05") is None

    def test_detect_magic_from_mem_request_all_known_cards(self):
        codec = RuidaCodec()
        for card_id, expected_magic in CARD_ID_TO_MAGIC.items():
            mem_hi = (card_id >> 8) & 0xFF
            mem_lo = card_id & 0xFF
            unswizzled = bytes([0xDA, 0x00, mem_hi, mem_lo])
            detected = codec.detect_magic_from_mem_request(unswizzled)
            assert detected == expected_magic, (
                f"Card 0x{card_id:04X}->0x{expected_magic:02X}"
            )
