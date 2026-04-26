"""
Tests for Ruida protocol utility functions.
"""

import pytest

from rayforge.machine.driver.ruida.ruida_util import (
    UM_PER_MM,
    build_swizzle_lut,
    calculate_checksum,
    decode14,
    decode35,
    decode_abs_coords,
    decode_rel_coords,
    decodeu14,
    decodeu35,
    encode14,
    encode35,
    estimate_packet_length,
    frame_packet,
    parse_mem,
    swizzle_byte,
    unswizzle_byte,
    validate_packet,
)


class TestSwizzleByte:
    """Test single-byte swizzle operations."""

    @pytest.mark.parametrize("magic", [0x00, 0x55, 0x88, 0xAA, 0xFF])
    @pytest.mark.parametrize("byte_val", [0x00, 0x42, 0x7F, 0x80, 0xCC, 0xFF])
    def test_roundtrip(self, byte_val, magic):
        swizzled = swizzle_byte(byte_val, magic)
        unswizzled = unswizzle_byte(swizzled, magic)
        assert unswizzled == byte_val

    def test_default_magic(self):
        assert swizzle_byte(0x00) == swizzle_byte(0x00, 0x88)

    def test_different_magic_different_result(self):
        assert swizzle_byte(0x42, 0x88) != swizzle_byte(0x42, 0x55)

    def test_swizzle_is_not_identity(self):
        assert swizzle_byte(0x00, 0x88) != 0x00
        assert swizzle_byte(0xFF, 0x88) != 0xFF


class TestBuildSwizzleLut:
    """Test lookup table generation."""

    def test_returns_two_luts(self):
        swiz, unswiz = build_swizzle_lut(0x88)
        assert len(swiz) == 256
        assert len(unswiz) == 256

    def test_lut_roundtrip(self):
        swiz, unswiz = build_swizzle_lut(0x88)
        for i in range(256):
            assert unswiz[swiz[i]] == i

    def test_different_magic_different_lut(self):
        swiz88, _ = build_swizzle_lut(0x88)
        swiz55, _ = build_swizzle_lut(0x55)
        assert swiz88 != swiz55


class TestEncode14:
    """Test 14-bit encoding."""

    def test_encode_zero(self):
        assert encode14(0) == b"\x00\x00"

    def test_encode_max(self):
        assert encode14(0x3FFF) == b"\x7f\x7f"

    def test_encode_truncates(self):
        assert encode14(0x7FFF) == encode14(0x3FFF)

    def test_roundtrip(self):
        for val in [0, 100, 1000, 0x3FFF]:
            encoded = encode14(val)
            decoded = decodeu14(encoded)
            assert decoded == val


class TestDecode14:
    """Test 14-bit signed decoding."""

    def test_decode_zero(self):
        assert decode14(b"\x00\x00") == 0

    def test_decode_positive(self):
        assert decode14(encode14(1000)) == 1000

    def test_decode_negative(self):
        assert decode14(encode14(-128)) == -128

    def test_roundtrip(self):
        for val in [0, 100, 1000, -100, -1000]:
            encoded = encode14(val)
            decoded = decode14(encoded)
            assert decoded == val


class TestDecodeU14:
    """Test 14-bit unsigned decoding."""

    def test_decode_zero(self):
        assert decodeu14(b"\x00\x00") == 0

    def test_decode_max(self):
        assert decodeu14(b"\x7f\x7f") == 0x3FFF

    def test_decode_positive(self):
        assert decodeu14(b"\x07\x68") == 1000


class TestEncode35:
    """Test 35-bit encoding."""

    def test_encode_zero(self):
        assert encode35(0) == b"\x00\x00\x00\x00\x00"

    def test_encode_positive(self):
        encoded = encode35(10000)
        assert len(encoded) == 5

    def test_encode_large(self):
        result = encode35(1000000)
        assert len(result) == 5

    def test_encode_max(self):
        result = encode35(0x7FFFFFFFF)
        assert len(result) == 5

    def test_encode_truncates(self):
        result = encode35(0xFFFFFFFFFFFFFFFF)
        assert result == encode35(0x7FFFFFFFF)

    def test_roundtrip(self):
        for val in [0, 100, 10000, 1000000]:
            encoded = encode35(val)
            decoded = decodeu35(encoded)
            assert decoded == val


class TestDecode35:
    """Test 35-bit signed decoding."""

    def test_decode_zero(self):
        assert decode35(b"\x00\x00\x00\x00\x00") == 0

    def test_decode_positive(self):
        encoded = encode35(10000)
        assert decode35(encoded) == 10000

    def test_decode_negative(self):
        result = decode35(b"\x7f\x7f\x7f\x7f\x7f")
        assert result < 0

    def test_roundtrip(self):
        for val in [0, 100, 10000, 1000000, -100, -10000]:
            encoded = encode35(val)
            decoded = decode35(encoded)
            assert decoded == val


class TestDecodeU35:
    """Test 35-bit unsigned decoding."""

    def test_decode_zero(self):
        assert decodeu35(b"\x00\x00\x00\x00\x00") == 0

    def test_decode_positive(self):
        encoded = encode35(10000)
        assert decodeu35(encoded) == 10000


class TestParseMem:
    """Test memory address parsing."""

    def test_parse_zero(self):
        assert parse_mem(b"\x00\x00") == 0x0000

    def test_parse_card_id(self):
        assert parse_mem(b"\x05\x7e") == 0x057E

    def test_parse_max(self):
        assert parse_mem(b"\xff\xff") == 0xFFFF


class TestCalculateChecksum:
    """Test checksum calculation."""

    def test_empty_data(self):
        assert calculate_checksum(b"") == 0

    def test_single_byte(self):
        assert calculate_checksum(b"\x42") == 0x42

    def test_multiple_bytes(self):
        assert calculate_checksum(b"\x01\x02\x03") == 0x06

    def test_overflow(self):
        assert calculate_checksum(b"\xff\xff\xff") == 0x2FD


class TestDecodeAbsCoords:
    """Test absolute coordinate decoding."""

    def test_decode_origin(self):
        data = encode35(0) + encode35(0)
        x, y = decode_abs_coords(data)
        assert x == 0.0
        assert y == 0.0

    def test_decode_positive_coords(self):
        x_um = 10000
        y_um = 20000
        data = encode35(x_um) + encode35(y_um)
        x, y = decode_abs_coords(data)
        assert x == pytest.approx(x_um / UM_PER_MM)
        assert y == pytest.approx(y_um / UM_PER_MM)

    def test_requires_10_bytes(self):
        with pytest.raises(IndexError):
            decode_abs_coords(b"\x00" * 5)


class TestDecodeRelCoords:
    """Test relative coordinate decoding."""

    def test_decode_zero(self):
        data = encode14(0) + encode14(0)
        dx, dy = decode_rel_coords(data)
        assert dx == 0.0
        assert dy == 0.0

    def test_decode_positive(self):
        dx_um = 100
        dy_um = -100
        data = encode14(dx_um) + encode14(dy_um)
        dx, dy = decode_rel_coords(data)
        assert dx == pytest.approx(dx_um / UM_PER_MM)
        assert dy == pytest.approx(dy_um / UM_PER_MM)


class TestFramePacket:
    """Test packet framing function."""

    def test_frame_empty_payload(self):
        payload = b""
        framed = frame_packet(payload)
        assert len(framed) == 2
        assert framed == b"\x00\x00"

    def test_frame_single_byte(self):
        payload = b"\x42"
        framed = frame_packet(payload)
        assert len(framed) == 3
        assert framed[0] == 0
        assert framed[1] == 0x42
        assert framed[2:] == payload

    def test_frame_multi_byte(self):
        payload = b"\xda\x00\x05\x7e"
        framed = frame_packet(payload)
        expected_checksum = calculate_checksum(payload)
        assert framed[0] == (expected_checksum >> 8)
        assert framed[1] == (expected_checksum & 0xFF)
        assert framed[2:] == payload

    def test_frame_checksum_correct(self):
        payload = bytes([0x01, 0x02, 0x03, 0x04])
        framed = frame_packet(payload)
        checksum = (framed[0] << 8) | framed[1]
        assert checksum == sum(payload) & 0xFFFF


class TestValidatePacket:
    """Test packet validation function."""

    def test_validate_empty_packet(self):
        is_valid, payload, exp, act = validate_packet(b"")
        assert is_valid is False
        assert payload == b""

    def test_validate_single_byte_packet(self):
        is_valid, payload, exp, act = validate_packet(b"\x00")
        assert is_valid is False
        assert payload == b""

    def test_validate_checksum_only(self):
        is_valid, payload, exp, act = validate_packet(b"\x00\x00")
        assert is_valid is True
        assert payload == b""
        assert exp == 0
        assert act == 0

    def test_validate_correct_checksum(self):
        payload = b"\x42\x43\x44"
        checksum = calculate_checksum(payload)
        packet = bytes([checksum >> 8, checksum & 0xFF]) + payload
        is_valid, ret_payload, exp, act = validate_packet(packet)
        assert is_valid is True
        assert ret_payload == payload
        assert exp == checksum
        assert act == checksum

    def test_validate_incorrect_checksum(self):
        packet = b"\xff\xff\x42\x43\x44"
        is_valid, ret_payload, exp, act = validate_packet(packet)
        assert is_valid is False
        assert ret_payload == b"\x42\x43\x44"
        assert exp == 0xFFFF
        assert act == calculate_checksum(b"\x42\x43\x44")


class TestEstimatePacketLength:
    """Test packet length estimation."""

    def test_empty_payload(self):
        assert estimate_packet_length(b"") == -1

    def test_ack_command(self):
        assert estimate_packet_length(b"\xcc") == 1

    def test_err_command(self):
        assert estimate_packet_length(b"\xcd") == 1

    def test_keepalive_command(self):
        assert estimate_packet_length(b"\xce") == 1

    def test_d0_command_incomplete(self):
        assert estimate_packet_length(b"\xd0") == -1

    def test_d0_command_complete(self):
        assert estimate_packet_length(b"\xd0\x00") == 2

    def test_d7_eof(self):
        assert estimate_packet_length(b"\xd7") == 1

    def test_d8_command_incomplete(self):
        assert estimate_packet_length(b"\xd8") == -1

    def test_d8_command_complete(self):
        assert estimate_packet_length(b"\xd8\x00") == 2

    def test_da_00_command(self):
        assert estimate_packet_length(b"\xda\x00\x05\x7e") == 4

    def test_a5_command_incomplete(self):
        assert estimate_packet_length(b"\xa5\x50") == -1

    def test_a5_command_complete(self):
        assert estimate_packet_length(b"\xa5\x50\x01") == 3

    def test_a7_command(self):
        assert estimate_packet_length(b"\xa7\x01") == 2

    def test_unknown_command_returns_len(self):
        assert estimate_packet_length(b"\xfe\x00\x00") == 3
