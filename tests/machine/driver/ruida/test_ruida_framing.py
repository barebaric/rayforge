"""
Tests for Layer 2 (Framing) module.
"""

from rayforge.machine.driver.ruida.ruida_framing import (
    FramedPacket,
    PacketFramer,
    PacketStatus,
    estimate_packet_length,
    frame_packet,
    validate_packet,
)
from rayforge.machine.driver.ruida.ruida_util import calculate_checksum


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


class TestPacketFramer:
    """Test the PacketFramer class."""

    def test_empty_buffer(self):
        framer = PacketFramer()
        assert framer.buffer_size == 0
        packets = list(framer.extract_packets())
        assert packets == []

    def test_incomplete_packet(self):
        framer = PacketFramer()
        framer.add_data(b"\x00")
        packets = list(framer.extract_packets())
        assert packets == []
        assert framer.buffer_size == 1

    def test_single_valid_packet(self):
        framer = PacketFramer()
        payload = b"\xcc"
        packet = frame_packet(payload)
        framer.add_data(packet)
        packets = list(framer.extract_packets())
        assert len(packets) == 1
        assert packets[0].status == PacketStatus.VALID
        assert packets[0].payload == payload
        assert framer.buffer_size == 0

    def test_single_invalid_packet(self):
        framer = PacketFramer()
        framer.add_data(b"\xff\xff\xcc")
        packets = list(framer.extract_packets())
        assert len(packets) == 1
        assert packets[0].status == PacketStatus.INVALID_CHECKSUM
        assert packets[0].payload == b"\xcc"

    def test_multiple_packets(self):
        framer = PacketFramer()
        payload1 = b"\xcc"
        payload2 = b"\xce"
        packet1 = frame_packet(payload1)
        packet2 = frame_packet(payload2)
        framer.add_data(packet1 + packet2)
        packets = list(framer.extract_packets())
        assert len(packets) == 2
        assert packets[0].status == PacketStatus.VALID
        assert packets[0].payload == payload1
        assert packets[1].status == PacketStatus.VALID
        assert packets[1].payload == payload2

    def test_streaming_data(self):
        framer = PacketFramer()
        payload = b"\xcc"
        packet = frame_packet(payload)

        framer.add_data(packet[:1])
        packets = list(framer.extract_packets())
        assert packets == []

        framer.add_data(packet[1:])
        packets = list(framer.extract_packets())
        assert len(packets) == 1
        assert packets[0].status == PacketStatus.VALID

    def test_reset(self):
        framer = PacketFramer()
        framer.add_data(b"\x00\x01\x02")
        assert framer.buffer_size == 3
        framer.reset()
        assert framer.buffer_size == 0

    def test_packet_with_da_command(self):
        framer = PacketFramer()
        payload = b"\xda\x00\x05\x7e"
        packet = frame_packet(payload)
        framer.add_data(packet)
        packets = list(framer.extract_packets())
        assert len(packets) == 1
        assert packets[0].status == PacketStatus.VALID
        assert packets[0].payload == payload


class TestFramedPacket:
    """Test FramedPacket dataclass."""

    def test_valid_packet_fields(self):
        packet = FramedPacket(
            status=PacketStatus.VALID,
            payload=b"\x42",
            raw=b"\x00\x42\x42",
            expected_checksum=0x0042,
            actual_checksum=0x0042,
        )
        assert packet.status == PacketStatus.VALID
        assert packet.payload == b"\x42"
        assert packet.raw == b"\x00\x42\x42"

    def test_invalid_packet_fields(self):
        packet = FramedPacket(
            status=PacketStatus.INVALID_CHECKSUM,
            payload=b"\x42",
            raw=b"\xff\xff\x42",
            expected_checksum=0xFFFF,
            actual_checksum=0x0042,
        )
        assert packet.status == PacketStatus.INVALID_CHECKSUM
        assert packet.expected_checksum == 0xFFFF
        assert packet.actual_checksum == 0x0042
