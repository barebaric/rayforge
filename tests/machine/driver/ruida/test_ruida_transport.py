"""
Tests for Ruida L2 Transport layer.

Tests framing, swizzle encoding/decoding, and magic key detection.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from rayforge.machine.driver.ruida.ruida_transport import (
    RuidaCodec,
    RuidaTransport,
    RuidaServerTransport,
)
from rayforge.machine.driver.ruida.ruida_util import frame_packet


class TestRuidaCodec:
    """Tests for RuidaCodec swizzle handling."""

    def test_default_magic(self):
        codec = RuidaCodec()
        assert codec.magic == 0x88

    def test_custom_magic(self):
        codec = RuidaCodec(magic=0x55)
        assert codec.magic == 0x55

    def test_swizzle_unswizzle_roundtrip(self):
        codec = RuidaCodec()
        original = b"\x00\x01\x02\xff\xce\xda"
        swizzled = codec.swizzle(original)
        unswizzled = codec.unswizzle(swizzled)
        assert unswizzled == original

    def test_swizzle_different_magic(self):
        codec88 = RuidaCodec(magic=0x88)
        codec55 = RuidaCodec(magic=0x55)
        original = b"\x00\x01\x02\xff"
        swizzled88 = codec88.swizzle(original)
        swizzled55 = codec55.swizzle(original)
        assert swizzled88 != swizzled55

    def test_set_magic_changes_lut(self):
        codec = RuidaCodec(magic=0x88)
        original = b"\x00\x01\x02"
        swizzled1 = codec.swizzle(original)

        codec.set_magic(0x55)
        swizzled2 = codec.swizzle(original)

        assert swizzled1 != swizzled2

    def test_set_same_magic_returns_false(self):
        codec = RuidaCodec(magic=0x88)
        assert codec.set_magic(0x88) is False

    def test_set_different_magic_returns_true(self):
        codec = RuidaCodec(magic=0x88)
        assert codec.set_magic(0x55) is True

    def test_detect_magic_from_payload_known_key(self):
        codec = RuidaCodec()
        swizzled = codec.swizzle(b"\xda\x00\x05\x7e")
        detected = codec.detect_magic_from_payload(swizzled)
        assert detected == 0x88

    def test_detect_magic_from_payload_unknown(self):
        codec = RuidaCodec()
        swizzled = codec.swizzle(b"\x00\x01\x02")
        detected = codec.detect_magic_from_payload(swizzled)
        assert detected is None

    def test_detect_magic_from_mem_request_card_id(self):
        codec = RuidaCodec()
        card_id = 0x2210
        mem_bytes = bytes([card_id >> 8, card_id & 0xFF])
        request = b"\xda\x00" + mem_bytes
        detected = codec.detect_magic_from_mem_request(request)
        assert detected == 0x16

    def test_detect_magic_from_mem_request_not_card_id(self):
        codec = RuidaCodec()
        mem_bytes = bytes([0x02, 0x00])
        request = b"\xda\x00" + mem_bytes
        detected = codec.detect_magic_from_mem_request(request)
        assert detected is None

    def test_detect_magic_from_mem_request_not_da(self):
        codec = RuidaCodec()
        request = b"\xce"
        detected = codec.detect_magic_from_mem_request(request)
        assert detected is None

    def test_detect_magic_from_mem_request_too_short(self):
        codec = RuidaCodec()
        request = b"\xda\x00"
        detected = codec.detect_magic_from_mem_request(request)
        assert detected is None


class TestRuidaTransport:
    """Tests for RuidaTransport client mode."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send",
                "purge",
                "received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send = AsyncMock()
        self.mock_transport.purge = AsyncMock()
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()
        self.mock_transport.is_connected = False

    def test_init_subscribes_to_transport(self):
        RuidaTransport(self.mock_transport)
        self.mock_transport.received.connect.assert_called_once()

    def test_magic_property(self):
        ruida = RuidaTransport(self.mock_transport, magic=0x55)
        assert ruida.magic == 0x55

    def test_magic_setter(self):
        ruida = RuidaTransport(self.mock_transport)
        ruida.magic = 0x55
        assert ruida.magic == 0x55

    @pytest.mark.asyncio
    async def test_connect_delegates(self):
        ruida = RuidaTransport(self.mock_transport)
        await ruida.connect()
        self.mock_transport.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_delegates(self):
        ruida = RuidaTransport(self.mock_transport)
        await ruida.disconnect()
        self.mock_transport.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_delegates(self):
        ruida = RuidaTransport(self.mock_transport)
        await ruida.send(b"raw")
        self.mock_transport.send.assert_called_once_with(b"raw")

    @pytest.mark.asyncio
    async def test_send_command_swizzles_and_frames(self):
        ruida = RuidaTransport(self.mock_transport)
        command = b"\xda\x00\x05\x7e"
        await ruida.send_command(command)

        self.mock_transport.send.assert_called_once()
        sent = self.mock_transport.send.call_args[0][0]

        assert len(sent) == 6
        assert sent[0:2] != command[0:2]

    @pytest.mark.asyncio
    async def test_send_response_swizzles_no_frame(self):
        ruida = RuidaTransport(self.mock_transport)
        response = b"\xcc"
        await ruida.send_response(response)

        self.mock_transport.send.assert_called_once()
        sent = self.mock_transport.send.call_args[0][0]

        assert len(sent) == 1
        assert sent != response

    def test_on_raw_received_valid_packet_emits_decoded(self):
        ruida = RuidaTransport(self.mock_transport)
        handler = MagicMock()
        ruida.decoded_received.connect(handler)

        original = b"\xce"
        codec = RuidaCodec()
        swizzled = codec.swizzle(original)
        framed = frame_packet(swizzled)

        connect_call = self.mock_transport.received.connect.call_args
        callback = connect_call[0][0]
        callback(self.mock_transport, framed)

        assert handler.called
        call_kwargs = handler.call_args[1]
        assert call_kwargs["data"] == original

    def test_on_raw_received_invalid_checksum_no_emit(self):
        ruida = RuidaTransport(self.mock_transport)
        handler = MagicMock()
        ruida.decoded_received.connect(handler)

        codec = RuidaCodec()
        swizzled = codec.swizzle(b"\xce")
        bad_framed = b"\xff\xff" + swizzled

        connect_call = self.mock_transport.received.connect.call_args
        callback = connect_call[0][0]
        callback(self.mock_transport, bad_framed)

        handler.assert_not_called()

    def test_on_raw_received_detects_magic_from_payload(self):
        ruida = RuidaTransport(self.mock_transport, magic=0x55)
        magic_handler = MagicMock()
        ruida.magic_changed.connect(magic_handler)

        codec = RuidaCodec(magic=0x88)
        magic_probe = codec.swizzle(b"\xda\x00\x05\x7e")
        framed = frame_packet(magic_probe)

        connect_call = self.mock_transport.received.connect.call_args
        callback = connect_call[0][0]
        callback(self.mock_transport, framed)

        assert ruida.magic == 0x88
        magic_handler.assert_called_once()

    @pytest.mark.asyncio
    async def test_purge_resets_framer(self):
        ruida = RuidaTransport(self.mock_transport)
        await ruida.purge()
        self.mock_transport.purge.assert_called_once()


class TestRuidaServerTransport:
    """Tests for RuidaServerTransport UDP server mode."""

    def setup_method(self):
        self.mock_transport = MagicMock(
            spec=[
                "connect",
                "disconnect",
                "send_to",
                "received",
                "status_changed",
            ]
        )
        self.mock_transport.connect = AsyncMock()
        self.mock_transport.disconnect = AsyncMock()
        self.mock_transport.send_to = AsyncMock()
        self.mock_transport.received = MagicMock()
        self.mock_transport.received.connect = MagicMock()
        self.mock_transport.status_changed = MagicMock()
        self.mock_transport.status_changed.connect = MagicMock()

    def test_init_subscribes_to_transport(self):
        RuidaServerTransport(self.mock_transport)
        self.mock_transport.received.connect.assert_called_once()

    def test_magic_property(self):
        ruida = RuidaServerTransport(self.mock_transport, magic=0x55)
        assert ruida.magic == 0x55

    @pytest.mark.asyncio
    async def test_connect_delegates(self):
        ruida = RuidaServerTransport(self.mock_transport)
        await ruida.connect()
        self.mock_transport.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_delegates(self):
        ruida = RuidaServerTransport(self.mock_transport)
        await ruida.disconnect()
        self.mock_transport.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_response_swizzles_no_frame(self):
        ruida = RuidaServerTransport(self.mock_transport)
        addr = ("192.168.1.100", 40200)
        response = b"\xcc"
        await ruida.send_response(response, addr)

        self.mock_transport.send_to.assert_called_once()
        sent, sent_addr = self.mock_transport.send_to.call_args[0]

        assert sent_addr == addr
        assert len(sent) == 1
        assert sent != response

    @pytest.mark.asyncio
    async def test_send_command_swizzles_and_frames(self):
        ruida = RuidaServerTransport(self.mock_transport)
        addr = ("192.168.1.100", 40200)
        command = b"\xda\x00\x05\x7e"
        await ruida.send_command(command, addr)

        self.mock_transport.send_to.assert_called_once()
        sent, sent_addr = self.mock_transport.send_to.call_args[0]

        assert sent_addr == addr
        assert len(sent) == 6
        assert sent[0:2] != command[0:2]

    def test_on_raw_received_valid_packet_emits_decoded(self):
        ruida = RuidaServerTransport(self.mock_transport)
        handler = MagicMock()
        ruida.decoded_received.connect(handler)

        original = b"\xce"
        codec = RuidaCodec()
        swizzled = codec.swizzle(original)
        framed = frame_packet(swizzled)
        addr = ("192.168.1.100", 40200)

        connect_call = self.mock_transport.received.connect.call_args
        callback = connect_call[0][0]
        callback(self.mock_transport, framed, addr)

        assert handler.called
        call_kwargs = handler.call_args[1]
        assert call_kwargs["data"] == original
        assert call_kwargs["addr"] == addr

    def test_on_raw_received_invalid_checksum_no_emit(self):
        ruida = RuidaServerTransport(self.mock_transport)
        handler = MagicMock()
        ruida.decoded_received.connect(handler)

        codec = RuidaCodec()
        swizzled = codec.swizzle(b"\xce")
        bad_framed = b"\xff\xff" + swizzled
        addr = ("192.168.1.100", 40200)

        connect_call = self.mock_transport.received.connect.call_args
        callback = connect_call[0][0]
        callback(self.mock_transport, bad_framed, addr)

        handler.assert_not_called()

    def test_on_raw_received_detects_magic_from_mem_request(self):
        ruida = RuidaServerTransport(self.mock_transport, magic=0x55)
        magic_handler = MagicMock()
        ruida.magic_changed.connect(magic_handler)

        codec = RuidaCodec(magic=0x88)
        mem_request = b"\xda\x00\x05\x7e"
        swizzled = codec.swizzle(mem_request)
        framed = frame_packet(swizzled)
        addr = ("192.168.1.100", 40200)

        connect_call = self.mock_transport.received.connect.call_args
        callback = connect_call[0][0]
        callback(self.mock_transport, framed, addr)

        assert ruida.magic == 0x88
        magic_handler.assert_called_once()


class TestSwizzleEncodingVariants:
    """Test swizzle encoding with various magic keys."""

    @pytest.mark.parametrize("magic", [0x00, 0x55, 0x88, 0xAA, 0xFF])
    def test_roundtrip_various_magic(self, magic):
        codec = RuidaCodec(magic=magic)
        test_data = [
            b"\x00",
            b"\xff",
            b"\xce",
            b"\xcc",
            b"\xda\x00\x05\x7e",
            bytes(range(256)),
        ]
        for original in test_data:
            swizzled = codec.swizzle(original)
            unswizzled = codec.unswizzle(swizzled)
            assert unswizzled == original, (
                f"Failed for magic 0x{magic:02X}, data {original.hex()}"
            )

    def test_swizzle_is_not_identity(self):
        codec = RuidaCodec()
        original = b"\x00\x01\x02"
        swizzled = codec.swizzle(original)
        assert swizzled != original

    def test_ack_byte_swizzled(self):
        codec = RuidaCodec(magic=0x88)
        ack = b"\xcc"
        swizzled = codec.swizzle(ack)
        unswizzled = codec.unswizzle(swizzled)
        assert unswizzled == ack
        assert len(swizzled) == 1
