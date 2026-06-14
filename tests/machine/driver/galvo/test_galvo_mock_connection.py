"""
Tests for MockGalvoConnection.

Verifies the mock correctly simulates USB communication with
EzCad2/LMC controller boards.
"""

import struct

from rayforge.machine.driver.galvo.galvo_consts import (
    COMMAND_SIZE,
    LIST_BUFFER_SIZE,
    GetSerialNo,
    GetVersion,
    ReadPort,
)
from rayforge.machine.driver.galvo.galvo_mock_connection import (
    MockGalvoConnection,
)


class TestMockGalvoConnection:
    def test_initial_state(self):
        """Test that a new mock connection starts disconnected."""
        conn = MockGalvoConnection()
        assert not conn.is_connected
        assert conn.sent_singles == []
        assert conn.sent_lists == []

    def test_open_and_close(self):
        """Test open/close lifecycle."""
        conn = MockGalvoConnection()
        idx = conn.open()
        assert idx == 0
        assert conn.is_connected

        conn.close()
        assert not conn.is_connected

    def test_write_single_command(self):
        """Test writing a single command stores it."""
        conn = MockGalvoConnection()
        conn.open()

        cmd = struct.pack("<6H", 0x0007, 0, 0, 0, 0, 0)
        conn.write(packet=cmd)

        assert len(conn.sent_singles) == 1
        assert conn.sent_singles[0].cmd == 0x0007

    def test_write_list_packet(self):
        """Test writing a list packet stores it."""
        conn = MockGalvoConnection()
        conn.open()

        packet = bytearray(LIST_BUFFER_SIZE)
        for i in range(0, LIST_BUFFER_SIZE, COMMAND_SIZE):
            struct.pack_into("<6H", packet, i, 0x8001, 100, 200, 0, 500, 0)
        conn.write(packet=bytes(packet))

        assert len(conn.sent_lists) == 1

    def test_read_returns_implied_response(self):
        """Test that set_implied_response controls read output."""
        conn = MockGalvoConnection()
        conn.open()

        expected = struct.pack("<4H", 1, 2, 3, 4)
        conn.set_implied_response(expected)
        result = conn.read()

        assert result == expected

    def test_read_random_when_no_implied_response(self):
        """Test that read returns random data when no response is set."""
        conn = MockGalvoConnection()
        conn.open()

        result = conn.read()
        assert len(result) == 8

    def test_get_serial_number_response(self):
        """Test that GetSerialNo sets implied response."""
        conn = MockGalvoConnection()
        conn.open()

        cmd = struct.pack("<6H", GetSerialNo, 0, 0, 0, 0, 0)
        conn.write(packet=cmd)
        result = conn.read()

        assert b"MOCK" in result

    def test_get_version_response(self):
        """Test that GetVersion sets implied response."""
        conn = MockGalvoConnection()
        conn.open()

        cmd = struct.pack("<6H", GetVersion, 0, 0, 0, 0, 0)
        conn.write(packet=cmd)
        result = conn.read()

        words = struct.unpack("<4H", result)
        assert words[3] != 0

    def test_read_port_response(self):
        """Test that ReadPort returns port bits."""
        conn = MockGalvoConnection()
        conn.open()

        cmd = struct.pack("<6H", ReadPort, 0, 0, 0, 0, 0)
        conn.write(packet=cmd)
        result = conn.read()

        words = struct.unpack("<4H", result)
        assert words[1] == 0

    def test_was_list_executed_after_list_write(self):
        """Test that was_list_executed tracks list writes."""
        conn = MockGalvoConnection()
        conn.open()

        assert not conn.was_list_executed

        packet = bytearray(LIST_BUFFER_SIZE)
        conn.write(packet=bytes(packet))

        assert conn.was_list_executed

    def test_reset_clears_state(self):
        """Test that reset() clears all recorded state."""
        conn = MockGalvoConnection()
        conn.open()

        cmd = struct.pack("<6H", 0x0007, 0, 0, 0, 0, 0)
        conn.write(packet=cmd)
        packet = bytearray(LIST_BUFFER_SIZE)
        conn.write(packet=bytes(packet))

        assert len(conn.sent_singles) == 1
        assert len(conn.sent_lists) == 1

        conn.reset()

        assert conn.sent_singles == []
        assert conn.sent_lists == []
        assert not conn.was_list_executed

    def test_on_send_callback_for_single(self):
        """Test that on_send callback is called for single commands."""
        received = []

        def callback(msg):
            received.append(msg)

        conn = MockGalvoConnection(on_send=callback)
        conn.open()

        cmd = struct.pack("<6H", 0x0007, 0, 0, 0, 0, 0)
        conn.write(packet=cmd)

        assert len(received) == 1
        assert "GetVersion" in received[0]

    def test_on_send_callback_for_list(self):
        """Test that on_send callback is called for list packets."""
        received = []

        def callback(msg):
            received.append(msg)

        conn = MockGalvoConnection(on_send=callback)
        conn.open()

        packet = bytearray(LIST_BUFFER_SIZE)
        conn.write(packet=bytes(packet))

        assert len(received) > 0

    def test_write_raises_for_no_packet(self):
        """Test that writing None does nothing."""
        conn = MockGalvoConnection()
        conn.open()

        conn.write(packet=None)
        assert conn.sent_singles == []
        assert conn.sent_lists == []

    def test_reopen_resets_state(self):
        """Test that reopening clears sent commands."""
        conn = MockGalvoConnection()
        conn.open()

        cmd = struct.pack("<6H", 0x0007, 0, 0, 0, 0, 0)
        conn.write(packet=cmd)

        conn.close()
        conn.open()

        assert conn.sent_singles == []
        assert conn.sent_lists == []
