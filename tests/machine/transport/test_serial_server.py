import asyncio
import os
import pytest

from rayforge.machine.transport.serial_server import SerialServerTransport
from rayforge.machine.transport import TransportStatus


class SignalTracker:
    """A helper to track calls to a blinker Signal."""

    def __init__(self, signal):
        self.calls = []
        signal.connect(self._callback)

    def _callback(self, sender, **kwargs):
        self.calls.append({"sender": sender, "kwargs": kwargs})

    def clear(self):
        self.calls = []

    @property
    def last_call(self):
        return self.calls[-1] if self.calls else None

    @property
    def last_data(self) -> bytes:
        """Returns the data from the last 'received' signal call."""
        if not self.calls:
            return b""
        return self.calls[-1]["kwargs"].get("data", b"")


@pytest.mark.asyncio
class TestSerialServerTransport:
    """Tests for SerialServerTransport using real PTY."""

    async def test_connect_creates_pty(self):
        """Test that connect() creates a PTY with a valid slave path."""
        transport = SerialServerTransport(baudrate=115200)

        try:
            await transport.connect()
            assert transport.is_connected
            assert transport.slave_path is not None
            assert os.path.exists(transport.slave_path)
        finally:
            await transport.disconnect()

    async def test_disconnect_closes_pty(self):
        """Test that disconnect() closes the PTY."""
        transport = SerialServerTransport()

        await transport.connect()
        slave_path = transport.slave_path
        assert slave_path is not None

        await transport.disconnect()
        assert not transport.is_connected
        assert transport.slave_path is None

    async def test_connect_disconnect_signals(self):
        """Test status signals during connect/disconnect cycle."""
        transport = SerialServerTransport()
        status_tracker = SignalTracker(transport.status_changed)

        await transport.connect()
        await transport.disconnect()

        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [
            TransportStatus.CONNECTING,
            TransportStatus.CONNECTED,
            TransportStatus.DISCONNECTED,
        ]

    async def test_connect_when_already_connected(self):
        """Test that connecting twice is a no-op."""
        transport = SerialServerTransport()

        try:
            await transport.connect()
            assert transport.is_connected
            slave_path = transport.slave_path

            await transport.connect()
            assert transport.is_connected
            assert transport.slave_path == slave_path
        finally:
            await transport.disconnect()

    async def test_disconnect_when_not_connected(self):
        """Test that disconnecting when not connected is safe."""
        transport = SerialServerTransport()
        assert not transport.is_connected

        await transport.disconnect()
        assert not transport.is_connected

    async def test_send_without_connect(self):
        """Test that sending without connecting raises ConnectionError."""
        transport = SerialServerTransport()

        with pytest.raises(ConnectionError, match="not started"):
            await transport.send(b"test")

    async def test_purge_is_noop(self):
        """Test that purge() is a no-op for serial server."""
        transport = SerialServerTransport()

        try:
            await transport.connect()
            await transport.purge()
        finally:
            await transport.disconnect()

    async def test_purge_on_disconnected_transport(self):
        """Test that purge on disconnected transport is safe."""
        transport = SerialServerTransport()
        assert not transport.is_connected

        await transport.purge()

    async def test_baudrate_property(self):
        """Test that baudrate is stored correctly."""
        transport = SerialServerTransport(baudrate=9600)
        assert transport.baudrate == 9600

        transport = SerialServerTransport(baudrate=115200)
        assert transport.baudrate == 115200

    async def test_slave_path_property(self):
        """Test slave_path property returns None before connect."""
        transport = SerialServerTransport()
        assert transport.slave_path is None

        try:
            await transport.connect()
            assert transport.slave_path is not None
            assert isinstance(transport.slave_path, str)
        finally:
            await transport.disconnect()

        assert transport.slave_path is None

    async def test_connection_error_handling(self, mocker):
        """Test error handling when PTY creation fails."""
        mocker.patch(
            "rayforge.machine.transport.serial_server.pty.openpty",
            side_effect=OSError("PTY creation failed"),
        )

        transport = SerialServerTransport()
        status_tracker = SignalTracker(transport.status_changed)

        with pytest.raises(OSError, match="PTY creation failed"):
            await transport.connect()

        assert not transport.is_connected
        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert TransportStatus.ERROR in statuses

    async def test_is_connected_reflects_state(self):
        """Test that is_connected properly reflects transport state."""
        transport = SerialServerTransport()
        assert not transport.is_connected

        await transport.connect()
        assert transport.is_connected

        await transport.disconnect()
        assert not transport.is_connected

    async def test_slave_path_is_valid_tty(self):
        """Test that slave_path points to a valid TTY device."""
        transport = SerialServerTransport()

        try:
            await transport.connect()
            slave_path = transport.slave_path
            assert slave_path is not None

            assert os.path.exists(slave_path)
            assert os.access(slave_path, os.R_OK | os.W_OK)
        finally:
            await transport.disconnect()

    async def test_send_to_connected_transport(self):
        """Test that send() works on connected transport."""
        transport = SerialServerTransport()

        try:
            await transport.connect()
            test_data = b"TESTDATA"
            await transport.send(test_data)
        finally:
            await transport.disconnect()

    async def test_multiple_connect_disconnect_cycles(self):
        """Test multiple connect/disconnect cycles."""
        transport = SerialServerTransport()

        for _ in range(3):
            await transport.connect()
            assert transport.is_connected
            await transport.disconnect()
            assert not transport.is_connected


@pytest.mark.asyncio
class TestSerialServerTransportIO:
    """Tests for SerialServerTransport I/O operations."""

    async def test_receive_from_slave(self):
        """Test receiving data written to the slave PTY."""
        transport = SerialServerTransport()
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()
            slave_path = transport.slave_path
            assert slave_path is not None

            response = b"PONG"

            slave_fd = os.open(slave_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(slave_fd, response)
            finally:
                os.close(slave_fd)

            for _ in range(20):
                if received_tracker.last_data == response:
                    break
                await asyncio.sleep(0.05)

            assert received_tracker.last_data == response
        finally:
            await transport.disconnect()

    async def test_send_to_slave(self):
        """Test that send() writes data to the PTY (readable from slave)."""
        transport = SerialServerTransport()

        try:
            await transport.connect()
            slave_path = transport.slave_path
            assert slave_path is not None

            test_data = b"HELLO"

            slave_fd = os.open(slave_path, os.O_RDWR | os.O_NONBLOCK)
            try:
                await transport.send(test_data)

                received = None
                for _ in range(20):
                    try:
                        received = os.read(slave_fd, 1024)
                        break
                    except BlockingIOError:
                        await asyncio.sleep(0.05)

                assert received == test_data
            finally:
                os.close(slave_fd)
        finally:
            await transport.disconnect()

    async def test_bidirectional_communication(self):
        """Test full bidirectional communication."""
        transport = SerialServerTransport()
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()
            slave_path = transport.slave_path
            assert slave_path is not None

            slave_fd = os.open(slave_path, os.O_RDWR | os.O_NONBLOCK)
            try:
                await transport.send(b"PING")

                received = None
                for _ in range(20):
                    try:
                        received = os.read(slave_fd, 1024)
                        break
                    except BlockingIOError:
                        await asyncio.sleep(0.05)
                assert received == b"PING"

                os.write(slave_fd, b"PONG")
                for _ in range(20):
                    if received_tracker.last_data == b"PONG":
                        break
                    await asyncio.sleep(0.05)
                assert received_tracker.last_data == b"PONG"
            finally:
                os.close(slave_fd)
        finally:
            await transport.disconnect()
