import asyncio
import pytest
import pytest_asyncio
from rayforge.machine.transport.udp import UdpTransport
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


class MockUdpServer:
    """
    A simple UDP Echo Server for testing.
    """

    class ServerProtocol(asyncio.DatagramProtocol):
        def __init__(self, server_instance):
            self.server = server_instance

        def connection_made(self, transport):
            self.server.transport = transport

        def datagram_received(self, data, addr):
            self.server.received_data.append((data, addr))
            # Echo back
            self.server.transport.sendto(data, addr)

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self.port = port
        self.transport = None
        self.received_data = []

    async def start(self):
        loop = asyncio.get_running_loop()
        transport, protocol = await loop.create_datagram_endpoint(
            lambda: self.ServerProtocol(self),
            local_addr=(self.host, self.port),
        )
        self.transport = transport
        # Update port if 0 was used (dynamic allocation)
        self.port = transport.get_extra_info("sockname")[1]
        return self.host, self.port

    def stop(self):
        if self.transport:
            self.transport.close()


@pytest_asyncio.fixture
async def udp_server():
    """A pytest fixture that manages the lifecycle of the MockUdpServer."""
    server = MockUdpServer()
    await server.start()
    yield server
    server.stop()


@pytest.mark.asyncio
class TestUdpTransportIntegration:
    """
    Tests the UdpTransport against a real local network socket.
    """

    async def test_connect_disconnect_cycle(self, udp_server):
        """Test the connection state transitions."""
        host, port = udp_server.host, udp_server.port
        transport = UdpTransport(host=host, port=port)
        status_tracker = SignalTracker(transport.status_changed)

        assert not transport.is_connected

        await transport.connect()
        assert transport.is_connected

        # Check statuses emitted during connect
        # Expect: CONNECTING -> CONNECTED
        statuses = [c["kwargs"]["status"] for c in status_tracker.calls]
        assert TransportStatus.CONNECTING in statuses
        assert TransportStatus.CONNECTED in statuses

        status_tracker.clear()

        await transport.disconnect()
        assert not transport.is_connected

        # Check statuses emitted during disconnect
        # Expect: CLOSING -> DISCONNECTED
        statuses = [c["kwargs"]["status"] for c in status_tracker.calls]
        assert TransportStatus.DISCONNECTED in statuses

    async def test_send_and_receive_echo(self, udp_server):
        """Test sending data and receiving the echo via signals."""
        host, port = udp_server.host, udp_server.port
        transport = UdpTransport(host=host, port=port)
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()

            message = b"UDP Ping"
            await transport.send(message)

            # UDP is fast locally, but we allow a small window
            for _ in range(10):
                if received_tracker.last_data == message:
                    break
                await asyncio.sleep(0.05)

            # Verify Server received it
            assert len(udp_server.received_data) > 0
            assert udp_server.received_data[0][0] == message

            # Verify Transport received the echo
            assert received_tracker.last_data == message

        finally:
            await transport.disconnect()

    async def test_send_not_connected(self):
        """Test that sending without connecting raises an error."""
        transport = UdpTransport(host="127.0.0.1", port=12345)
        with pytest.raises(
            ConnectionError, match="UDP transport not connected"
        ):
            await transport.send(b"test")

    async def test_invalid_address(self):
        """Test handling of invalid hostname resolution."""
        # Using a TLD that definitely doesn't exist
        transport = UdpTransport(
            host="invalid.hostname.test.local", port=12345
        )
        status_tracker = SignalTracker(transport.status_changed)

        with pytest.raises(OSError):
            await transport.connect()

        assert not transport.is_connected

        # Check that ERROR status was emitted
        statuses = [c["kwargs"]["status"] for c in status_tracker.calls]
        assert TransportStatus.ERROR in statuses

    async def test_concurrent_sends(self, udp_server):
        """Test sending multiple packets rapidly."""
        host, port = udp_server.host, udp_server.port
        transport = UdpTransport(host=host, port=port)
        received_tracker = SignalTracker(transport.received)

        await transport.connect()
        try:
            count = 5
            for i in range(count):
                await transport.send(f"msg{i}".encode())

            # Wait for all echoes
            max_wait = 20  # 1 second total
            while len(received_tracker.calls) < count and max_wait > 0:
                await asyncio.sleep(0.05)
                max_wait -= 1

            assert len(received_tracker.calls) == count
            # Order isn't guaranteed in UDP generally, but on localhost it
            # usually holds.
            # We just verify we got 5 responses.
        finally:
            await transport.disconnect()
