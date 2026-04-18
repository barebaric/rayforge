import asyncio
import pytest
import pytest_asyncio

from rayforge.machine.transport.udp_server import (
    UdpServerTransport,
    UdpServerProtocol,
)
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

    @property
    def last_addr(self):
        """Returns the addr from the last 'received' signal call."""
        if not self.calls:
            return None
        return self.calls[-1]["kwargs"].get("addr")


class MockUdpClient:
    """A simple UDP client for testing UdpServerTransport."""

    def __init__(self):
        self.transport = None
        self._addr = None

    async def start(self):
        loop = asyncio.get_event_loop()
        self.transport, _ = await loop.create_datagram_endpoint(
            asyncio.DatagramProtocol,
            local_addr=("127.0.0.1", 0),
        )
        self._addr = self.transport.get_extra_info("sockname")
        return self._addr

    async def stop(self):
        if self.transport:
            self.transport.close()

    @property
    def addr(self):
        return self._addr

    async def send_to(self, data: bytes, addr: tuple):
        if self.transport:
            self.transport.sendto(data, addr)


@pytest_asyncio.fixture
async def udp_client():
    """A pytest fixture that manages the lifecycle of the MockUdpClient."""
    client = MockUdpClient()
    addr = await client.start()
    yield client, addr
    await client.stop()


@pytest_asyncio.fixture
async def udp_server():
    """A pytest fixture that manages the lifecycle of UdpServerTransport."""
    server = UdpServerTransport(host="127.0.0.1", port=0)
    await server.connect()
    assert server._transport is not None
    sockname = server._transport.get_extra_info("sockname")
    server.port = sockname[1]
    yield server
    await server.disconnect()


@pytest.mark.asyncio
class TestUdpServerProtocol:
    """Tests for the UdpServerProtocol class."""

    async def test_datagram_received(self, udp_server):
        """Test that datagram_received triggers the transport callback."""
        received_tracker = SignalTracker(udp_server.received)

        protocol = UdpServerProtocol(udp_server)
        test_data = b"test data"
        test_addr = ("127.0.0.1", 12345)

        protocol.datagram_received(test_data, test_addr)

        assert len(received_tracker.calls) == 1
        assert received_tracker.last_data == test_data
        assert received_tracker.last_addr == test_addr

    async def test_error_received(self, udp_server):
        """Test that error_received sends error status signal."""
        status_tracker = SignalTracker(udp_server.status_changed)

        protocol = UdpServerProtocol(udp_server)
        test_error = Exception("Test error")

        protocol.error_received(test_error)

        assert len(status_tracker.calls) == 1
        assert (
            status_tracker.calls[0]["kwargs"]["status"]
            == TransportStatus.ERROR
        )
        assert "Test error" in status_tracker.calls[0]["kwargs"]["message"]


@pytest.mark.asyncio
class TestUdpServerTransport:
    """Tests for UdpServerTransport."""

    async def test_connect_creates_socket(self):
        """Test that connect() creates a UDP socket."""
        transport = UdpServerTransport(host="127.0.0.1", port=0)

        try:
            await transport.connect()
            assert transport.is_connected
        finally:
            await transport.disconnect()

    async def test_disconnect_closes_socket(self):
        """Test that disconnect() closes the socket."""
        transport = UdpServerTransport(host="127.0.0.1", port=0)

        await transport.connect()
        assert transport.is_connected

        await transport.disconnect()
        assert not transport.is_connected

    async def test_connect_disconnect_signals(self):
        """Test status signals during connect/disconnect cycle."""
        transport = UdpServerTransport(host="127.0.0.1", port=0)
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
        transport = UdpServerTransport(host="127.0.0.1", port=0)

        try:
            await transport.connect()
            assert transport.is_connected

            await transport.connect()
            assert transport.is_connected
        finally:
            await transport.disconnect()

    async def test_disconnect_when_not_connected(self):
        """Test that disconnecting when not connected is safe."""
        transport = UdpServerTransport()
        assert not transport.is_connected

        await transport.disconnect()
        assert not transport.is_connected

    async def test_send_raises_not_implemented(self, udp_server):
        """Test that send() raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="Use send_to"):
            await udp_server.send(b"test")

    async def test_send_to_without_connect(self):
        """Test that send_to without connecting raises ConnectionError."""
        transport = UdpServerTransport()

        with pytest.raises(ConnectionError, match="not started"):
            await transport.send_to(b"test", ("127.0.0.1", 12345))

    async def test_receive_from_client(self, udp_server, udp_client):
        """Test receiving data from a UDP client."""
        client, client_addr = udp_client
        received_tracker = SignalTracker(udp_server.received)

        test_data = b"Hello Server"
        server_addr = ("127.0.0.1", udp_server.port)

        await client.send_to(test_data, server_addr)

        for _ in range(20):
            if received_tracker.last_data == test_data:
                break
            await asyncio.sleep(0.05)

        assert received_tracker.last_data == test_data
        assert received_tracker.last_addr == client_addr

    async def test_send_to_client(self, udp_server, udp_client):
        """Test sending data to a specific client address."""
        client, client_addr = udp_client

        test_data = b"Hello Client"
        await udp_server.send_to(test_data, client_addr)

    async def test_bidirectional_communication(self, udp_server, udp_client):
        """Test full bidirectional communication."""
        client, client_addr = udp_client
        received_tracker = SignalTracker(udp_server.received)

        server_addr = ("127.0.0.1", udp_server.port)

        request = b"REQUEST"
        await client.send_to(request, server_addr)

        for _ in range(20):
            if received_tracker.last_data == request:
                break
            await asyncio.sleep(0.05)

        assert received_tracker.last_data == request
        assert received_tracker.last_addr == client_addr

        response = b"RESPONSE"
        await udp_server.send_to(response, client_addr)

    async def test_multiple_clients(self, udp_server):
        """Test handling multiple clients."""
        received_tracker = SignalTracker(udp_server.received)
        server_addr = ("127.0.0.1", udp_server.port)

        clients = []
        for i in range(3):
            client = MockUdpClient()
            await client.start()
            clients.append(client)

        try:
            for i, client in enumerate(clients):
                msg = f"Client {i}".encode()
                await client.send_to(msg, server_addr)

            for _ in range(20):
                if len(received_tracker.calls) >= 3:
                    break
                await asyncio.sleep(0.05)

            assert len(received_tracker.calls) == 3

            for i, client in enumerate(clients):
                response = f"Response {i}".encode()
                client_addr = received_tracker.calls[i]["kwargs"]["addr"]
                await udp_server.send_to(response, client_addr)
        finally:
            for client in clients:
                await client.stop()

    async def test_purge_is_noop(self, udp_server):
        """Test that purge() is a no-op for UDP server."""
        await udp_server.purge()

    async def test_purge_on_disconnected_transport(self):
        """Test that purge on disconnected transport is safe."""
        transport = UdpServerTransport()
        assert not transport.is_connected

        await transport.purge()

    async def test_host_and_port_properties(self):
        """Test that host and port are stored correctly."""
        transport = UdpServerTransport(host="0.0.0.0", port=50200)
        assert transport.host == "0.0.0.0"
        assert transport.port == 50200

    async def test_connection_error_handling(self, mocker):
        """Test error handling when binding fails."""

        async def raise_error(*args, **kwargs):
            raise OSError("Address not available")

        mock_loop = mocker.Mock()
        mock_loop.create_datagram_endpoint = raise_error
        mocker.patch(
            "rayforge.machine.transport.udp_server.asyncio.get_event_loop",
            return_value=mock_loop,
        )

        transport = UdpServerTransport(host="127.0.0.1", port=12345)
        status_tracker = SignalTracker(transport.status_changed)

        with pytest.raises(OSError, match="Address not available"):
            await transport.connect()

        assert not transport.is_connected
        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert TransportStatus.ERROR in statuses

    async def test_large_datagram(self, udp_server, udp_client):
        """Test sending and receiving larger datagrams."""
        client, client_addr = udp_client
        received_tracker = SignalTracker(udp_server.received)
        server_addr = ("127.0.0.1", udp_server.port)

        large_data = b"X" * 1024
        await client.send_to(large_data, server_addr)

        for _ in range(20):
            if received_tracker.last_data == large_data:
                break
            await asyncio.sleep(0.05)

        assert received_tracker.last_data == large_data

        large_response = b"Y" * 1024
        await udp_server.send_to(large_response, client_addr)

    async def test_empty_datagram(self, udp_server, udp_client):
        """Test receiving an empty datagram."""
        client, client_addr = udp_client
        received_tracker = SignalTracker(udp_server.received)
        server_addr = ("127.0.0.1", udp_server.port)

        await client.send_to(b"", server_addr)

        for _ in range(20):
            if len(received_tracker.calls) > 0:
                break
            await asyncio.sleep(0.05)

        assert received_tracker.last_data == b""
