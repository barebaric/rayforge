import asyncio
import pytest
import pytest_asyncio
import asyncudp
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
    A simple, asynchronous UDP server for integration testing.
    It binds to a local port and allows verification of received data
    and sending of data back to the client.
    """

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self.port = port
        self.socket = None
        self.received_messages = []  # List of (data, addr) tuples
        self._running = False
        self._listen_task = None
        self.last_addr = None  # The address of the last sender

    async def start(self):
        """
        Starts the server and returns the host and port it's listening on.
        """
        self.socket = await asyncudp.create_socket(
            local_addr=(self.host, self.port)
        )
        addr = self.socket.getsockname()
        self.host = addr[0]
        self.port = addr[1]
        self._running = True
        self._listen_task = asyncio.create_task(self._listen_loop())
        return self.host, self.port

    async def stop(self):
        """Stops the server."""
        self._running = False
        if self.socket:
            self.socket.close()
        if self._listen_task:
            try:
                await self._listen_task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass

    async def _listen_loop(self):
        while self._running:
            if self.socket is None:
                break
            try:
                data, addr = await self.socket.recvfrom()
                self.received_messages.append((data, addr))
                self.last_addr = addr
            except asyncio.CancelledError:
                break
            except Exception:
                # Socket closed or other error
                break

    async def send_to_last_client(self, data: bytes):
        """Sends data back to the address that last sent us data."""
        if self.socket and self.last_addr:
            self.socket.sendto(data, self.last_addr)
        else:
            raise ConnectionError("No client has connected yet")


@pytest_asyncio.fixture
async def udp_server():
    """A pytest fixture that manages the lifecycle of the MockUdpServer."""
    server = MockUdpServer()
    await server.start()
    yield server
    await server.stop()


@pytest.mark.asyncio
class TestUdpTransportIntegration:
    """
    Tests the UdpTransport against a live, local UDP socket.
    """

    async def test_connect_disconnect_cycle(self, udp_server):
        """Test the connection and disconnection lifecycle and signals."""
        host, port = udp_server.host, udp_server.port
        transport = UdpTransport(host=host, port=port)
        status_tracker = SignalTracker(transport.status_changed)

        assert not transport.is_connected

        await transport.connect()
        assert transport.is_connected
        # UDP is connectionless, so the server won't know we exist yet,
        # but the transport should consider itself 'connected' (socket
        # created).

        await transport.disconnect()
        assert not transport.is_connected

        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [
            TransportStatus.CONNECTING,
            TransportStatus.CONNECTED,
            TransportStatus.DISCONNECTED,
        ]

    async def test_send_data(self, udp_server):
        """Test sending data from transport to server."""
        host, port = udp_server.host, udp_server.port
        transport = UdpTransport(host=host, port=port)

        try:
            await transport.connect()

            test_message = b"Hello UDP"
            await transport.send(test_message)

            # Wait for server to receive
            for _ in range(20):
                if udp_server.received_messages:
                    break
                await asyncio.sleep(0.05)

            assert len(udp_server.received_messages) == 1
            data, _ = udp_server.received_messages[0]
            assert data == test_message

        finally:
            await transport.disconnect()

    async def test_receive_data(self, udp_server):
        """
        Test receiving data from server to transport.
        Note: The transport must send something first so the server knows
        where to reply (NAT/Stateful logic simulation).
        """
        host, port = udp_server.host, udp_server.port
        transport = UdpTransport(host=host, port=port)
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()

            # 1. Transport sends "ping" so server knows our address
            await transport.send(b"ping")

            # Wait for server to register the ping
            for _ in range(20):
                if udp_server.last_addr:
                    break
                await asyncio.sleep(0.05)

            assert udp_server.last_addr is not None

            # 2. Server sends "pong" back
            response_message = b"pong"
            await udp_server.send_to_last_client(response_message)

            # 3. Transport should receive it
            for _ in range(20):
                if received_tracker.last_data == response_message:
                    break
                await asyncio.sleep(0.05)

            assert received_tracker.last_data == response_message

        finally:
            await transport.disconnect()

    async def test_round_trip_echo(self, udp_server):
        """Test a full round trip: Transport -> Server -> Transport."""
        host, port = udp_server.host, udp_server.port
        transport = UdpTransport(host=host, port=port)
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()

            msg_out = b"echo_request"
            msg_in = b"echo_reply"

            await transport.send(msg_out)

            # Manually act as the server logic
            for _ in range(20):
                if len(udp_server.received_messages) > 0:
                    break
                await asyncio.sleep(0.05)

            # Verify server got request
            assert udp_server.received_messages[0][0] == msg_out

            # Server replies
            await udp_server.send_to_last_client(msg_in)

            # Verify transport got reply
            for _ in range(20):
                if received_tracker.last_data == msg_in:
                    break
                await asyncio.sleep(0.05)

            assert received_tracker.last_data == msg_in

        finally:
            await transport.disconnect()

    async def test_send_without_connect(self, udp_server):
        """Test that sending without connecting raises ConnectionError."""
        transport = UdpTransport(host=udp_server.host, port=udp_server.port)

        with pytest.raises(ConnectionError, match="Not connected"):
            await transport.send(b"fail")

    async def test_connection_error_handling(self, mocker):
        """
        Test error handling when initializing with an invalid hostname.
        """
        # Patch socket.gethostbyname to simulate failure, as some environments
        # (like CI or custom DNS) might not fail on garbage hostnames.
        mocker.patch(
            "rayforge.machine.transport.udp.socket.gethostbyname",
            side_effect=OSError("Resolution failed"),
        )

        with pytest.raises(OSError, match="Resolution failed"):
            UdpTransport(host="invalid.hostname", port=1234)
