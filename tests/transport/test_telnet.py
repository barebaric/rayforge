import asyncio
import pytest
import pytest_asyncio
from rayforge.machine.transport.telnet import TelnetTransport
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


class MockTelnetServer:
    """
    A simple, asynchronous Telnet server for testing purposes.
    It listens on a local port, echoes back any data it receives,
    and stores all received data for inspection by the test.
    """

    def __init__(self, host="127.0.0.1", port=0):
        self.host = host
        self.port = port
        self.server = None
        self.clients = []
        self.received_data = bytearray()

    async def start(self):
        """
        Starts the server and returns the host and port it's listening on.
        """
        self.server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        addr = self.server.sockets[0].getsockname()
        self.host = addr[0]
        self.port = addr[1]
        return self.host, self.port

    async def stop(self):
        """Stops the server and closes all client connections."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Close all client writers
        for writer in list(self.clients):
            if not writer.is_closing():
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass
        self.clients.clear()

    async def send_to_clients(self, data: bytes):
        """Sends data from the server to all connected clients."""
        for writer in self.clients:
            writer.write(data)
            await writer.drain()

    async def _handle_client(self, reader, writer):
        """Callback for handling a new client connection."""
        self.clients.append(writer)
        try:
            while True:
                data = await reader.read(1024)
                if not data:
                    break  # Client disconnected
                self.received_data.extend(data)
                # Echo the data back
                writer.write(data)
                await writer.drain()
        except Exception:
            pass
        finally:
            if writer in self.clients:
                self.clients.remove(writer)
            writer.close()


@pytest_asyncio.fixture
async def telnet_server():
    """A pytest fixture that manages the lifecycle of the MockTelnetServer."""
    server = MockTelnetServer()
    await server.start()
    yield server
    await server.stop()


@pytest.mark.asyncio
class TestTelnetTransportIntegration:
    """
    Tests the TelnetTransport against a live, local mock server. This provides
    a higher level of confidence than just mocking asyncio.open_connection.
    """

    async def test_connect_disconnect_cycle(self, telnet_server):
        """Test the connection and disconnection lifecycle and signals."""
        host, port = telnet_server.host, telnet_server.port
        transport = TelnetTransport(host=host, port=port)
        status_tracker = SignalTracker(transport.status_changed)

        assert not transport.is_connected
        await transport.connect()
        assert transport.is_connected
        assert len(telnet_server.clients) == 1

        await transport.disconnect()
        assert not transport.is_connected
        # Allow a moment for the server to process the client disconnect
        await asyncio.sleep(0.1)
        assert len(telnet_server.clients) == 0

        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [
            TransportStatus.CONNECTING,
            TransportStatus.CONNECTED,
            TransportStatus.DISCONNECTED,
        ]

    async def test_connection_failure(self):
        """Test that connection failures are handled gracefully."""
        # Use a port that is almost certainly not in use
        transport = TelnetTransport(host="127.0.0.1", port=65530)
        status_tracker = SignalTracker(transport.status_changed)

        with pytest.raises(
            OSError
        ):  # ConnectionRefusedError is an OSError subclass
            await transport.connect()

        assert not transport.is_connected
        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [TransportStatus.CONNECTING, TransportStatus.ERROR]
        error_call = status_tracker.last_call
        assert error_call is not None
        assert "message" in error_call["kwargs"]
        # On Windows, the error message differs ([WinError 1225]), so we just
        # check content presence
        assert error_call["kwargs"]["message"]

    async def test_send_and_receive(self, telnet_server):
        """Test sending data to the server and receiving its echo back."""
        host, port = telnet_server.host, telnet_server.port
        transport = TelnetTransport(host=host, port=port)
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()
            assert transport.is_connected

            # Test sending data to the server
            test_message = b"G1 X10 F1000\n"
            await transport.send(test_message)

            # Wait for data with a timeout loop
            for _ in range(20):  # Wait up to 2 seconds
                if telnet_server.received_data == test_message:
                    break
                await asyncio.sleep(0.1)

            assert telnet_server.received_data == test_message

            # Wait for echo
            for _ in range(20):
                if received_tracker.last_data == test_message:
                    break
                await asyncio.sleep(0.1)

            assert received_tracker.last_data == test_message

        finally:
            await transport.disconnect()

    async def test_receive_from_server(self, telnet_server):
        """Test receiving unsolicited data from the server."""
        host, port = telnet_server.host, telnet_server.port
        transport = TelnetTransport(host=host, port=port)
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()

            server_message = b"ok\r\n"
            await telnet_server.send_to_clients(server_message)

            # Yield control to allow the transport to process the received data
            for _ in range(10):
                if received_tracker.last_data == server_message:
                    break
                await asyncio.sleep(0.1)

            assert len(received_tracker.calls) > 0
            assert received_tracker.last_data == server_message
        finally:
            await transport.disconnect()

    async def test_server_disconnects_unexpectedly(self, telnet_server):
        """Test how the transport handles an abrupt server-side disconnect."""
        host, port = telnet_server.host, telnet_server.port
        transport = TelnetTransport(host=host, port=port)
        status_tracker = SignalTracker(transport.status_changed)

        await transport.connect()
        assert transport.is_connected

        # Abruptly close the connection from the server side
        server_writer = telnet_server.clients[0]
        server_writer.close()
        await server_writer.wait_closed()

        # The transport's background task should detect the closed connection
        # and update its status. We wait for this to happen.
        for _ in range(20):
            if not transport.is_connected:
                break
            await asyncio.sleep(0.1)

        assert not transport.is_connected
        last_call = status_tracker.last_call
        assert last_call is not None
        final_status = last_call["kwargs"]["status"]
        assert final_status == TransportStatus.DISCONNECTED
