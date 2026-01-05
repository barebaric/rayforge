import asyncio
import logging
from typing import Optional, Tuple, cast
from .transport import Transport, TransportStatus

logger = logging.getLogger(__name__)


class _UdpProtocol(asyncio.DatagramProtocol):
    """
    Internal Protocol class required by asyncio.create_datagram_endpoint.
    Bridges low-level protocol events to the UdpTransport high-level logic.
    """

    def __init__(self, transport_wrapper: "UdpTransport"):
        self.transport_wrapper = transport_wrapper

    def connection_made(self, transport: asyncio.BaseTransport) -> None:
        # We handle status updates in the UdpTransport.connect method
        # to ensure the transport reference is set before the signal fires.
        pass

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        if self.transport_wrapper:
            self.transport_wrapper._handle_data(data)

    def error_received(self, exc: Exception) -> None:
        logger.error(f"UDP Error received: {exc}")
        if self.transport_wrapper:
            self.transport_wrapper._handle_error(exc)

    def connection_lost(self, exc: Optional[Exception]) -> None:
        # This is called when the socket is closed or no longer usable.
        if self.transport_wrapper:
            self.transport_wrapper._handle_close(exc)


class UdpTransport(Transport):
    """
    UDP Transport implementation.

    Since UDP is connectionless, 'Connected' state implies the socket is bound
    and configured to send to a specific remote address.
    """

    def __init__(self, host: str, port: int):
        """
        Initialize UDP transport.

        Args:
            host: Remote hostname or IP address.
            port: Remote port.
        """
        super().__init__()
        self.host = host
        self.port = port
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._protocol: Optional[_UdpProtocol] = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        """Check if the socket is created and active."""
        return self._transport is not None and not self._transport.is_closing()

    async def connect(self) -> None:
        if self.is_connected:
            return

        self._running = True
        self.status_changed.send(self, status=TransportStatus.CONNECTING)
        logger.info(f"Setting up UDP endpoint for {self.host}:{self.port}...")

        loop = asyncio.get_running_loop()
        try:
            # remote_addr acts as a default destination for sendto(), making
            # the socket "connected" in OS terms (filtering incoming packets).
            transport, protocol = await loop.create_datagram_endpoint(
                lambda: _UdpProtocol(self), remote_addr=(self.host, self.port)
            )

            self._transport = cast(asyncio.DatagramTransport, transport)
            self._protocol = cast(_UdpProtocol, protocol)

            logger.info(f"UDP socket bound to {self.host}:{self.port}")
            self.status_changed.send(self, status=TransportStatus.CONNECTED)

        except Exception as e:
            logger.error(f"Failed to create UDP endpoint: {e}")
            self.status_changed.send(
                self, status=TransportStatus.ERROR, message=str(e)
            )
            # Ensure cleanup if partially created
            await self.disconnect()
            raise

    async def disconnect(self) -> None:
        """
        Close the UDP socket.
        """
        logger.info("Closing UDP transport...")
        self._running = False

        if self._transport:
            self.status_changed.send(self, status=TransportStatus.CLOSING)
            self._transport.close()
            # We do not await transport.wait_closed() here because
            # asyncio.DatagramTransport does not always expose it consistently
            # across versions/implementations, and close() is immediate.

        self._transport = None
        self._protocol = None

        # Signal explicitly here, though connection_lost also triggers it.
        # This ensures the state is cleared even if connection_lost doesn't
        # fire.
        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)

    async def send(self, data: bytes) -> None:
        if not self.is_connected or self._transport is None:
            raise ConnectionError("UDP transport not connected")

        try:
            self._transport.sendto(data)
        except Exception as e:
            # Basic send errors (e.g. buffer full, network down immediate
            # check)
            raise ConnectionError(f"Failed to send UDP packet: {e}") from e

    def _handle_data(self, data: bytes) -> None:
        """Internal callback from protocol."""
        if self._running:
            self.received.send(self, data=data)

    def _handle_error(self, exc: Exception) -> None:
        """Internal callback for protocol errors."""
        self.status_changed.send(
            self, status=TransportStatus.ERROR, message=str(exc)
        )

    def _handle_close(self, exc: Optional[Exception]) -> None:
        """Internal callback for connection loss."""
        if self._running:
            self._running = False
            msg = str(exc) if exc else None
            if exc:
                self.status_changed.send(
                    self, status=TransportStatus.ERROR, message=msg
                )
            self.status_changed.send(self, status=TransportStatus.DISCONNECTED)
