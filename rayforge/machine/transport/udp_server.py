import asyncio
import logging
from typing import Optional, Tuple

from .transport import Transport, TransportStatus

logger = logging.getLogger(__name__)


class UdpServerProtocol(asyncio.DatagramProtocol):
    """Protocol handler for UDP server."""

    def __init__(self, transport: "UdpServerTransport"):
        self.transport = transport

    def datagram_received(self, data: bytes, addr: Tuple[str, int]) -> None:
        self.transport._on_datagram_received(data, addr)

    def error_received(self, exc: Exception) -> None:
        logger.error(f"UDP server error: {exc}")
        self.transport.status_changed.send(
            self.transport, status=TransportStatus.ERROR, message=str(exc)
        )


class UdpServerTransport(Transport):
    """
    UDP server transport that listens for incoming packets.

    Unlike UdpTransport (client), this binds to a local address
    and responds to any client that sends data.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 50200):
        super().__init__()
        self.host = host
        self.port = port
        self._transport: Optional[asyncio.DatagramTransport] = None
        self._running = False

    @property
    def is_connected(self) -> bool:
        return self._transport is not None

    async def connect(self) -> None:
        if self.is_connected:
            return

        self._running = True
        self.status_changed.send(self, status=TransportStatus.CONNECTING)
        logger.info(f"Starting UDP server on {self.host}:{self.port}...")

        try:
            loop = asyncio.get_event_loop()

            transport, _ = await loop.create_datagram_endpoint(
                lambda: UdpServerProtocol(self),
                local_addr=(self.host, self.port),
            )
            self._transport = transport

            sock = transport.get_extra_info("socket")
            if sock:
                self.port = sock.getsockname()[1]

            self.status_changed.send(self, status=TransportStatus.CONNECTED)
            logger.info(f"UDP server listening on {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"Failed to start UDP server: {e}")
            self.status_changed.send(
                self, status=TransportStatus.ERROR, message=str(e)
            )
            raise

    async def disconnect(self) -> None:
        logger.info(f"Stopping UDP server on {self.host}:{self.port}...")
        self._running = False

        if self._transport:
            self._transport.close()
            self._transport = None

        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)
        logger.info("UDP server stopped")

    async def send(self, data: bytes) -> None:
        raise NotImplementedError(
            "Use send_to(data, addr) for UDP server transport"
        )

    async def send_to(self, data: bytes, addr: Tuple[str, int]) -> None:
        if not self._transport:
            raise ConnectionError("UDP server not started")
        self._transport.sendto(data, addr)

    async def purge(self) -> None:
        pass

    def _on_datagram_received(
        self, data: bytes, addr: Tuple[str, int]
    ) -> None:
        self.received.send(self, data=data, addr=addr)
