import asyncio
import logging
from typing import Optional
from .transport import Transport, TransportStatus


logger = logging.getLogger(__name__)


class TelnetTransport(Transport):
    def __init__(self, host: str, port: int):
        super().__init__()
        self.host = host
        self.port = port
        self.reader: Optional[asyncio.StreamReader] = None
        self.writer: Optional[asyncio.StreamWriter] = None
        self._running = False
        self._reconnect_interval = 5
        self._connection_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if the transport is actively connected."""
        return self.writer is not None and not self.writer.is_closing()

    async def connect(self) -> None:
        if self.is_connected:
            return

        self._running = True
        self.status_changed.send(self, status=TransportStatus.CONNECTING)
        logger.info(
            f"Connecting to telnet server at {self.host}:{self.port}..."
        )
        try:
            self.reader, self.writer = await asyncio.open_connection(
                self.host, self.port
            )
            self.status_changed.send(self, status=TransportStatus.CONNECTED)
            logger.info(f"Successfully connected to {self.host}:{self.port}.")
            # Connection is successful, start the management task
            self._connection_task = asyncio.create_task(
                self._manage_connection()
            )
        except Exception as e:
            # Failed to connect, report error and re-raise so caller knows.
            logger.error(f"Failed to connect to {self.host}:{self.port}: {e}")
            self.status_changed.send(
                self, status=TransportStatus.ERROR, message=str(e)
            )
            raise

    async def _manage_connection(self) -> None:
        """
        Manages an active connection: receives data and handles disconnects.
        """
        try:
            await self._receive_loop()
        except Exception as e:
            self.status_changed.send(
                self, status=TransportStatus.ERROR, message=str(e)
            )
        finally:
            # Connection was lost or an error occurred.
            if self.writer:
                self.writer.close()
            self.writer = None
            self.reader = None
            self.status_changed.send(self, status=TransportStatus.DISCONNECTED)

    async def disconnect(self) -> None:
        logger.info(
            f"Disconnecting from telnet server at {self.host}:{self.port}..."
        )
        self._running = False
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass  # Expected
        if self.writer:
            self.writer.close()
            try:
                await self.writer.wait_closed()
            except ConnectionResetError:
                pass  # The other end might have already closed it.
        self.writer = None
        self.reader = None
        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)
        logger.info(f"Disconnected from {self.host}:{self.port}.")

    async def send(self, data: bytes) -> None:
        if not self.is_connected or self.writer is None:
            raise ConnectionError("Not connected")
        self.writer.write(data)
        await self.writer.drain()

    async def _receive_loop(self) -> None:
        while self.reader:
            try:
                data = await self.reader.read(1024)
                if data:
                    self.received.send(self, data=data)
                else:
                    logger.info(
                        f"Telnet connection to {self.host}:{self.port} "
                        "closed by peer."
                    )
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.status_changed.send(
                    self, status=TransportStatus.ERROR, message=str(e)
                )
                break
