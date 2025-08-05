import asyncio
from typing import Optional
import aiohttp
from .transport import Transport, TransportStatus


class HttpTransport(Transport):
    """
    HTTP transport using persistent connection with auto-reconnect.
    """

    def __init__(self, base_url: str, receive_interval: float):
        """
        Initialize HTTP transport.

        Args:
            base_url: Server endpoint URL (schema://host:port)
        """
        super().__init__()
        self.base_url = base_url
        self._running = False
        self._reconnect_interval = 5
        self._receive_interval = receive_interval
        self._connection_task: Optional[asyncio.Task] = None

    async def connect(self) -> None:
        """
        Maintain persistent connection with reconnect logic.
        """
        self._running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        while self._running:
            try:
                self.status_changed.send(
                    self,
                    status=TransportStatus.CONNECTING
                )
                async with aiohttp.ClientSession() as session:
                    await self._receive_loop(session)
            except Exception as e:
                self.status_changed.send(
                    self,
                    status=TransportStatus.ERROR,
                    message=str(e)
                )
            finally:
                self.status_changed.send(
                    self,
                    status=TransportStatus.DISCONNECTED
                )

            if self._running:
                self.status_changed.send(
                    self,
                    status=TransportStatus.SLEEPING
                )
                await asyncio.sleep(self._reconnect_interval)
        self.status_changed.send(
            self,
            status=TransportStatus.DISCONNECTED
        )

    async def disconnect(self) -> None:
        """
        Terminate connection and cancel background tasks.
        """
        self._running = False
        if self._connection_task:
            self._connection_task.cancel()

    async def send(self, data: bytes) -> None:
        """
        Send data to HTTP endpoint via POST request.
        """
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}",
                data=data,
                timeout=aiohttp.ClientTimeout(total=5),
            ) as response:
                if response.status != 200:
                    err = f"Send failed: {await response.text()}"
                    self.status_changed.send(self, message=err)
                    raise IOError(err)

    async def _receive_loop(self, session) -> None:
        """
        Listen for server-sent events from streaming endpoint.
        """
        while self._running:
            async with session.get(
                f"{self.base_url}",
                timeout=aiohttp.ClientTimeout(total=30),
            ) as response:
                if response.status == 200:
                    data = await response.read()
                    if data:
                        self.received.send(self, data=data)

            if self._running and self._receive_interval:
                await asyncio.sleep(self._receive_interval)
