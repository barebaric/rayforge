import asyncio
from typing import Optional
from .transport import Transport, TransportStatus


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

    async def connect(self) -> None:
        self._running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        while self._running:
            try:
                self.status_changed.send(
                    self, status=TransportStatus.CONNECTING
                )
                self.reader, self.writer = await asyncio.open_connection(
                    self.host, self.port
                )
                self.status_changed.send(
                    self, status=TransportStatus.CONNECTED
                )
                await self._receive_loop()
            except Exception as e:
                self.status_changed.send(
                    self, status=TransportStatus.ERROR, message=str(e)
                )
            finally:
                if self.writer:
                    self.writer.close()
                    await self.writer.wait_closed()
                self.status_changed.send(
                    self, status=TransportStatus.DISCONNECTED
                )

            if self._running:
                self.status_changed.send(self, status=TransportStatus.SLEEPING)
                await asyncio.sleep(self._reconnect_interval)
        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)

    async def disconnect(self) -> None:
        self._running = False
        if self._connection_task:
            self._connection_task.cancel()

    async def send(self, data: bytes) -> None:
        if not self.writer:
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
                    break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.status_changed.send(
                    self, status=TransportStatus.ERROR, message=str(e)
                )
                break
