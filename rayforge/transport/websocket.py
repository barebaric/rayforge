import asyncio
import websockets
from typing import Optional
from websockets.exceptions import ConnectionClosed
from .transport import Transport, TransportStatus


class WebSocketTransport(Transport):
    """
    WebSocket transport with robust state management.
    """

    def __init__(self, uri: str, origin=None):
        super().__init__()
        self.uri = uri
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._origin = origin
        self._running = False
        self._reconnect_interval = 5
        self._lock = asyncio.Lock()
        self._receive_task: Optional[asyncio.Task] = None
        self._status = TransportStatus.DISCONNECTED

    @property
    def is_connected(self) -> bool:
        """Check if the transport's status is CONNECTED."""
        return self._status == TransportStatus.CONNECTED

    def _set_status(
        self, status: TransportStatus, message: Optional[str] = None
    ) -> None:
        """
        Internal helper to set status and send signal, avoiding duplicates.
        """
        if self._status == status:
            return
        self._status = status
        self.status_changed.send(self, status=status, message=message)

    async def connect(self) -> None:
        """
        Establish and maintain a connection, reconnecting on failure.
        """
        async with self._lock:
            if self._running:
                return
            self._running = True

        while self._running:
            try:
                self._set_status(TransportStatus.CONNECTING)
                self._websocket = await websockets.connect(
                    self.uri,
                    origin=self._origin,
                    additional_headers=(
                        ("Connection", "Upgrade"),
                        ("Upgrade", "websocket"),
                    ),
                )
                self._set_status(TransportStatus.CONNECTED)
                self._receive_task = asyncio.create_task(self._receive_loop())
                await self._receive_task

            except (asyncio.CancelledError, ConnectionClosed):
                # This is an expected part of a clean shutdown or reconnect
                # cycle.
                pass
            except Exception as e:
                self._set_status(TransportStatus.ERROR, message=str(e))
            finally:
                # Always clean up the connection before the next step.
                await self._safe_close()
                # If we are still supposed to be running, wait and reconnect.
                if self._running:
                    self._set_status(TransportStatus.SLEEPING)
                    await asyncio.sleep(self._reconnect_interval)

        # When the loop is fully stopped, we are disconnected.
        self._set_status(TransportStatus.DISCONNECTED)

    async def disconnect(self) -> None:
        """
        Terminate the connection immediately and permanently.
        """
        self._set_status(TransportStatus.CLOSING)
        async with self._lock:
            if not self._running:
                return
            self._running = False
            if self._receive_task:
                self._receive_task.cancel()
            await self._safe_close()
        self._set_status(TransportStatus.DISCONNECTED)

    async def send(self, data: bytes) -> None:
        """
        Send data through the active connection.
        """
        if not self.is_connected or self._websocket is None:
            raise ConnectionError("Not connected")
        try:
            await self._websocket.send(data)
        except ConnectionClosed:
            # The main `connect` loop will detect the closure via the
            # `_receive_loop` and handle the reconnect automatically.
            # We just need to signal that this specific send operation failed.
            raise ConnectionError("Connection lost while sending")

    async def _receive_loop(self) -> None:
        """
        Receive messages and handle connection state internally.
        """
        if self._websocket is None:
            return
        try:
            async for message in self._websocket:
                if isinstance(message, bytes):
                    self.received.send(self, data=message)
        except ConnectionClosed:
            pass  # The outer connect() loop will handle this.
        except Exception as e:
            self._set_status(TransportStatus.ERROR, message=str(e))

    async def _safe_close(self) -> None:
        """
        Safely close connection and reset the internal websocket object.
        """
        if self._websocket is not None:
            try:
                await self._websocket.close()
            except Exception:
                # Ignore errors on close, as we are tearing down the
                # connection.
                pass
            finally:
                self._websocket = None
