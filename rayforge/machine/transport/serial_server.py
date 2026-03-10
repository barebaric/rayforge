import asyncio
import fcntl
import logging
import os
import pty
import termios
from typing import Optional

from .transport import Transport, TransportStatus

logger = logging.getLogger(__name__)


class SerialServerTransport(Transport):
    """
    Serial server transport that creates a PTY pair.

    Creates a pseudo-terminal pair where the master end is used by this
    transport and the slave path is exposed for clients to connect to.
    Useful for simulating serial devices without hardware.
    """

    def __init__(self, baudrate: int = 115200):
        super().__init__()
        self.baudrate = baudrate
        self._master_fd: Optional[int] = None
        self._slave_path: Optional[str] = None
        self._running = False

    @property
    def slave_path(self) -> Optional[str]:
        """Returns the path clients should connect to."""
        return self._slave_path

    @property
    def is_connected(self) -> bool:
        return self._master_fd is not None and self._running

    async def connect(self) -> None:
        if self.is_connected:
            return

        self._running = True
        self.status_changed.send(self, status=TransportStatus.CONNECTING)
        logger.info("Creating PTY for serial server...")

        try:
            master_fd, slave_fd = pty.openpty()
            self._master_fd = master_fd
            self._slave_path = os.ttyname(slave_fd)
            os.close(slave_fd)  # Close slave fd - clients will open their own

            attrs = termios.tcgetattr(master_fd)
            attrs[4] = termios.B115200
            attrs[5] = termios.B115200
            attrs[0] &= ~(
                termios.IGNBRK
                | termios.BRKINT
                | termios.PARMRK
                | termios.ISTRIP
                | termios.INLCR
                | termios.IGNCR
                | termios.ICRNL
                | termios.IXON
            )
            attrs[1] &= ~termios.OPOST
            attrs[2] &= ~(termios.CSIZE | termios.PARENB)
            attrs[2] |= termios.CS8
            attrs[3] &= ~(
                termios.ECHO
                | termios.ECHONL
                | termios.ICANON
                | termios.ISIG
                | termios.IEXTEN
            )
            termios.tcsetattr(master_fd, termios.TCSANOW, attrs)

            flags = fcntl.fcntl(master_fd, fcntl.F_GETFL)
            fcntl.fcntl(master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

            loop = asyncio.get_event_loop()
            loop.add_reader(master_fd, self._on_readable)

            self.status_changed.send(self, status=TransportStatus.CONNECTED)
            logger.info(f"Serial server PTY created: {self._slave_path}")
        except Exception as e:
            logger.error(f"Failed to create PTY: {e}")
            self.status_changed.send(
                self, status=TransportStatus.ERROR, message=str(e)
            )
            raise

    async def disconnect(self) -> None:
        logger.info("Stopping serial server...")
        self._running = False

        if self._master_fd is not None:
            loop = asyncio.get_event_loop()
            loop.remove_reader(self._master_fd)

            try:
                os.close(self._master_fd)
            except OSError:
                pass
            self._master_fd = None

        self._slave_path = None
        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)
        logger.info("Serial server stopped")

    async def send(self, data: bytes) -> None:
        if self._master_fd is None:
            raise ConnectionError("Serial server not started")
        os.write(self._master_fd, data)

    async def purge(self) -> None:
        pass

    def _on_readable(self) -> None:
        """Called when the master FD is readable (data from slave)."""
        if not self._running or self._master_fd is None:
            return

        try:
            data = os.read(self._master_fd, 4096)
            if data:
                logger.debug(f"Received data: {data!r}")
                self.received.send(self, data=data)
            else:
                logger.info("PTY closed by peer")
                asyncio.create_task(self._handle_peer_close())
        except BlockingIOError:
            pass
        except OSError as e:
            logger.error(f"Error reading from PTY: {e}")
            self.status_changed.send(
                self, status=TransportStatus.ERROR, message=str(e)
            )
            asyncio.create_task(self._handle_error())

    async def _handle_peer_close(self) -> None:
        """Handle PTY peer close asynchronously."""
        self._running = False
        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)

    async def _handle_error(self) -> None:
        """Handle read error asynchronously."""
        self._running = False
