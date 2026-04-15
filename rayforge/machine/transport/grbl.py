import asyncio
import threading
import logging
from typing import Optional, NamedTuple

from ...shared.tasker import task_mgr
from .serial import SerialTransport

logger = logging.getLogger(__name__)

GRBL_RX_BUFFER_SIZE = 128


class PendingCommand(NamedTuple):
    length: int
    op_index: Optional[int]


class GrblSerialTransport:
    """
    Wraps a SerialTransport with GRBL's character-counting flow control
    protocol.

    GRBL devices have a 128-byte serial receive buffer. This transport
    tracks how many bytes are outstanding and provides backpressure so
    callers never overflow it. Thread-safe: the buffer count is mutated
    from both async tasks and the serial receive callback (which may run
    in a separate thread).
    """

    def __init__(self, transport: SerialTransport):
        self._transport = transport
        self._rx_buffer_count = 0
        self._lock = threading.Lock()
        self._pending: asyncio.Queue[PendingCommand] = asyncio.Queue()
        self._space_available = asyncio.Event()
        self._space_available.set()

    @property
    def is_connected(self) -> bool:
        return self._transport.is_connected

    @property
    def port(self) -> str:
        return self._transport.port

    async def connect(self) -> None:
        await self._transport.connect()

    async def disconnect(self) -> None:
        await self._transport.disconnect()

    @property
    def received(self):
        return self._transport.received

    @property
    def status_changed(self):
        return self._transport.status_changed

    # --- Flow-controlled sending ---

    async def send_gcode(
        self, data: bytes, op_index: Optional[int] = None
    ) -> int:
        """
        Send G-code with buffer accounting.  Does NOT wait for buffer
        space; the caller should call wait_for_space() first.
        Returns the buffer fill level after sending.
        """
        command_len = len(data)
        self._pending.put_nowait(PendingCommand(command_len, op_index))
        count = self._add(command_len)
        await self._transport.send(data)
        return count

    async def send_poll(self, data: bytes) -> int:
        """
        Send a status poll without buffer accounting.
        Real-time commands (?) bypass the GRBL RX buffer.
        """
        # From the GRBL docs: "Like all real-time commands, the '?'
        # character is intercepted and never enters the serial buffer"
        # https://github.com/gnea/grbl/blob/master/doc/markdown/interface.md
        await self._transport.send(data)
        return self._get()

    async def send_command(self, data: bytes) -> int:
        """
        Send an interactive command ($$, $G, etc.) with buffer
        accounting.  Waits for buffer space if needed (up to 1 s),
        then sends regardless.  Returns the buffer fill level after
        sending.
        """
        command_len = len(data)
        while self.needs_space(command_len):
            try:
                await asyncio.wait_for(self.wait_for_space(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for buffer space for "
                    "command. Sending anyway."
                )
                break
        self._pending.put_nowait(PendingCommand(command_len, None))
        count = self._add(command_len)
        await self._transport.send(data)
        return count

    async def send_control(self, data: bytes) -> None:
        """
        Send a realtime control character (\\x18, !, ~) without
        buffer accounting.  Makes a best-effort attempt to wait for
        buffer space (up to 1 s), but sends regardless if space does
        not become available.
        """
        data_len = len(data)
        if self.needs_space(data_len):
            try:
                await asyncio.wait_for(self.wait_for_space(), timeout=1.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "Timed out waiting for buffer space for "
                    "control. Sending anyway."
                )
        await self._transport.send(data)

    @property
    def buffer_count(self) -> int:
        return self._get()

    def needs_space(self, needed: int) -> bool:
        """Return True if *needed* bytes would overflow the RX buffer."""
        return self._get() + needed > GRBL_RX_BUFFER_SIZE

    async def wait_for_space(self) -> None:
        """Clear and wait for one space-available signal."""
        self._space_available.clear()
        await self._space_available.wait()

    def ack_ok(self) -> PendingCommand:
        """
        Process an ``ok`` acknowledgement.  Pops the corresponding
        pending command, frees buffer space, and signals waiters.

        Raises ``asyncio.QueueEmpty`` when no pending command exists.
        """
        pending = self._pending.get_nowait()
        self._sub(pending.length)
        self._pending.task_done()
        task_mgr.loop.call_soon_threadsafe(self._space_available.set)
        return pending

    def ack_status_report(self) -> int:
        """
        Process a status report.
        Since polls bypass the RX buffer, we do not decrement the
        buffer count here.
        """
        return self._get()

    def reset(self) -> None:
        """Reset all buffer state (cancel, reconnect, cleanup)."""
        with self._lock:
            self._rx_buffer_count = 0
        self._pending = asyncio.Queue()
        self._space_available = asyncio.Event()
        self._space_available.set()

    def signal_space_available(self) -> None:
        """Unblock any waiters (e.g. on cancel)."""
        task_mgr.loop.call_soon_threadsafe(self._space_available.set)

    @property
    def pending_queue(self) -> asyncio.Queue[PendingCommand]:
        return self._pending

    async def wait_all_pending(self, timeout: float = 10.0) -> None:
        """Wait for all sent commands to be acknowledged."""
        await asyncio.wait_for(self._pending.join(), timeout=timeout)

    def _add(self, n: int) -> int:
        with self._lock:
            self._rx_buffer_count += n
            return self._rx_buffer_count

    def _sub(self, n: int) -> int:
        with self._lock:
            self._rx_buffer_count -= n
            if self._rx_buffer_count < 0:
                self._rx_buffer_count = 0
            return self._rx_buffer_count

    def _get(self) -> int:
        with self._lock:
            return self._rx_buffer_count
