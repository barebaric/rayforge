import asyncio
import re
import threading
import logging
from typing import Optional, NamedTuple, List
from enum import Enum, auto

from .serial import SerialTransport
from .transport import Transport

logger = logging.getLogger(__name__)

# Default RX buffer size for standard Grbl 1.1 (128-byte buffer,
# safe limit 127). Custom firmwares may report a smaller size via
# the $I OPT line.
DEFAULT_GRBL_RX_BUFFER_SIZE = 127


class PendingCommand(NamedTuple):
    length: int
    op_index: Optional[int]


class GrblResponseType(Enum):
    OK = auto()
    ERROR = auto()
    LINE = auto()


class GrblResponse(NamedTuple):
    type: GrblResponseType
    text: str
    pending: Optional[PendingCommand] = None


class GrblSerialTransport:
    """
    Wraps a SerialTransport with GRBL's character-counting flow control
    protocol.

    GRBL devices have a serial receive buffer whose size is reported
    in the $I OPT line (e.g. ``[OPT:VMPH,63,511]``).  Standard Grbl
    1.1 uses 127 bytes (128-byte hardware buffer minus one).  Custom
    firmwares may use smaller buffers.

    This transport tracks how many bytes are outstanding and provides
    backpressure so callers never overflow the device's RX buffer.

    Also handles low-level GRBL response parsing: extracts 'ok' and
    'error:' from the raw byte stream for buffer accounting, even when
    they are interleaved with status reports by buggy firmware.
    Thread-safe: the buffer count is mutated from both async tasks and
    the serial receive callback.
    """

    _ok_ack_re = re.compile(rb"ok\r+\n")

    def __init__(self, transport: Transport):
        self._transport = transport
        self._rx_buffer_count = 0
        self._rx_buffer_size = DEFAULT_GRBL_RX_BUFFER_SIZE
        self._lock = threading.Lock()
        self._pending: asyncio.Queue[PendingCommand] = asyncio.Queue()
        self._space_available = asyncio.Event()
        self._space_available.set()
        self._status_buffer = bytearray()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_connected(self) -> bool:
        return self._transport.is_connected

    @property
    def port(self) -> str:
        if isinstance(self._transport, SerialTransport):
            return self._transport.port
        return ""

    async def connect(self) -> None:
        self._loop = asyncio.get_running_loop()
        await self._transport.connect()

    async def disconnect(self) -> None:
        await self._transport.disconnect()

    @property
    def received(self):
        return self._transport.received

    @property
    def status_changed(self):
        return self._transport.status_changed

    def parse_incoming(self, data: bytes) -> List[GrblResponse]:
        """
        Parse raw serial bytes into GRBL responses.

        Scans for 'ok' and 'error:' responses in the byte stream,
        handling buffer accounting for them, before line-splitting.
        This prevents lost 'ok' acknowledgements when firmware
        interleaves them with status reports.

        Returns a list of GrblResponse for non-ok/error lines
        (status reports, alarms, info messages, etc.).
        """
        self._status_buffer.extend(data)
        responses: List[GrblResponse] = []

        self._extract_acks_from_buffer(responses)

        while b"\r\n" in self._status_buffer:
            end_idx = self._status_buffer.find(b"\r\n") + 2
            message_bytes = self._status_buffer[:end_idx]
            self._status_buffer = self._status_buffer[end_idx:]

            try:
                message = message_bytes.decode("utf-8")
            except UnicodeDecodeError:
                logger.warning(
                    f"Dropped invalid UTF-8 bytes: {message_bytes!r}"
                )
                continue

            for line in message.strip().splitlines():
                if not line:
                    continue
                resp = self._parse_line(line)
                if resp:
                    responses.append(resp)

        return responses

    def _extract_acks_from_buffer(self, responses):
        """
        Scan _status_buffer for 'ok\\r+\\n' and 'error:...\\r+\\n'
        patterns and extract them before line-based processing.

        This ensures buffer accounting is always correct, even when
        the firmware interleaves these responses into other messages.
        Also detects NULL-byte corrupted 'ok' responses as a hardware
        fault indicator.

        Handles \\r\\r\\n line endings that some serial layers
        produce by collapsing double-CR sequences.
        """
        while True:
            m = self._ok_ack_re.search(self._status_buffer)
            if m is None:
                break
            self._status_buffer = (
                self._status_buffer[: m.start()]
                + self._status_buffer[m.end() :]
            )
            pending = self._ack_ok()
            responses.append(GrblResponse(GrblResponseType.OK, "ok", pending))

        # Detect NULL-byte corrupted 'ok' (hardware fault)
        null_ok = self._find_null_corrupted_ok()
        while null_ok is not None:
            start, end = null_ok
            self._status_buffer = (
                self._status_buffer[:start] + self._status_buffer[end:]
            )
            logger.critical(
                "HARDWARE FAULT DETECTED: A corrupted 'ok' "
                "acknowledgement with NULL bytes was received. "
                "This indicates a critical problem with the "
                "USB cable, electrical noise (EMI), or power "
                "supply. The hardware connection MUST be fixed "
                "for reliable operation."
            )
            pending = self._ack_ok()
            responses.append(GrblResponse(GrblResponseType.OK, "ok", pending))
            null_ok = self._find_null_corrupted_ok()

        error_marker = b"error:"
        error_end = b"\r\n"
        while error_marker in self._status_buffer:
            start = self._status_buffer.index(error_marker)
            end = self._status_buffer.find(error_end, start)
            if end == -1:
                break
            end += 2
            error_bytes = self._status_buffer[start:end]
            self._status_buffer = (
                self._status_buffer[:start] + self._status_buffer[end:]
            )
            try:
                error_text = error_bytes.decode("utf-8").strip()
            except UnicodeDecodeError:
                continue
            logger.debug(
                f"Extracted '{error_text}' from raw buffer "
                f"(interleaved recovery)"
            )
            self._ack_ok()
            responses.append(GrblResponse(GrblResponseType.ERROR, error_text))

    def _find_null_corrupted_ok(self):
        """
        Search _status_buffer for a pattern like
        b'o\\x00k\\r+\\n' where NULL bytes are interspersed
        in 'ok'. Returns (start, end) indices or None.
        """
        buf = self._status_buffer
        i = 0
        while i < len(buf):
            if buf[i] == ord(b"o"):
                j = i + 1
                while j < len(buf) and buf[j] == 0:
                    j += 1
                if j < len(buf) and buf[j] == ord(b"k") and j > i + 1:
                    k = j + 1
                    while k < len(buf) and buf[k] == ord(b"\r"):
                        k += 1
                    if k < len(buf) and buf[k] == ord(b"\n"):
                        return (i, k + 1)
            i += 1
        return None

    def _parse_line(self, line: str) -> Optional[GrblResponse]:
        """
        Parse a single decoded line into a GrblResponse.
        Handles 'ok', 'error:', and returns everything else
        as a LINE type.
        """
        if line == "ok":
            pending = self._ack_ok()
            return GrblResponse(GrblResponseType.OK, "ok", pending)
        if line.startswith("error:"):
            self._ack_ok()
            return GrblResponse(GrblResponseType.ERROR, line)
        return GrblResponse(GrblResponseType.LINE, line)

    def clear_buffer(self) -> None:
        """Clear the internal line buffer."""
        self._status_buffer = bytearray()

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
        buffer accounting.  Realtime commands are intercepted by
        GRBL before entering the RX buffer, so no space check is
        needed.
        """
        await self._transport.send(data)

    @property
    def buffer_count(self) -> int:
        return self._get()

    def needs_space(self, needed: int) -> bool:
        """Return True if *needed* bytes would overflow the RX buffer."""
        return self._get() + needed > self._rx_buffer_size

    def set_rx_buffer_size(self, size: int) -> None:
        """
        Update the device's RX buffer size (as reported by the $I OPT
        line).  Must be called before streaming begins.
        """
        if size > 0:
            old_size = self._rx_buffer_size
            with self._lock:
                self._rx_buffer_size = size
            logger.info(
                f"GRBL RX buffer size set to {size} bytes (was {old_size})"
            )

    async def wait_for_space(self) -> None:
        """Clear and wait for one space-available signal."""
        self._space_available.clear()
        await self._space_available.wait()

    def _ack_ok(self) -> Optional[PendingCommand]:
        """
        Process an ``ok`` or ``error:`` acknowledgement.  Pops the
        corresponding pending command, frees buffer space, and
        signals waiters.

        Returns None when no pending command exists.
        """
        try:
            pending = self._pending.get_nowait()
        except asyncio.QueueEmpty:
            logger.warning("Received ack but sent gcode queue was empty.")
            return None
        logger.debug(
            f"Buffer ack: freeing {pending.length} bytes for "
            f"op_index={pending.op_index} "
            f"(pending queue size: {self._pending.qsize()})"
        )
        self._sub(pending.length)
        self._pending.task_done()
        self.signal_space_available()
        return pending

    def ack_status_report(self) -> int:
        """
        Process a status report.
        Since polls bypass the RX buffer, we do not decrement the
        buffer count here.
        """
        return self._get()

    def reset_flow_control(self) -> None:
        """Reset flow-control state without clearing the parse buffer."""
        with self._lock:
            self._rx_buffer_count = 0
        self._pending = asyncio.Queue()
        self._space_available = asyncio.Event()
        self._space_available.set()

    def reset(self) -> None:
        """Reset all buffer state (cancel, reconnect, cleanup)."""
        self.reset_flow_control()
        self._status_buffer = bytearray()

    def signal_space_available(self) -> None:
        """Unblock any waiters (e.g. on cancel)."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._space_available.set)
        else:
            self._space_available.set()

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
                logger.warning(
                    f"Buffer underflow: count went to "
                    f"{self._rx_buffer_count} after freeing {n} "
                    f"bytes. Clamping to 0."
                )
                self._rx_buffer_count = 0
            return self._rx_buffer_count

    def _get(self) -> int:
        with self._lock:
            return self._rx_buffer_count
