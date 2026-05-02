import logging
import asyncio
import inspect
import serial.serialutil
from gettext import gettext as _
from typing import (
    Optional,
    Any,
    List,
    Dict,
    cast,
    TYPE_CHECKING,
    Callable,
    Union,
    Awaitable,
)
from ....context import RayforgeContext
from ....core.varset import Var, VarSet, SerialPortVar, BaudrateVar
from ....pipeline.encoder.base import OpsEncoder, EncodedOutput
from ....pipeline.encoder.gcode import GcodeEncoder
from ...transport import TransportStatus, SerialTransport
from ...transport.grbl import (
    GrblSerialTransport,
    GrblResponseType,
    DEFAULT_GRBL_RX_BUFFER_SIZE,
)
from ...transport.serial import SerialPortPermissionError
from ..driver import (
    Driver,
    DriverSetupError,
    DeviceStatus,
    DriverPrecheckError,
    DeviceConnectionError,
    Axis,
    Pos,
    DeviceError,
)
from .grbl_util import (
    parse_state,
    get_grbl_setting_varsets,
    grbl_setting_re,
    wcs_re,
    prb_re,
    gcode_to_p_number,
    error_code_to_device_error,
    alarm_code_to_device_error,
    CommandRequest,
    parse_grbl_parser_state,
    parse_opt_info,
    parse_version,
    strip_gcode_comments,
)

if TYPE_CHECKING:
    from ....core.doc import Doc
    from ...models.machine import Machine
    from ...models.laser import Laser

logger = logging.getLogger(__name__)


class GrblSerialDriver(Driver):
    """
    An advanced GRBL serial driver that supports reading and writing
    device settings ($$ commands).
    """

    label = _("GRBL (Serial)")
    subtitle = _("GRBL-compatible serial connection")
    supports_settings = True
    reports_granular_progress = True

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self.grbl_transport: Optional[GrblSerialTransport] = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._current_request: Optional[CommandRequest] = None
        self._interactive_request: Optional[CommandRequest] = None
        self._cmd_lock = asyncio.Lock()
        self._command_queue: asyncio.Queue[CommandRequest] = asyncio.Queue()
        self._command_task: Optional[asyncio.Task] = None
        self._is_cancelled = False
        self._raw_grbl_status: DeviceStatus = DeviceStatus.UNKNOWN
        self._job_running = False
        self._on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None
        self._last_reported_op_index = -1
        self._job_exception: Optional[Exception] = None
        self._poll_status_while_running: bool = False
        self._handshake_received = asyncio.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def machine_space_wcs(self) -> str:
        return "G53"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates (G53)")

    @property
    def resource_uri(self) -> Optional[str]:
        if self.grbl_transport and self.grbl_transport.port:
            return f"serial://{self.grbl_transport.port}"
        return None

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        """Checks for systemic serial port issues before setup."""
        try:
            SerialTransport.check_serial_permissions_globally()
        except SerialPortPermissionError as e:
            # Re-raise as a precheck error for the UI.
            raise DriverPrecheckError(str(e)) from e

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                SerialPortVar(
                    key="port",
                    label=_("Port"),
                    description=_("Serial port for the device"),
                ),
                BaudrateVar(
                    "baudrate",
                    choices=SerialTransport.list_baud_rates(),
                ),
                Var(
                    key="poll_status_while_running",
                    label=_("Poll device status during jobs"),
                    description=_(
                        "Periodically query the device for position and "
                        "status while a job is running. Warning: Some "
                        "devices have trouble maintaining a stable "
                        "connection if this is used!"
                    ),
                    var_type=bool,
                    default=False,
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        """Returns a GcodeEncoder configured for the machine's dialect."""
        assert machine.dialect is not None
        return GcodeEncoder(machine.dialect)

    def _setup_implementation(self, **kwargs: Any) -> None:
        port = cast(str, kwargs.get("port", ""))
        baudrate = kwargs.get("baudrate", 115200)
        self._poll_status_while_running = bool(
            kwargs.get("poll_status_while_running", False)
        )

        if not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baud rate must be configured."))

        # Note that we intentionally do not check if the serial
        # port exists, as a missing port is a common occurance when
        # e.g. the USB cable is not plugged in, and not a sign of
        # misconfiguration.

        if port.startswith("/dev/ttyS"):
            logger.warning(
                f"Port {port} is a hardware serial port, which is unlikely "
                f"for USB-based GRBL devices."
            )

        serial_transport = SerialTransport(port, baudrate)
        self.grbl_transport = GrblSerialTransport(serial_transport)
        self.grbl_transport.received.connect(self.on_serial_data_received)
        self.grbl_transport.status_changed.connect(
            self.on_serial_status_changed
        )

    def on_serial_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        """
        Handle status changes from the serial transport.

        Suppresses the transport-level CONNECTED signal to prevent
        premature actions (like $G/$# sync) before the driver has
        verified the device responds. The driver emits its own
        CONNECTED in _connection_loop after handshake verification.
        """
        if status == TransportStatus.CONNECTED:
            logger.debug(
                "Suppressing transport-level CONNECTED. Driver will "
                "emit CONNECTED after handshake verification."
            )
            return
        logger.debug(
            f"Serial transport status changed: {status}, message: {message}"
        )
        self._update_connection_status(status, message)

    async def cleanup(self):
        logger.debug("Cleanup initiated.")
        self.keep_running = False
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done = None
        if self.grbl_transport:
            self.grbl_transport.reset()
        self._job_exception = None

        # Cancel tasks and wait for them to ensure loops terminate
        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(
                    f"Ignored exception in connection task during cleanup: {e}"
                )
            self._connection_task = None

        if self._command_task:
            self._command_task.cancel()
            try:
                await self._command_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(
                    f"Ignored exception in command task during cleanup: {e}"
                )
            self._command_task = None

        if self.grbl_transport:
            self.grbl_transport.received.disconnect(
                self.on_serial_data_received
            )
            self.grbl_transport.status_changed.disconnect(
                self.on_serial_status_changed
            )
            # Close the serial port to prevent duplicate readers/writers
            if self.grbl_transport.is_connected:
                await self.grbl_transport.disconnect()

        await super().cleanup()
        logger.debug("Cleanup completed.")

    async def _send_realtime(self, command: str, add_newline: bool = True):
        logger.debug(f"Sending realtime command: {command}")
        if not self.grbl_transport or not self.grbl_transport.is_connected:
            raise ConnectionError("Serial transport not initialized")
        payload = (command + ("\n" if add_newline else "")).encode("utf-8")
        logger.debug(
            f"TX: {payload!r}",
            extra={
                "log_category": "RAW_IO",
                "direction": "TX",
                "data": payload,
            },
        )
        await self.grbl_transport.send_control(payload)

    async def _connect_implementation(self):
        """
        Launches the connection loop as a background task and returns,
        allowing the UI to remain responsive.
        """
        # Defensive cleanup of existing tasks
        if self._connection_task and not self._connection_task.done():
            logger.warning(
                "Connect called with active connection task. Cleaning up."
            )
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

        if self._command_task and not self._command_task.done():
            self._command_task.cancel()
            try:
                await self._command_task
            except asyncio.CancelledError:
                pass

        # Check if setup was successful (grbl_transport exists)
        if not self.grbl_transport:
            logger.error(
                "Cannot connect: Transport not initialized "
                "(check port settings)."
            )
            self._update_connection_status(
                TransportStatus.ERROR, _("Port not configured")
            )
            return

        logger.debug("Connect initiated.")
        self._loop = asyncio.get_running_loop()
        self.keep_running = True
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done = None
        self.grbl_transport.reset()
        self._job_exception = None
        self._connection_task = asyncio.create_task(self._connection_loop())
        self._command_task = asyncio.create_task(self._process_command_queue())

    def _set_request_finished(self, request: "CommandRequest") -> None:
        if not request.finished.is_set():
            if self._loop is not None:
                self._loop.call_soon_threadsafe(request.finished.set)
            else:
                request.finished.set()

    async def _connection_loop(self) -> None:
        logger.debug("Entering _connection_loop.")
        while self.keep_running:
            logger.debug("Attempting connection…")

            try:
                transport = self.grbl_transport
                if not transport:
                    raise DriverSetupError("Transport not initialized")

                await transport.connect()
                logger.debug(
                    "Serial port opened. Verifying device response..."
                )

                self._handshake_received.clear()
                await self._send_realtime("?", add_newline=False)

                try:
                    await asyncio.wait_for(
                        self._handshake_received.wait(), timeout=2.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "No response from device. Port may be a phantom "
                        "COM port without a connected device."
                    )
                    await transport.disconnect()
                    self._update_connection_status(
                        TransportStatus.ERROR,
                        _("No response from device"),
                    )
                    self._update_connection_status(TransportStatus.SLEEPING)
                    await asyncio.sleep(5)
                    continue

                logger.info("Connection established successfully.")

                try:
                    await self._execute_interactive_command("$I")
                except (ConnectionError, asyncio.TimeoutError) as e:
                    logger.warning(f"Failed to retrieve build info: {e}")

                self._update_connection_status(TransportStatus.CONNECTED)

                logger.debug("Connection verified. Starting status polling.")
                while transport.is_connected and self.keep_running:
                    # Skip status polling during jobs if configured
                    if (
                        not self._poll_status_while_running
                        and self._job_running
                    ):
                        await asyncio.sleep(0.5)
                        continue

                    async with self._cmd_lock:
                        try:
                            payload = b"?"
                            count = await transport.send_poll(payload)
                            buf_info = (
                                f"buf: {count}/{transport._rx_buffer_size}"
                            )
                            logger.debug(
                                f"TX: {payload!r} ({buf_info})",
                                extra={
                                    "log_category": "RAW_IO",
                                    "direction": "TX",
                                    "data": payload,
                                },
                            )
                        except ConnectionError as e:
                            logger.warning(
                                "Connection lost while sending poll"
                                f" command: {e}"
                            )
                            break
                    await asyncio.sleep(0.5)

                    if not self.keep_running or not transport.is_connected:
                        break

            except (serial.serialutil.SerialException, OSError) as e:
                logger.error(f"Connection error: {e}")
                # Don't update status here - the transport's status_changed
                # signal already sent ERROR status via
                # on_serial_status_changed.
            except asyncio.CancelledError:
                logger.info("Connection loop cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error in connection loop: {e}",
                    exc_info=True,
                )
                self._update_connection_status(TransportStatus.ERROR, str(e))
            finally:
                if self.grbl_transport and self.grbl_transport.is_connected:
                    logger.debug("Disconnecting transport in finally block")
                    await self.grbl_transport.disconnect()

            if not self.keep_running:
                break

            logger.debug("Connection lost. Reconnecting in 5s…")
            self._update_connection_status(TransportStatus.SLEEPING)
            await asyncio.sleep(5)

        logger.debug("Leaving _connection_loop.")

    async def _process_command_queue(self) -> None:
        logger.debug("Entering _process_command_queue.")
        while self.keep_running:
            try:
                request = await self._command_queue.get()
                if (
                    not self.grbl_transport
                    or not self.grbl_transport.is_connected
                    or self._is_cancelled
                ):
                    logger.warning(
                        "Cannot process command: Serial transport not "
                        "connected or job is cancelled. Dropping command."
                    )
                    self._set_request_finished(request)
                    self._command_queue.task_done()
                    continue

                self._current_request = request
                try:
                    cmd_text = request.command.strip()
                    if cmd_text:
                        logger.info(
                            cmd_text,
                            extra=self._log_extra("USER_COMMAND"),
                        )
                    logger.debug(
                        f"TX: {request.payload!r}",
                        extra={
                            "log_category": "RAW_IO",
                            "direction": "TX",
                            "data": request.payload,
                        },
                    )

                    async with self._cmd_lock:
                        if (
                            not self.grbl_transport
                            or not self.grbl_transport.is_connected
                        ):
                            raise ConnectionError(
                                "Serial transport disconnected during command."
                            )
                        await self.grbl_transport.send_command(request.payload)

                    # Wait for the response to arrive outside the lock
                    # so that status polling and gcode streaming are not
                    # blocked. The timeout is handled by the caller
                    # (_execute_command).
                    await request.finished.wait()

                except ConnectionError as e:
                    logger.error(f"Connection error during command: {e}")
                    self._update_connection_status(
                        TransportStatus.ERROR,
                        str(e),
                    )
                finally:
                    self._current_request = None
                    self._command_queue.task_done()

                # Release lock briefly to allow status polling
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("Command queue processing cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error in command queue: {e}",
                    exc_info=True,
                )
                self._update_connection_status(TransportStatus.ERROR, str(e))
        logger.debug("Leaving _process_command_queue.")

    def _start_job(
        self,
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ):
        """Initializes state for a new streaming job."""
        self._is_cancelled = False
        self._job_running = True
        self._on_command_done = on_command_done
        self._last_reported_op_index = -1
        self._job_exception = None
        if self.grbl_transport:
            self.grbl_transport.reset()

    async def _recover_from_deadlock(self, transport) -> None:
        """
        Recover from a detected deadlock by sending a G4 P0.01 dwell.
        When its 'ok' arrives, the planner buffer is guaranteed empty.
        Then reset host-side buffer accounting.
        """
        if not transport or not transport.is_connected:
            logger.warning("Cannot recover: transport disconnected.")
            return
        logger.info("Deadlock recovery: sending G4 P0.01 to drain planner.")
        try:
            async with self._cmd_lock:
                dwell = b"G4 P0.01\n"
                await transport.send_gcode(dwell)
                logger.debug(f"TX: {dwell!r} (deadlock recovery)")
            try:
                await asyncio.wait_for(
                    transport.pending_queue.join(), timeout=30.0
                )
                logger.info("Deadlock recovery: all pending acks received.")
            except asyncio.TimeoutError:
                logger.warning(
                    "Deadlock recovery: timed out waiting for acks "
                    "after G4 P0.01. Resetting host buffers."
                )
            transport.reset()
        except (ConnectionError, OSError) as e:
            logger.warning(f"Deadlock recovery failed: {e}")

    def _is_grbl_idle_or_desynced(self, transport) -> bool:
        """
        Check if GRBL is actually idle or buffer tracking has
        desynchronized.

        During jobs, the IDLE status is overridden to RUN for the
        UI, so ``self.state.status == IDLE`` is never true.  Instead
        we check: (1) the raw GRBL status before the override, (2)
        whether GRBL reports full buffer availability while we still
        have pending commands, or (3) whether a non-busy raw status
        (IDLE/HOLD) has been seen repeatedly with a non-empty
        pending queue — indicating a desync even without ``Bf:``.
        """
        if self.state.status == DeviceStatus.IDLE:
            return True
        if self._raw_grbl_status in (
            DeviceStatus.IDLE,
            DeviceStatus.HOLD,
        ):
            return True
        buf_avail = self.state.buffer_available
        if buf_avail is not None and transport._rx_buffer_size > 0:
            if buf_avail >= transport._rx_buffer_size:
                return True
        return False

    async def _poll_and_check_idle(self, transport) -> bool:
        """
        Send a realtime status poll and check if GRBL is idle.

        Resets ``_raw_grbl_status`` to UNKNOWN, sends a ``?`` poll
        (which bypasses the RX buffer), and waits briefly for a
        fresh status report.  Returns True only if the fresh
        response confirms GRBL is idle or buffer-desynchronized.

        This avoids false deadlock detection when status polling is
        disabled during jobs and ``_raw_grbl_status`` is stale.
        """
        self._raw_grbl_status = DeviceStatus.UNKNOWN
        try:
            async with self._cmd_lock:
                if (
                    not self.grbl_transport
                    or not self.grbl_transport.is_connected
                ):
                    return False
                await transport.send_poll(b"?")
        except (ConnectionError, OSError):
            return False
        for _attempt in range(10):
            await asyncio.sleep(0.1)
            if self._raw_grbl_status != DeviceStatus.UNKNOWN:
                break
        return self._is_grbl_idle_or_desynced(transport)

    async def _wait_for_buffer_space(
        self, transport, command_len: int, timeout: float
    ) -> None:
        """Wait for buffer space, recovering from deadlocks if needed."""
        while transport.needs_space(command_len):
            if self.state.status == DeviceStatus.ALARM:
                if not self._job_exception:
                    self._job_exception = DeviceConnectionError(
                        "Machine entered ALARM state during job."
                    )
                return

            try:
                await asyncio.wait_for(
                    transport.wait_for_space(), timeout=timeout
                )
            except asyncio.TimeoutError:
                if await self._poll_and_check_idle(transport):
                    logger.warning(
                        "Deadlock detected during streaming. "
                        "Attempting G4 P0.01 recovery."
                    )
                    await self._recover_from_deadlock(transport)
                    if transport.needs_space(command_len):
                        logger.error("Recovery failed: buffer still full.")
                        self._job_exception = DeviceConnectionError(
                            "Deadlock recovery failed."
                        )
                        return
                else:
                    logger.warning(
                        "Timeout waiting for buffer space "
                        "(machine not IDLE). Retrying."
                    )

            if self._job_exception:
                return
            if self._is_cancelled:
                raise asyncio.CancelledError("Job cancelled")

    async def _send_gcode_line(
        self,
        transport,
        line: str,
        command_bytes: bytes,
        op_index: Optional[int],
    ) -> None:
        """Send a single gcode line with buffer accounting."""
        async with self._cmd_lock:
            if not self.grbl_transport or not self.grbl_transport.is_connected:
                raise ConnectionError(
                    "Serial transport disconnected during job."
                )

            logger.info(line, extra=self._log_extra("USER_COMMAND"))

            # Add to queue BEFORE sending.
            # Fast machines can reply with 'ok' before await
            # send() returns. If we queue after sending, the RX
            # handler finds an empty queue and drops the 'ok',
            # causing a deadlock.
            count = await transport.send_gcode(command_bytes, op_index)
            buf_info = f"buf: {count}/{transport._rx_buffer_size}"
            logger.debug(
                f"TX: {command_bytes!r} ({buf_info})",
                extra={
                    "log_category": "RAW_IO",
                    "direction": "TX",
                    "data": command_bytes,
                },
            )

    async def _drain_pending_acks(self, transport, timeout: float) -> None:
        """Wait for all pending acks, recovering from deadlocks."""
        while not transport.pending_queue.empty():
            if self._job_exception or self.state.status == DeviceStatus.ALARM:
                break
            if self._is_cancelled:
                break

            try:
                await asyncio.wait_for(
                    transport.pending_queue.join(), timeout=timeout
                )
                logger.debug("All 'ok' responses received.")
                break
            except asyncio.TimeoutError:
                if await self._poll_and_check_idle(transport):
                    logger.warning(
                        "Deadlock detected at end of job. "
                        "Attempting G4 P0.01 recovery."
                    )
                    await self._recover_from_deadlock(transport)
                else:
                    logger.warning(
                        "Timeout waiting for acks "
                        "(machine not IDLE). Retrying."
                    )

    async def _stream_gcode(
        self,
        gcode_lines: List[str],
        machine_code_to_op_map: Optional[Dict[int, int]] = None,
    ):
        """
        The core G-code streaming logic using character-counting protocol.
        Assumes _start_job() has been called.
        """
        logger.debug(
            f"Starting GRBL streaming job with {len(gcode_lines)} lines."
        )
        transport = self.grbl_transport
        if not transport:
            raise ConnectionError("Transport not initialized")
        job_completed_successfully = False
        deadlock_timeout = 10.0 if not self._poll_status_while_running else 2.0
        try:
            for line_idx, line in enumerate(gcode_lines):
                if (
                    self._is_cancelled
                    or self._job_exception
                    or self.state.status == DeviceStatus.ALARM
                ):
                    logger.info(
                        "Job cancelled, errored, or machine in ALARM "
                        "state. Stopping G-code sending."
                    )
                    if self.state.status == DeviceStatus.ALARM:
                        if not self._job_exception:
                            self._job_exception = DeviceConnectionError(
                                "Machine entered ALARM state during job."
                            )
                    break

                line = strip_gcode_comments(line)
                if not line:
                    continue

                op_index = (
                    machine_code_to_op_map.get(line_idx)
                    if machine_code_to_op_map
                    else None
                )
                command_bytes = (line + "\n").encode("utf-8")

                await self._wait_for_buffer_space(
                    transport, len(command_bytes), deadlock_timeout
                )
                if (
                    self._job_exception
                    or self.state.status == DeviceStatus.ALARM
                ):
                    break

                await self._send_gcode_line(
                    transport, line, command_bytes, op_index
                )
                await asyncio.sleep(0)

            if not self._is_cancelled and not self._job_exception:
                logger.debug(
                    "All G-code sent. Waiting for all 'ok' responses."
                )
                await self._drain_pending_acks(transport, deadlock_timeout)

            if self._job_exception:
                raise self._job_exception

            job_completed_successfully = not self._is_cancelled

        except (
            asyncio.CancelledError,
            ConnectionError,
            DeviceConnectionError,
        ) as e:
            logger.warning(f"Job interrupted: {e!r}")
            job_completed_successfully = False
            # If not cancelled explicitly, send a cancel command
            if not self._is_cancelled:
                logger.info(f"Calling cancel() due to interruption: {e!r}")
                await self.cancel()
            # Do not re-raise ConnectionError or
            # DeviceConnectionError, let the task finish "failed"
            # Only re-raise CancelledError to propagate cancellation
            # upwards.
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self._raw_grbl_status = DeviceStatus.UNKNOWN
            self._job_running = False
            self._on_command_done = None
            if job_completed_successfully:
                self.job_finished.send(self)
            # Signal job termination for UI cleanup, even on failure.
            elif not self._is_cancelled:
                self.job_finished.send(self)
            logger.debug("G-code streaming finished.")

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        self._start_job(on_command_done)

        mapping = encoded.op_map.machine_code_to_op if encoded.op_map else None
        gcode_lines = encoded.text.splitlines()

        try:
            await self._stream_gcode(gcode_lines, mapping)
        except DeviceConnectionError as e:
            # Catch the device error here to prevent it from propagating
            # up and tearing down the connection task. The error has
            # already been logged inside _stream_gcode.
            logger.warning(
                f"Job terminated due to device error: {e}. "
                "Connection remains active."
            )

    async def run_raw(self, machine_code: str) -> None:
        """
        Executes a raw G-code string using the character-counting
        streaming protocol.
        """
        lines = [
            line.strip() for line in machine_code.splitlines() if line.strip()
        ]
        if not lines:
            return
        self._start_job()
        try:
            await self._stream_gcode(lines)
        except DeviceConnectionError as e:
            logger.warning(
                f"Raw G-code terminated due to device error: {e}. "
                "Connection remains active."
            )

    async def cancel(self) -> None:
        logger.debug("Cancel command initiated.")
        job_was_running = self._job_running
        self._is_cancelled = True
        self._job_running = False
        self._on_command_done = None

        # Unblock the run loop if it's waiting
        if self.grbl_transport:
            self.grbl_transport.signal_space_available()

            logger.info("Sending Soft Reset (Ctrl-X) to device.")
            payload = b"\x18"
            logger.debug(
                f"TX: {payload!r}",
                extra={
                    "log_category": "RAW_IO",
                    "direction": "TX",
                    "data": payload,
                },
            )
            await self.grbl_transport.send_control(payload)
            while not self._command_queue.empty():
                try:
                    request = self._command_queue.get_nowait()
                    self._set_request_finished(request)
                    self._command_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            logger.debug("Command queue cleared after cancel.")

            # Clear the streaming queue and buffer state
            self.grbl_transport.reset()
            logger.debug("Streaming queue cleared after cancel.")

            if job_was_running:
                self.job_finished.send(self)
        else:
            raise ConnectionError("Serial transport not initialized")

    async def _execute_command(self, command: str) -> List[str]:
        self._is_cancelled = False
        request = CommandRequest(command)
        await self._command_queue.put(request)
        try:
            await asyncio.wait_for(request.finished.wait(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.error(
                f"Command '{command}' timed out after 10 seconds. "
                "Unblocking command queue."
            )
            self._set_request_finished(request)
            raise
        except asyncio.CancelledError:
            logger.debug(
                f"Command '{command}' was cancelled. Unblocking queue."
            )
            self._set_request_finished(request)
            raise
        return request.response_lines

    async def _execute_interactive_command(self, command: str) -> List[str]:
        """
        Send a command and synchronously await its full response.

        Unlike ``_execute_command`` (which queues commands for
        asynchronous processing by ``_process_command_queue``), this
        method holds the command lock for the entire send-and-wait
        cycle so that no other command can be interleaved:

        1. Acquire ``_cmd_lock`` (blocks ``_process_command_queue``
           and status polling from sending anything).
        2. Drain the transport's pending-ack queue so that no stale
           ``ok``/``error`` is in flight.
        3. Set ``_interactive_request``, send the command, and wait
           for its response.
        4. Release the lock.

        ``_interactive_request`` is checked *before*
        ``_current_request`` in ``_handle_ok`` / ``_handle_error``,
        so an interactive command always claims the next
        acknowledgement even if ``_process_command_queue`` has a
        pending ``_current_request`` from before it blocked on the
        lock.
        """
        transport = self.grbl_transport
        if not transport or not transport.is_connected:
            raise ConnectionError("Serial transport not connected")

        request = CommandRequest(command)

        async with self._cmd_lock:
            await transport.pending_queue.join()

            self._interactive_request = request
            try:
                logger.info(
                    command.strip(),
                    extra=self._log_extra("USER_COMMAND"),
                )
                logger.debug(
                    f"TX: {request.payload!r}",
                    extra={
                        "log_category": "RAW_IO",
                        "direction": "TX",
                        "data": request.payload,
                    },
                )
                await transport.send_command(request.payload)
                await asyncio.wait_for(request.finished.wait(), timeout=10.0)
            except asyncio.TimeoutError:
                logger.error(
                    f"Interactive command '{command}' timed out "
                    "after 10 seconds."
                )
                raise
            finally:
                self._interactive_request = None

        return request.response_lines

    async def set_hold(self, hold: bool = True) -> None:
        self._is_cancelled = False
        await self._send_realtime("!" if hold else "~", add_newline=False)

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        """GRBL supports homing for all axes."""
        return True

    async def home(self, axes: Optional[Axis] = None) -> None:
        """
        Homes the specified axes or all axes if none specified.

        Args:
            axes: Optional axis or combination of axes to home. If None,
                 homes all axes. Can be a single Axis or multiple axes
                 using binary operators (e.g. Axis.X|Axis.Y)
        """
        dialect = self.dialect

        # Execute the homing command(s)
        if axes is None:
            await self._execute_command(dialect.home_all)
        else:
            for axis in Axis:
                if axes & axis:
                    assert axis.name
                    axis_letter: str = axis.name.upper()
                    cmd = dialect.home_axis.format(axis_letter=axis_letter)
                    await self._execute_command(cmd)

        # The following works around a quirk in some Grbl versions:
        # After homing, the machine is still in G54, but forgets its
        # offset. To re-activate the offset, we toggle to another
        # WCS and then back.
        # Just sending G54 is ignored if GRBL thinks it's already
        # in G54.
        active_wcs = self._machine.active_wcs
        temp_wcs = "G55" if active_wcs == "G54" else "G54"

        # Flush planner buffer
        await self._execute_command("G4 P0.01")

        # Toggle sequence
        await self._execute_command(temp_wcs)
        await self._execute_command(active_wcs)
        self.state.error = None
        self.state_changed.send(self, state=self.state)

    async def move_to(self, pos_x, pos_y) -> None:
        dialect = self.dialect
        cmd = dialect.move_to.format(
            speed=1500, x=float(pos_x), y=float(pos_y)
        )
        await self._execute_command(cmd)

    async def select_tool(self, tool_number: int) -> None:
        """Sends a tool change command for the given tool number."""
        dialect = self.dialect
        cmd = dialect.tool_change.format(tool_number=tool_number)
        await self._execute_command(cmd)

    async def clear_alarm(self) -> None:
        dialect = self.dialect
        response = await self._execute_command(dialect.clear_alarm)
        has_error = any(line.startswith("error:") for line in response)
        if not has_error:
            self.state.error = None
            self.state_changed.send(self, state=self.state)

    async def set_power(self, head: "Laser", percent: float) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control.
            percent: Power percentage (0.0-1.0). 0 disables power.
        """
        # Get the dialect for power control commands
        dialect = self.dialect

        if percent <= 0:
            # Disable power
            cmd = dialect.laser_off
        else:
            # Enable power with specified percentage
            power_abs = percent * head.max_power
            cmd = dialect.laser_on.format(power=power_abs)

        await self._execute_command(cmd)

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        """
        Sets the laser power for focus mode using the focus_laser_on
        command.

        Args:
            head: The laser head to control.
            percent: Power percentage (0.0-1.0). 0 disables power.
        """
        dialect = self.dialect

        if percent <= 0:
            cmd = dialect.laser_off
        else:
            power_abs = percent * head.max_power
            cmd = dialect.focus_laser_on.format(power=power_abs)

        await self._execute_command(cmd)

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """GRBL supports jogging for all axes."""
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        """
        Jogs the machine using GRBL's $J command.

        Args:
            speed: The jog speed in mm/min
            **deltas: Axis names and distances (e.g. x=10.0, y=5.0)
        """
        # Build the command with all specified axes
        dialect = self.dialect
        cmd_parts = [dialect.jog.format(speed=speed)]

        for axis_name, distance in deltas.items():
            cmd_parts.append(f"{axis_name.upper()}{distance}")

        if len(cmd_parts) == 1:
            return

        cmd = " ".join(cmd_parts)
        await self._execute_command(cmd)

    def get_setting_vars(self) -> List["VarSet"]:
        return get_grbl_setting_varsets()

    async def read_settings(self) -> None:
        response_lines = await self._execute_interactive_command("$$")
        # Get the list of VarSets, which serve as our template
        known_varsets = self.get_setting_vars()

        # For efficient lookup, map each setting key to its parent VarSet
        key_to_varset_map = {
            var_key: varset
            for varset in known_varsets
            for var_key in varset.keys()
        }

        unknown_vars = VarSet(
            title=_("Unknown Settings"),
            description=_(
                "Settings reported by the device not in the standard list."
            ),
        )

        for line in response_lines:
            match = grbl_setting_re.match(line)
            if match:
                key, value_str = match.groups()
                # Find which VarSet this key belongs to
                target_varset = key_to_varset_map.get(key)
                if target_varset:
                    # Update the value in the correct VarSet
                    target_varset[key] = value_str
                else:
                    # This setting is not defined in our known VarSets
                    unknown_vars.add(
                        Var(
                            key=key,
                            label=f"${key}",
                            var_type=str,
                            value=value_str,
                            description=_("Unknown setting from device"),
                        )
                    )

        # The result is the list of known VarSets (now populated)
        result = known_varsets
        if len(unknown_vars) > 0:
            # Append the VarSet of unknown settings if any were found
            result.append(unknown_vars)

        num_settings = sum(len(vs) for vs in result)
        logger.info(
            f"Driver settings read with {num_settings} settings.",
            extra={"log_category": "DRIVER_EVENT"},
        )
        self.settings_read.send(self, settings=result)

    async def write_setting(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            value = 1 if value else 0
        cmd = f"${key}={value}"
        await self._execute_command(cmd)

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        p_num = gcode_to_p_number(wcs_slot)
        if p_num is None:
            raise ValueError(f"Invalid WCS slot: {wcs_slot}")
        dialect = self.dialect
        cmd = dialect.set_wcs_offset.format(p_num=p_num, x=x, y=y, z=z)
        await self._execute_command(cmd)

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        response_lines = await self._execute_interactive_command("$#")
        offsets = {}
        for line in response_lines:
            match = wcs_re.match(line)
            if match:
                slot, x_str, y_str, z_str = match.groups()
                z_str = z_str or "0.000"
                offsets[slot] = (
                    float(x_str),
                    float(y_str),
                    float(z_str),
                )
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> Optional[str]:
        """Reads the $G parser state to determine the active WCS."""
        try:
            response_lines = await self._execute_interactive_command("$G")
            return parse_grbl_parser_state(response_lines)
        except DeviceConnectionError as e:
            logger.warning(f"Could not read parser state: {e}")
            return None

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        assert axis.name, "Probing requires a single, named axis."
        axis_letter = axis.name.upper()
        dialect = self.dialect
        cmd = dialect.probe_cycle.format(
            axis_letter=axis_letter,
            max_travel=max_travel,
            feed_rate=feed_rate,
        )

        self.probe_status_changed.send(
            self, message=f"Probing {axis_letter}..."
        )
        try:
            response_lines = await self._execute_interactive_command(cmd)
        except DeviceConnectionError:
            self.probe_status_changed.send(
                self, message="Probe failed: Timed out"
            )
            return None

        for line in response_lines:
            match = prb_re.match(line)
            if match:
                x_str, y_str, z_str, success = match.groups()
                if int(success) == 1:
                    pos: Pos = (
                        float(x_str),
                        float(y_str),
                        float(z_str),
                    )
                    self.probe_status_changed.send(
                        self, message=f"Probe triggered at {pos}"
                    )
                    return pos

        self.probe_status_changed.send(self, message="Probe failed")
        return None

    def on_serial_data_received(self, sender, data: bytes):
        """
        Primary handler for incoming serial data. Delegates parsing
        to the transport layer and processes structured responses.
        """
        buf_count = (
            self.grbl_transport.buffer_count if self.grbl_transport else 0
        )
        buf_size = (
            self.grbl_transport._rx_buffer_size
            if self.grbl_transport
            else DEFAULT_GRBL_RX_BUFFER_SIZE
        )
        buf_info = f"buf: {buf_count}/{buf_size}"
        logger.debug(
            f"RX: {data!r} ({buf_info})",
            extra={
                "log_category": "RAW_IO",
                "direction": "RX",
                "data": data,
            },
        )

        if not self.grbl_transport:
            return

        responses = self.grbl_transport.parse_incoming(data)

        # Process LINE responses before OK/ERROR to ensure
        # informational lines are collected before the command
        # is marked as finished. _extract_acks_from_buffer
        # returns OK/ERROR first, but _handle_ok schedules
        # request.finished.set() via call_soon_threadsafe on a
        # separate thread. If that thread executes before
        # _handle_line runs, the lines would be lost.
        lines = []
        acks = []
        for resp in responses:
            if resp.type == GrblResponseType.LINE:
                lines.append(resp)
            else:
                acks.append(resp)
        for resp in lines + acks:
            self._handle_response(resp)

    def _handle_response(self, resp):
        """
        Route a parsed GrblResponse to the appropriate handler.
        """
        if resp.type == GrblResponseType.OK:
            self._handle_ok(resp)
        elif resp.type == GrblResponseType.ERROR:
            self._handle_error(resp.text)
        else:
            self._handle_line(resp.text)

    def _handle_status_report(self, report: str):
        """
        Parses a GRBL status report (e.g., '<Idle|WPos:0,0,0|...>')
        and updates the device state.
        """
        if self.grbl_transport:
            count = self.grbl_transport.ack_status_report()
        else:
            count = 0
        buf_size = (
            self.grbl_transport._rx_buffer_size
            if self.grbl_transport
            else DEFAULT_GRBL_RX_BUFFER_SIZE
        )
        buf_info = f"buf: {count}/{buf_size}"
        logger.debug(f"Processing status report: {report} ({buf_info})")
        logger.info(report, extra=self._log_extra("STATUS_POLL"))

        state = parse_state(
            report, self.state, lambda message: logger.info(message)
        )

        self._raw_grbl_status = state.status

        # If a job is active, 'Idle' state between commands should be
        # reported as 'Run' to the UI.
        if self._job_running and state.status == DeviceStatus.IDLE:
            state.status = DeviceStatus.RUN

        old_status = self.state.status
        if state != self.state:
            self.state = state
            if state.status != old_status:
                logger.info(
                    f"Device state changed: {self.state.status.name}",
                    extra=self._log_extra("STATE_CHANGE"),
                )
            self.state_changed.send(self, state=self.state)

    def _handle_ok(self, resp):
        """Handle a parsed 'ok' response."""
        pending = resp.pending
        logger.info("ok", extra=self._log_extra("MACHINE_RESPONSE"))

        if pending is not None:
            transport = self.grbl_transport
            assert transport is not None
            logger.debug(
                f"Processed 'ok', freed {pending.length} bytes "
                f"(buf: {transport.buffer_count}"
                f"/{transport._rx_buffer_size}, "
                f"op_index={pending.op_index})"
            )

        # Logic for single, interactive commands
        request = self._interactive_request or self._current_request
        if request and not request.finished.is_set():
            request.response_lines.append("ok")
            self.command_status_changed.send(self, status=TransportStatus.IDLE)
            logger.debug(f"Command '{request.command}' completed with 'ok'")
            self._set_request_finished(request)

        # Logic for streaming protocol during a job
        if self._job_running and pending is not None:
            if self._on_command_done and pending.op_index is not None:
                for i in range(
                    self._last_reported_op_index + 1,
                    pending.op_index + 1,
                ):
                    try:
                        logger.debug(
                            f"Firing on_command_done for op_index {i}"
                        )
                        result = self._on_command_done(i)
                        if inspect.isawaitable(result):
                            asyncio.ensure_future(result)
                    except Exception as e:
                        logger.error(
                            "Error in on_command_done callback",
                            exc_info=e,
                        )
                self._last_reported_op_index = pending.op_index

    def _handle_error(self, text: str):
        """Handle a parsed 'error:...' response."""
        logger.info(text, extra=self._log_extra("MACHINE_EVENT"))
        error_code = text.split(":")[1].strip() if ":" in text else ""
        self.state.error = error_code_to_device_error(error_code)
        self.state_changed.send(self, state=self.state)

        request = self._interactive_request or self._current_request
        if request and not request.finished.is_set():
            request.response_lines.append(text)
            self.command_status_changed.send(
                self, status=TransportStatus.ERROR, message=text
            )
            self._set_request_finished(request)

        if self._job_running:
            self.command_status_changed.send(
                self, status=TransportStatus.ERROR, message=text
            )
            logger.error(
                f"GRBL error during job: {text}. Halting stream.",
                extra={"log_category": "ERROR"},
            )
            self._job_exception = DeviceConnectionError(f"GRBL error: {text}")
            if self.grbl_transport:
                self.grbl_transport.signal_space_available()

    def _handle_line(self, line: str):
        """Handle a parsed general line (status report, alarm, info)."""
        if "Pos:" in line and "|" in line and not line.startswith("<"):
            logger.debug(f"Ignoring fragmented status report: {line}")
            return

        if line.startswith("<") and line.endswith(">"):
            self._handshake_received.set()
            self._handle_status_report(line.strip())
            return

        logger.info(line, extra=self._log_extra("MACHINE_EVENT"))

        # Collect response lines for pending single commands
        request = self._interactive_request or self._current_request
        if request and not request.finished.is_set():
            request.response_lines.append(line)

        if line.startswith("ALARM:"):
            alarm_code = line.split(":")[1].strip()
            self.state.error = alarm_code_to_device_error(alarm_code)
            self.state_changed.send(self, state=self.state)
            self.command_status_changed.send(
                self, status=TransportStatus.ERROR, message=line
            )
            if self._job_running:
                logger.error(
                    f"GRBL ALARM during job: {line}. Halting stream.",
                    extra={"log_category": "ERROR"},
                )
                self._job_exception = DeviceConnectionError(
                    f"GRBL ALARM: {line}"
                )
                if self.grbl_transport:
                    self.grbl_transport.signal_space_available()
        elif line.startswith("[VER:"):
            version_info = parse_version([line])
            if version_info:
                ver_num, ver_let = version_info
                logger.info(f"Connected to GRBL version {ver_num}{ver_let}")
        elif line.startswith("[OPT:"):
            rx_buffer_size = parse_opt_info(line)
            if rx_buffer_size and self.grbl_transport:
                self.grbl_transport.set_rx_buffer_size(rx_buffer_size)
        elif line.startswith("Grbl "):
            self._handshake_received.set()
            logger.debug(f"Received Grbl welcome message: {line}")
        else:
            logger.debug(f"Received informational line: {line}")

    def get_error(self, error_code: str) -> Optional[DeviceError]:
        """
        Returns error details for a given GRBL error code.

        Args:
            error_code: The error code string from device (e.g., "1",
                        "2").

        Returns:
            An ErrorCode instance with title and description, or None
            if the error code is not recognized.
        """
        return error_code_to_device_error(error_code)

    def _update_connection_status(
        self, status: TransportStatus, message: Optional[str] = None
    ):
        log_data = f"Connection status: {status.name}"
        if message:
            log_data += f" - {message}"
        logger.info(log_data, extra=self._log_extra("MACHINE_EVENT"))
        self.connection_status_changed.send(
            self, status=status, message=message
        )
