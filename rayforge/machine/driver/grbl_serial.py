import logging
import asyncio
import inspect
import serial.serialutil
from typing import (
    Optional,
    Any,
    List,
    cast,
    TYPE_CHECKING,
    Callable,
    Union,
    Awaitable,
)
from ...debug import debug_log_manager, LogType
from ...shared.varset import Var, VarSet, SerialPortVar, BaudrateVar
from ...core.ops import Ops
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import TransportStatus, SerialTransport
from ..transport.serial import SerialPortPermissionError
from .driver import (
    Driver,
    DriverSetupError,
    DeviceStatus,
    DriverPrecheckError,
    DeviceConnectionError,
    Axis,
)
from .grbl_util import (
    parse_state,
    get_grbl_setting_varsets,
    grbl_setting_re,
    CommandRequest,
)

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ..models.machine import Machine

logger = logging.getLogger(__name__)

# GRBL's serial receive buffer is 128 bytes
GRBL_RX_BUFFER_SIZE = 128


class GrblSerialDriver(Driver):
    """
    An advanced GRBL serial driver that supports reading and writing
    device settings ($$ commands).
    """

    label = _("GRBL (Serial)")
    subtitle = _("GRBL-compatible serial connection")
    supports_settings = True
    reports_granular_progress = True

    def __init__(self):
        super().__init__()
        self.serial_transport: Optional[SerialTransport] = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._current_request: Optional[CommandRequest] = None
        self._cmd_lock = asyncio.Lock()
        self._command_queue: asyncio.Queue[CommandRequest] = asyncio.Queue()
        self._command_task: Optional[asyncio.Task] = None
        self._status_buffer = ""
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None
        self._last_reported_op_index = -1

        # For GRBL character counting streaming protocol
        self._rx_buffer_count = 0
        self._sent_gcode_queue: asyncio.Queue[tuple[int, Optional[int]]] = (
            asyncio.Queue()
        )
        self._buffer_has_space = asyncio.Event()
        self._job_exception: Optional[Exception] = None

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
                BaudrateVar("baudrate"),
            ]
        )

    def setup(self, **kwargs: Any):
        port = cast(str, kwargs.get("port", ""))
        baudrate = kwargs.get("baudrate", 115200)

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

        super().setup()

        self.serial_transport = SerialTransport(port, baudrate)
        self.serial_transport.received.connect(self.on_serial_data_received)
        self.serial_transport.status_changed.connect(
            self.on_serial_status_changed
        )

    def on_serial_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        """
        Handle status changes from the serial transport.
        """
        logger.debug(
            f"Serial transport status changed: {status}, message: {message}"
        )
        self._on_connection_status_changed(status, message)

    async def cleanup(self):
        logger.debug("GrblNextSerialDriver cleanup initiated.")
        self.keep_running = False
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done = None
        self._rx_buffer_count = 0
        self._job_exception = None
        if self._connection_task:
            self._connection_task.cancel()
        if self._command_task:
            self._command_task.cancel()
        if self.serial_transport:
            self.serial_transport.received.disconnect(
                self.on_serial_data_received
            )
            self.serial_transport.status_changed.disconnect(
                self.on_serial_status_changed
            )
        await super().cleanup()
        logger.debug("GrblNextSerialDriver cleanup completed.")

    async def _send_command(self, command: str, add_newline: bool = True):
        logger.debug(f"Sending fire-and-forget command: {command}")
        if not self.serial_transport or not self.serial_transport.is_connected:
            raise ConnectionError("Serial transport not initialized")
        payload = (command + ("\n" if add_newline else "")).encode("utf-8")
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.TX, payload
        )
        await self.serial_transport.send(payload)

    async def connect(self):
        """
        Launches the connection loop as a background task and returns,
        allowing the UI to remain responsive.
        """
        logger.debug("GrblNextSerialDriver connect initiated.")
        self.keep_running = True
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done = None
        self._rx_buffer_count = 0
        self._job_exception = None
        self._sent_gcode_queue = asyncio.Queue()
        self._buffer_has_space = asyncio.Event()
        self._buffer_has_space.set()
        self._connection_task = asyncio.create_task(self._connection_loop())
        self._command_task = asyncio.create_task(self._process_command_queue())

    async def _connection_loop(self) -> None:
        logger.debug("Entering _connection_loop.")
        while self.keep_running:
            self._on_connection_status_changed(TransportStatus.CONNECTING)
            logger.debug("Attempting connection…")

            transport = self.serial_transport
            assert transport, "Transport not initialized"

            try:
                await transport.connect()
                logger.info("Connection established successfully.")
                self._on_connection_status_changed(TransportStatus.CONNECTED)
                logger.debug(f"is_connected: {transport.is_connected}")

                logger.debug("Sending initial status query")
                await self._send_command("?", add_newline=False)
                logger.debug(
                    "Connection established. Starting status polling."
                )
                while transport.is_connected and self.keep_running:
                    async with self._cmd_lock:
                        try:
                            logger.debug("Sending status poll")
                            payload = b"?"
                            debug_log_manager.add_entry(
                                self.__class__.__name__, LogType.TX, payload
                            )
                            if self.serial_transport:
                                await self.serial_transport.send(payload)
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
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
            except asyncio.CancelledError:
                logger.info("Connection loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in connection loop: {e}")
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
            finally:
                if transport and transport.is_connected:
                    logger.debug("Disconnecting transport in finally block")
                    await transport.disconnect()

            if not self.keep_running:
                break

            logger.debug("Connection lost. Reconnecting in 5s…")
            self._on_connection_status_changed(TransportStatus.SLEEPING)
            await asyncio.sleep(5)

        logger.debug("Leaving _connection_loop.")

    async def _process_command_queue(self) -> None:
        logger.debug("Entering _process_command_queue.")
        while self.keep_running:
            try:
                request = await self._command_queue.get()
                async with self._cmd_lock:
                    if (
                        not self.serial_transport
                        or not self.serial_transport.is_connected
                        or self._is_cancelled
                    ):
                        logger.warning(
                            "Cannot process command: Serial transport not "
                            "connected or job is cancelled. Dropping command."
                        )
                        # Mark as done so get() doesn't block forever
                        if not request.finished.is_set():
                            request.finished.set()
                        self._command_queue.task_done()
                        continue

                    self._current_request = request
                    try:
                        logger.debug(f"Executing command: {request.command}")
                        debug_log_manager.add_entry(
                            self.__class__.__name__,
                            LogType.TX,
                            request.payload,
                        )
                        if self.serial_transport:
                            await self.serial_transport.send(request.payload)

                        # Wait for the response to arrive. The timeout is
                        # handled by the caller (_execute_command). This
                        # processor just waits for completion.
                        await request.finished.wait()

                    except ConnectionError as e:
                        logger.error(f"Connection error during command: {e}")
                        self._on_connection_status_changed(
                            TransportStatus.ERROR,
                            str(e),
                        )
                    finally:
                        self._current_request = None
                        self._command_queue.task_done()
                        # If a job was running and the queue is now empty,
                        # the job is finished.
                        if self._job_running and self._command_queue.empty():
                            logger.debug(
                                "Job finished: command queue is empty."
                            )
                            self._job_running = False
                            self._on_command_done = None
                            self.job_finished.send(self)

                # Release lock briefly to allow status polling
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("Command queue processing cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in command queue: {e}")
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
        logger.debug("Leaving _process_command_queue.")

    async def run(
        self,
        ops: Ops,
        machine: "Machine",
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        self._is_cancelled = False
        self._job_running = True
        self._on_command_done = on_command_done
        self._last_reported_op_index = -1
        self._rx_buffer_count = 0
        self._job_exception = None
        # Clear any old items from the queue
        while not self._sent_gcode_queue.empty():
            self._sent_gcode_queue.get_nowait()
        self._buffer_has_space.set()  # Initially, there is space

        encoder = GcodeEncoder.for_machine(machine)
        gcode, op_map = encoder.encode(ops, machine, doc)
        gcode_lines = gcode.splitlines()

        logger.debug(
            f"Starting GRBL streaming job with {len(gcode_lines)} lines."
        )

        try:
            for line_idx, line in enumerate(gcode_lines):
                if self._is_cancelled or self._job_exception:
                    logger.info(
                        "Job cancelled or errored, stopping G-code sending."
                    )
                    break

                line = line.strip()
                if not line:
                    continue

                op_index = op_map.gcode_to_op.get(line_idx)
                # Command is line + newline character
                command_bytes = (line + "\n").encode("utf-8")
                command_len = len(command_bytes)

                # Wait until there is enough space in the buffer
                while (
                    self._rx_buffer_count + command_len > GRBL_RX_BUFFER_SIZE
                ):
                    self._buffer_has_space.clear()
                    await self._buffer_has_space.wait()
                    if self._is_cancelled or self._job_exception:
                        raise asyncio.CancelledError(
                            "Job cancelled while waiting for buffer space"
                        )

                async with self._cmd_lock:
                    if (
                        not self.serial_transport
                        or not self.serial_transport.is_connected
                    ):
                        raise ConnectionError(
                            "Serial transport disconnected during job."
                        )

                    debug_log_manager.add_entry(
                        self.__class__.__name__, LogType.TX, command_bytes
                    )
                    await self.serial_transport.send(command_bytes)
                    self._rx_buffer_count += command_len
                    await self._sent_gcode_queue.put((command_len, op_index))

            # Wait for all sent commands to be acknowledged
            if not self._is_cancelled and not self._job_exception:
                logger.debug(
                    "All G-code sent. Waiting for all 'ok' responses."
                )
                await self._sent_gcode_queue.join()
                logger.debug("All 'ok' responses received.")

            if self._job_exception:
                raise self._job_exception

        except (asyncio.CancelledError, ConnectionError) as e:
            logger.warning(f"Job interrupted: {e!r}")
            # If not cancelled explicitly, send a cancel command to be safe
            if not self._is_cancelled:
                await self.cancel()
        finally:
            self._job_running = False
            self._on_command_done = None
            # Check if not cancelled, because cancel() already sends this.
            if not self._is_cancelled:
                self.job_finished.send(self)
            logger.debug("G-code streaming finished.")

    async def cancel(self) -> None:
        logger.debug("Cancel command initiated.")
        job_was_running = self._job_running
        self._is_cancelled = True
        self._job_running = False
        self._on_command_done = None

        # Unblock the run loop if it's waiting
        self._buffer_has_space.set()

        if self.serial_transport:
            payload = b"\x18"
            debug_log_manager.add_entry(
                self.__class__.__name__, LogType.TX, payload
            )
            await self.serial_transport.send(payload)
            # Clear the command queue for single commands
            while not self._command_queue.empty():
                try:
                    request = self._command_queue.get_nowait()
                    # Mark as finished to avoid hanging awaits
                    request.finished.set()
                    self._command_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            logger.debug("Command queue cleared after cancel.")

            # Clear the streaming queue
            self._rx_buffer_count = 0
            while not self._sent_gcode_queue.empty():
                try:
                    self._sent_gcode_queue.get_nowait()
                    self._sent_gcode_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            logger.debug("Streaming queue cleared after cancel.")

            if job_was_running:
                self.job_finished.send(self)
        else:
            raise ConnectionError("Serial transport not initialized")

    async def _execute_command(self, command: str) -> List[str]:
        self._is_cancelled = False
        request = CommandRequest(command)
        await self._command_queue.put(request)
        await asyncio.wait_for(request.finished.wait(), timeout=10.0)
        return request.response_lines

    async def set_hold(self, hold: bool = True) -> None:
        self._is_cancelled = False
        await self._send_command("!" if hold else "~", add_newline=False)

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
        if axes is None:
            await self._execute_command("$H")
            return

        # Handle multiple axes - home them one by one
        for axis in Axis:
            if axes & axis:
                assert axis.name
                axis_letter: str = axis.name.upper()
                cmd = f"$H{axis_letter}"
                await self._execute_command(cmd)

    async def move_to(self, pos_x, pos_y) -> None:
        cmd = f"$J=G90 G21 F1500 X{float(pos_x)} Y{float(pos_y)}"
        await self._execute_command(cmd)

    async def select_tool(self, tool_number: int) -> None:
        """Sends a tool change command for the given tool number."""
        cmd = f"T{tool_number}"
        await self._execute_command(cmd)

    async def clear_alarm(self) -> None:
        await self._execute_command("$X")

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """GRBL supports jogging for all axes."""
        return True

    async def jog(self, axis: Axis, distance: float, speed: int) -> None:
        """
        Jogs the machine along a specific axis or combination of axes
        using GRBL's $J command.

        Args:
            axis: The Axis enum value or combination of axes using
                  binary operators (e.g. Axis.X|Axis.Y)
            distance: The distance to jog in mm (positive or negative)
            speed: The jog speed in mm/min
        """
        # Build the command with all specified axes
        cmd_parts = [f"$J=G91 G21 F{speed}"]

        # Add each axis component to the command
        for single_axis in Axis:
            if axis & single_axis:
                assert single_axis.name
                axis_letter = single_axis.name.upper()
                cmd_parts.append(f"{axis_letter}{distance}")

        cmd = " ".join(cmd_parts)
        await self._execute_command(cmd)

    def get_setting_vars(self) -> List["VarSet"]:
        return get_grbl_setting_varsets()

    async def read_settings(self) -> None:
        response_lines = await self._execute_command("$$")
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
        self._on_settings_read(result)

    async def write_setting(self, key: str, value: Any) -> None:
        cmd = f"${key}={value}"
        await self._execute_command(cmd)

    def on_serial_data_received(self, sender, data: bytes):
        """
        Primary handler for incoming serial data. Decodes, buffers, and
        delegates processing of complete messages.
        """
        debug_log_manager.add_entry(self.__class__.__name__, LogType.RX, data)
        try:
            data_str = data.decode("utf-8")
        except UnicodeDecodeError:
            logger.warning("Received invalid UTF-8 data, ignoring.")
            return

        self._status_buffer += data_str
        # Process all complete messages (ending with '\r\n') in the buffer
        while "\r\n" in self._status_buffer:
            end_idx = self._status_buffer.find("\r\n") + 2
            message = self._status_buffer[:end_idx]
            self._status_buffer = self._status_buffer[end_idx:]
            self._process_message(message)

    def _process_message(self, message: str):
        """
        Routes a complete message to the appropriate handler based on its
        content.
        """
        stripped_message = message.strip()
        if not stripped_message:
            return

        # Status reports are frequent and start with '<'
        if stripped_message.startswith("<") and stripped_message.endswith(">"):
            self._handle_status_report(stripped_message)
        else:
            # Handle other responses line by line (e.g., 'ok', 'error:')
            for line in message.strip().splitlines():
                if line:  # Ensure we don't process empty lines
                    self._handle_general_response(line)

    def _handle_status_report(self, report: str):
        """
        Parses a GRBL status report (e.g., '<Idle|WPos:0,0,0|...>')
        and updates the device state.
        """
        logger.debug(f"Processing received status message: {report}")
        self._log(report)
        state = parse_state(report, self.state, self._log)

        # If a job is active, 'Idle' state between commands should be
        # reported as 'Run' to the UI.
        if self._job_running and state.status == DeviceStatus.IDLE:
            state.status = DeviceStatus.RUN

        if state != self.state:
            self.state = state
            self._on_state_changed()

    def _handle_general_response(self, line: str):
        """
        Handles non-status-report lines like 'ok', 'error:', welcome messages,
        or settings output.
        """
        logger.debug(f"Processing received line: {line}")
        self._log(line)

        # Logic for character-counting streaming protocol during a job
        if self._job_running:
            if line == "ok":
                try:
                    # Get the length and op_index of the command that finished
                    (
                        command_len,
                        op_index,
                    ) = self._sent_gcode_queue.get_nowait()
                    self._rx_buffer_count -= command_len
                    self._sent_gcode_queue.task_done()
                    self._buffer_has_space.set()  # Signal new buffer space

                    # If this command was part of a job, update progress.
                    if self._on_command_done and op_index is not None:
                        # Fire callbacks for all ops from the last reported
                        # one up to this one. This ensures ops with no
                        # G-code are also reported.
                        for i in range(
                            self._last_reported_op_index + 1, op_index + 1
                        ):
                            try:
                                logger.debug(
                                    "GrblSerialDriver: Firing on_command_done"
                                    f" for op_index {i}"
                                )
                                result = self._on_command_done(i)
                                if inspect.isawaitable(result):
                                    asyncio.ensure_future(result)
                            except Exception as e:
                                logger.error(
                                    "Error in on_command_done callback",
                                    exc_info=e,
                                )
                        self._last_reported_op_index = op_index
                except asyncio.QueueEmpty:
                    logger.warning(
                        "Received 'ok' during job, but sent gcode queue "
                        "was empty. Ignoring."
                    )
            elif line.startswith("error:"):
                self._on_command_status_changed(TransportStatus.ERROR, line)
                logger.error(f"GRBL error during job: {line}. Halting stream.")
                self._job_exception = DeviceConnectionError(
                    f"GRBL error: {line}"
                )
                self._buffer_has_space.set()  # Unblock run loop to terminate
                asyncio.create_task(self.cancel())  # Soft-reset the device
            return  # Do not process further for single commands

        # Logic for single, interactive commands (not during a job)
        request = self._current_request

        # Append the line to the response buffer of the current command
        if request and not request.finished.is_set():
            request.response_lines.append(line)

        # Check for command completion signals
        if line == "ok":
            self._on_command_status_changed(TransportStatus.IDLE)
            if request:
                logger.debug(
                    f"Command '{request.command}' completed with 'ok'"
                )
                request.finished.set()

        elif line.startswith("error:"):
            # This is a COMMAND error, not a CONNECTION error.
            self._on_command_status_changed(TransportStatus.ERROR, line)
            if request:
                request.finished.set()
        else:
            # This could be a welcome message, an alarm, or a setting line
            logger.debug(f"Received informational line: {line}")

    def can_g0_with_speed(self) -> bool:
        """GRBL doesn't support speed parameter in G0 commands."""
        return False
