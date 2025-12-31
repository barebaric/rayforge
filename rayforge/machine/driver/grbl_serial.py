import logging
import asyncio
import inspect
import serial.serialutil
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
from ...context import RayforgeContext
from ...core.ops import Ops
from ...core.varset import Var, VarSet, SerialPortVar, BaudrateVar
from ...pipeline.encoder.base import OpsEncoder
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
    from ..models.laser import Laser

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

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self.serial_transport: Optional[SerialTransport] = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._current_request: Optional[CommandRequest] = None
        self._cmd_lock = asyncio.Lock()
        self._command_queue: asyncio.Queue[CommandRequest] = asyncio.Queue()
        self._command_task: Optional[asyncio.Task] = None
        self._status_buffer = bytearray()
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

    @property
    def resource_uri(self) -> Optional[str]:
        if self.serial_transport and self.serial_transport.port:
            return f"serial://{self.serial_transport.port}"
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
                BaudrateVar("baudrate"),
            ]
        )

    def get_encoder(self) -> "OpsEncoder":
        """Returns a GcodeEncoder configured for the machine's dialect."""
        return GcodeEncoder(self._machine.dialect)

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
        self._update_connection_status(status, message)

    async def cleanup(self):
        logger.debug("GrblNextSerialDriver cleanup initiated.")
        self.keep_running = False
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done = None
        self._rx_buffer_count = 0
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

        if self.serial_transport:
            self.serial_transport.received.disconnect(
                self.on_serial_data_received
            )
            self.serial_transport.status_changed.disconnect(
                self.on_serial_status_changed
            )
            # Close the serial port to prevent duplicate readers/writers
            if self.serial_transport.is_connected:
                await self.serial_transport.disconnect()

        await super().cleanup()
        logger.debug("GrblNextSerialDriver cleanup completed.")

    async def _send_command(self, command: str, add_newline: bool = True):
        logger.debug(f"Sending fire-and-forget command: {command}")
        if not self.serial_transport or not self.serial_transport.is_connected:
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
        await self.serial_transport.send(payload)

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

        # Check if setup was successful (serial_transport exists)
        if not self.serial_transport:
            logger.error(
                "Cannot connect: Serial transport not initialized "
                "(check port settings)."
            )
            self._update_connection_status(
                TransportStatus.ERROR, _("Port not configured")
            )
            return

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
            self._update_connection_status(TransportStatus.CONNECTING)
            logger.debug("Attempting connection…")

            try:
                transport = self.serial_transport
                if not transport:
                    raise DriverSetupError("Transport not initialized")

                await transport.connect()
                logger.info("Connection established successfully.")
                self._update_connection_status(TransportStatus.CONNECTED)
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
                            logger.debug(
                                f"TX: {payload!r}",
                                extra={
                                    "log_category": "RAW_IO",
                                    "direction": "TX",
                                    "data": payload,
                                },
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
                self._update_connection_status(TransportStatus.ERROR, str(e))
            except asyncio.CancelledError:
                logger.info("Connection loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Unexpected error in connection loop: {e}")
                self._update_connection_status(TransportStatus.ERROR, str(e))
            finally:
                if (
                    self.serial_transport
                    and self.serial_transport.is_connected
                ):
                    logger.debug("Disconnecting transport in finally block")
                    await self.serial_transport.disconnect()

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
                        logger.debug(
                            f"TX: {request.payload!r}",
                            extra={
                                "log_category": "RAW_IO",
                                "direction": "TX",
                                "data": request.payload,
                            },
                        )
                        if self.serial_transport:
                            await self.serial_transport.send(request.payload)

                        # Wait for the response to arrive. The timeout is
                        # handled by the caller (_execute_command). This
                        # processor just waits for completion.
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
        self._rx_buffer_count = 0
        self._job_exception = None
        # Clear any old items from the queue
        while not self._sent_gcode_queue.empty():
            self._sent_gcode_queue.get_nowait()
        self._buffer_has_space.set()  # Initially, there is space

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

                op_index = (
                    machine_code_to_op_map.get(line_idx)
                    if machine_code_to_op_map
                    else None
                )
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

                    logger.debug(
                        f"TX: {command_bytes!r}",
                        extra={
                            "log_category": "RAW_IO",
                            "direction": "TX",
                            "data": command_bytes,
                        },
                    )

                    # Add to queue BEFORE sending.
                    # Fast machines can reply with 'ok' before await send()
                    # returns. If we queue after sending, the RX handler finds
                    # an empty queue and drops the 'ok', causing a deadlock.
                    self._rx_buffer_count += command_len
                    self._sent_gcode_queue.put_nowait((command_len, op_index))
                    await self.serial_transport.send(command_bytes)

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

    async def run(
        self,
        ops: Ops,
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        self._start_job(on_command_done)

        encoder = self.get_encoder()
        gcode, op_map = encoder.encode(ops, self._machine, doc)
        gcode_lines = gcode.splitlines()

        await self._stream_gcode(gcode_lines, op_map.machine_code_to_op)

    async def run_raw(self, gcode: str) -> None:
        """
        Executes a raw G-code string using the character-counting streaming
        protocol.
        """
        self._start_job()

        gcode_lines = gcode.splitlines()

        await self._stream_gcode(gcode_lines)

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
            logger.debug(
                f"TX: {payload!r}",
                extra={
                    "log_category": "RAW_IO",
                    "direction": "TX",
                    "data": payload,
                },
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
        try:
            await asyncio.wait_for(request.finished.wait(), timeout=10.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            # If the wait times out or is cancelled, we MUST ensure the
            # finished event is set. Otherwise, the _process_command_queue
            # background task will hang forever awaiting this request, holding
            # the lock and preventing any future commands from running.
            logger.warning(
                f"Command '{command}' timed out or was cancelled. "
                "Unblocking queue."
            )
            request.finished.set()
            raise
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

    async def set_power(self, head: "Laser", percent: float) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control.
            percent: Power percentage (0.0-1.0). 0 disables power.
        """
        # Get the dialect for power control commands
        dialect = self._machine.dialect

        if percent <= 0:
            # Disable power
            cmd = dialect.laser_off
        else:
            # Enable power with specified percentage
            power_abs = percent * head.max_power
            cmd = dialect.laser_on.format(power=power_abs)

        await self._execute_command(cmd)

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

    def on_serial_data_received(self, sender, data: bytes):
        """
        Primary handler for incoming serial data. Decodes, buffers, and
        delegates processing of complete messages.
        """
        logger.debug(
            f"RX: {data!r}",
            extra={"log_category": "RAW_IO", "direction": "RX", "data": data},
        )
        # Buffer bytes directly to avoid decoding errors on split characters
        self._status_buffer.extend(data)

        # Process all complete messages (ending with '\r\n') in the buffer
        while b"\r\n" in self._status_buffer:
            end_idx = self._status_buffer.find(b"\r\n") + 2
            message_bytes = self._status_buffer[:end_idx]
            self._status_buffer = self._status_buffer[end_idx:]

            try:
                message = message_bytes.decode("utf-8")
                self._process_message(message)
            except UnicodeDecodeError:
                logger.warning(
                    f"Dropped invalid UTF-8 message bytes: {message_bytes!r}"
                )

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
        logger.info(report, extra={"log_category": "MACHINE_EVENT"})
        state = parse_state(
            report, self.state, lambda message: logger.info(message)
        )

        # If a job is active, 'Idle' state between commands should be
        # reported as 'Run' to the UI.
        if self._job_running and state.status == DeviceStatus.IDLE:
            state.status = DeviceStatus.RUN

        if state != self.state:
            self.state = state
            logger.info(
                f"Device state changed: {self.state.status.name}",
                extra={"log_category": "STATE_CHANGE", "state": self.state},
            )
            self.state_changed.send(self, state=self.state)

    def _handle_general_response(self, line: str):
        """
        Handles non-status-report lines like 'ok', 'error:', welcome messages,
        or settings output.
        """
        logger.debug(f"Processing received line: {line}")

        # Basic heuristic to filter out broken/fragmented status reports that
        # missed the opening '<' due to serial corruption. Treating them as
        # general responses can confuse command handlers.
        if "Pos:" in line and "|" in line:
            logger.debug(f"Ignoring fragmented status report line: {line}")
            return

        logger.info(line, extra={"log_category": "MACHINE_EVENT"})

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
                self.command_status_changed.send(
                    self, status=TransportStatus.ERROR, message=line
                )
                logger.error(
                    f"GRBL error during job: {line}. Halting stream.",
                    extra={"log_category": "ERROR"},
                )
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
            self.command_status_changed.send(self, status=TransportStatus.IDLE)
            if request:
                logger.debug(
                    f"Command '{request.command}' completed with 'ok'"
                )
                request.finished.set()

        elif line.startswith("error:"):
            # This is a COMMAND error, not a CONNECTION error.
            self.command_status_changed.send(
                self, status=TransportStatus.ERROR, message=line
            )
            if request:
                request.finished.set()
        else:
            # This could be a welcome message, an alarm, or a setting line
            logger.debug(f"Received informational line: {line}")

    def can_g0_with_speed(self) -> bool:
        """GRBL doesn't support speed parameter in G0 commands."""
        return False

    def _update_connection_status(
        self, status: TransportStatus, message: Optional[str] = None
    ):
        log_data = f"Connection status: {status.name}"
        if message:
            log_data += f" - {message}"
        logger.info(log_data, extra={"log_category": "MACHINE_EVENT"})
        self.connection_status_changed.send(
            self, status=status, message=message
        )
