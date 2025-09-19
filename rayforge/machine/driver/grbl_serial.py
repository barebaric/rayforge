import logging
import asyncio
import serial.serialutil
from typing import Optional, Any, List, cast, TYPE_CHECKING
from ...debug import debug_log_manager, LogType
from ...shared.varset import Var, VarSet, SerialPortVar, BaudrateVar
from ...core.ops import Ops
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import TransportStatus, SerialTransport
from ..transport.serial import SerialPortPermissionError
from .driver import Driver, DriverSetupError, DeviceStatus
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


class GrblSerialDriver(Driver):
    """
    An advanced GRBL serial driver that supports reading and writing
    device settings ($$ commands).
    """

    label = _("GRBL (Serial)")
    subtitle = _("GRBL-compatible serial connection")
    supports_settings = True

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
        try:
            SerialTransport.check_serial_permissions_globally()
        except SerialPortPermissionError as e:
            raise DriverSetupError(str(e)) from e

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

    async def run(self, ops: Ops, machine: "Machine", doc: "Doc") -> None:
        self._is_cancelled = False
        self._job_running = True
        encoder = GcodeEncoder.for_machine(machine)
        gcode = encoder.encode(ops, machine, doc)

        for line in gcode.splitlines():
            if self._is_cancelled:
                logger.info("Job cancelled, stopping G-code queuing.")
                break
            if line.strip():
                request = CommandRequest(line.strip())
                await self._command_queue.put(request)

        # Check if the queue is empty immediately after adding. This handles
        # the case of an empty G-code file, ensuring _job_running is reset.
        if self._command_queue.empty():
            self._job_running = False

        logger.debug("All G-code commands queued for execution")

    async def cancel(self) -> None:
        logger.debug("Cancel command initiated.")
        self._is_cancelled = True
        self._job_running = False
        if self.serial_transport:
            payload = b"\x18"
            debug_log_manager.add_entry(
                self.__class__.__name__, LogType.TX, payload
            )
            await self.serial_transport.send(payload)
            # Clear the command queue
            while not self._command_queue.empty():
                try:
                    request = self._command_queue.get_nowait()
                    # Mark as finished to avoid hanging awaits
                    request.finished.set()
                    self._command_queue.task_done()
                except asyncio.QueueEmpty:
                    break
            logger.debug("Command queue cleared after cancel.")
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

    async def home(self) -> None:
        await self._execute_command("$H")

    async def move_to(self, pos_x, pos_y) -> None:
        cmd = f"$J=G90 G21 F1500 X{float(pos_x)} Y{float(pos_y)}"
        await self._execute_command(cmd)

    async def clear_alarm(self) -> None:
        await self._execute_command("$X")

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
        request = self._current_request

        # Append the line to the response buffer of the current command
        if request and not request.finished.is_set():
            request.response_lines.append(line)

        # Check for command completion signals
        if line == "ok":
            if request:
                logger.debug(
                    f"Command '{request.command}' completed with 'ok'"
                )
                request.finished.set()
        elif line.startswith("error:"):
            self._on_connection_status_changed(TransportStatus.ERROR, line)
            if request:
                request.finished.set()
        else:
            # This could be a welcome message, an alarm, or a setting line
            logger.debug(f"Received informational line: {line}")
