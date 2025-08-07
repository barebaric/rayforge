import logging
import asyncio
import serial.serialutil
from typing import Optional, Any, List, cast, TYPE_CHECKING
from ...debug import debug_log_manager, LogType
from ...shared.varset import Var, VarSet, SerialPortVar, IntVar
from ...core.ops import Ops
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import TransportStatus, SerialTransport
from .driver import Driver, DriverSetupError
from .grbl_util import (
    parse_state,
    get_grbl_setting_varsets,
    grbl_setting_re,
    CommandRequest,
)

if TYPE_CHECKING:
    from ..models.machine import Machine

logger = logging.getLogger(__name__)


class GrblNextSerialDriver(Driver):
    """
    An advanced GRBL serial driver that supports reading and writing
    device settings ($$ commands).
    """

    label = _("GRBL (Next, Serial)")
    subtitle = _("Advanced GRBL serial with settings support")
    supports_settings = True

    def __init__(self):
        super().__init__()
        self.serial_transport: Optional[SerialTransport] = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._current_request: Optional[CommandRequest] = None
        self._cmd_lock = asyncio.Lock()

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                SerialPortVar(
                    key="port",
                    label=_("Port"),
                    description=_("Serial port for the device"),
                ),
                IntVar(
                    key="baudrate",
                    label=_("Baud Rate"),
                    description=_("Connection speed in bits per second"),
                    default=115200,
                    min_val=1,
                ),
            ]
        )

    def setup(self, **kwargs: Any):
        port = cast(str, kwargs.get("port", ""))
        baudrate = kwargs.get("baudrate", 115200)

        if not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baud rate must be configured."))

        super().setup()

        self.serial_transport = SerialTransport(port, baudrate)
        self.serial_transport.received.connect(self.on_serial_data_received)
        self.serial_transport.status_changed.connect(
            self.on_serial_status_changed
        )

    async def cleanup(self):
        logger.debug("GrblNextSerialDriver cleanup initiated.")
        self.keep_running = False
        if self._connection_task:
            self._connection_task.cancel()
        if self.serial_transport:
            self.serial_transport.received.disconnect(
                self.on_serial_data_received
            )
            self.serial_transport.status_changed.disconnect(
                self.on_serial_status_changed
            )
        await super().cleanup()
        logger.debug("GrblNextSerialDriver cleanup completed.")

    async def _send_command(self, command: str):
        logger.debug(f"Sending fire-and-forget command: {command}")
        if not self.serial_transport or not self.serial_transport.is_connected:
            raise ConnectionError("Serial transport not initialized")
        payload = (command + "\n").encode("utf-8")
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
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        logger.debug("Entering _connection_loop.")
        while self.keep_running:
            self._on_connection_status_changed(TransportStatus.CONNECTING)
            logger.debug("Attempting connection…")
            try:
                assert self.serial_transport, "Transport not initialized"
                await self.serial_transport.connect()
                # Send a status report request to get initial state
                await self._send_command("?")

            except (serial.serialutil.SerialException, OSError) as e:
                logger.error(f"Connection error: {e}")
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
            except asyncio.CancelledError:
                logger.info("Connection loop cancelled.")
                break
            finally:
                if (
                    self.serial_transport
                    and self.serial_transport.is_connected
                ):
                    await self.serial_transport.disconnect()

            if not self.keep_running:
                break

            logger.debug("Connection lost. Reconnecting in 5s…")
            self._on_connection_status_changed(TransportStatus.SLEEPING)
            await asyncio.sleep(5)
        logger.debug("Leaving _connection_loop.")

    async def run(self, ops: Ops, machine: "Machine") -> None:
        encoder = GcodeEncoder.for_machine(machine)
        gcode = encoder.encode(ops, machine)

        # Split gcode into lines and send them one by one
        for line in gcode.splitlines():
            if line.strip():
                await self._execute_command(line.strip())

    async def _execute_command(self, command: str) -> List[str]:
        async with self._cmd_lock:
            if (
                not self.serial_transport
                or not self.serial_transport.is_connected
            ):
                raise ConnectionError("Serial transport not connected")

            request = CommandRequest(command=command)
            self._current_request = request
            try:
                logger.debug(f"Executing command: {command}")
                debug_log_manager.add_entry(
                    self.__class__.__name__, LogType.TX, request.payload
                )
                await self.serial_transport.send(request.payload)
                await asyncio.wait_for(request.finished.wait(), timeout=10.0)
                return request.response_lines
            except asyncio.TimeoutError as e:
                msg = f"Command '{command}' timed out."
                raise ConnectionError(msg) from e
            finally:
                self._current_request = None

    async def set_hold(self, hold: bool = True) -> None:
        await self._send_command("!" if hold else "~")

    async def cancel(self) -> None:
        # GRBL reset command (Ctrl-X) is usually sent as a byte, not a string
        if self.serial_transport:
            payload = b"\x18"
            debug_log_manager.add_entry(
                self.__class__.__name__, LogType.TX, payload
            )
            await self.serial_transport.send(payload)
        else:
            raise ConnectionError("Serial transport not initialized")

    async def home(self) -> None:
        await self._execute_command("$H")

    async def move_to(self, pos_x, pos_y) -> None:
        cmd = f"$J=G90 G21 F1500 X{float(pos_x)} Y{float(pos_y)}"
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
        debug_log_manager.add_entry(self.__class__.__name__, LogType.RX, data)
        data_str = data.decode("utf-8").strip()
        for line in data_str.splitlines():
            self._log(line)
            request = self._current_request
            if request and not request.finished.is_set():
                request.response_lines.append(line)
            if line.startswith("<") and line.endswith(">"):
                state = parse_state(line[1:-1], self.state, self._log)
                if state != self.state:
                    self.state = state
                    self._on_state_changed()
            elif line == "ok":
                self._on_command_status_changed(TransportStatus.IDLE)
                if request:
                    request.finished.set()
            elif line.startswith("error:"):
                self._on_command_status_changed(
                    TransportStatus.ERROR, message=line
                )
                if request:
                    request.finished.set()

    def on_serial_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        self._on_connection_status_changed(status, message)
