import aiohttp
import asyncio
import inspect
from typing import (
    Optional,
    cast,
    Any,
    TYPE_CHECKING,
    List,
    Callable,
    Union,
    Awaitable,
)

from ...debug import debug_log_manager, LogType
from ...shared.varset import Var, VarSet, HostnameVar
from ...core.ops import Ops
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import HttpTransport, WebSocketTransport, TransportStatus
from ..transport.validators import is_valid_hostname_or_ip
from .driver import (
    Driver,
    DriverSetupError,
    DriverPrecheckError,
    DeviceConnectionError,
    Axis,
)
from .grbl_util import (
    parse_state,
    get_grbl_setting_varsets,
    grbl_setting_re,
    CommandRequest,
    hw_info_url,
    fw_info_url,
    eeprom_info_url,
    command_url,
    upload_url,
    execute_url,
    status_url,
)

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ..models.machine import Machine


class GrblNetworkDriver(Driver):
    """
    A next-generation driver for GRBL-compatible controllers that use a
    modern file upload API and allows reading/writing device settings.
    """

    label = _("GRBL (Network)")
    subtitle = _("Connect to a GRBL-compatible device over the network")
    supports_settings = True
    reports_granular_progress = False

    def __init__(self):
        super().__init__()
        self.host = None
        self.http = None
        self.websocket = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._current_request: Optional[CommandRequest] = None
        self._cmd_lock = asyncio.Lock()

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        host = cast(str, kwargs.get("host", ""))
        if not is_valid_hostname_or_ip(host):
            raise DriverPrecheckError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname"),
                    description=_("The IP address or hostname of the device"),
                )
            ]
        )

    def setup(self, **kwargs: Any):
        host = cast(str, kwargs.get("host", ""))
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        super().setup()
        self.host = host

        self.http_base = f"http://{host}"
        self.http = HttpTransport(
            f"{self.http_base}{status_url}", receive_interval=0.5
        )
        self.http.received.connect(self.on_http_data_received)
        self.http.status_changed.connect(self.on_http_status_changed)

        self.websocket = WebSocketTransport(f"ws://{host}:81/", self.http_base)
        self.websocket.received.connect(self.on_websocket_data_received)
        self.websocket.status_changed.connect(self.on_websocket_status_changed)

    async def cleanup(self):
        self.keep_running = False
        if self._connection_task:
            self._connection_task.cancel()
        if self.websocket:
            await self.websocket.disconnect()
            self.websocket.received.disconnect(self.on_websocket_data_received)
            self.websocket.status_changed.disconnect(
                self.on_websocket_status_changed
            )
            self.websocket = None
        if self.http:
            await self.http.disconnect()
            self.http.received.disconnect(self.on_http_data_received)
            self.http.status_changed.disconnect(self.on_http_status_changed)
            self.http = None
        await super().cleanup()

    async def _get_hardware_info(self):
        url = f"{self.http_base}{hw_info_url}"
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.TX, f"GET {url}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.text()
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.RX, data.encode("utf-8")
        )
        return data

    async def _get_device_info(self):
        url = f"{self.http_base}{fw_info_url}"
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.TX, f"GET {url}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.text()
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.RX, data.encode("utf-8")
        )
        return data

    async def _get_eeprom_info(self):
        url = f"{self.http_base}{eeprom_info_url}"
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.TX, f"GET {url}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.text()
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.RX, data.encode("utf-8")
        )
        return data

    async def _send_command(self, command):
        if not self.host:
            # Raise a user-friendly error immediately if host is not configured
            raise DeviceConnectionError(
                _(
                    "Host is not configured. Please set a valid"
                    " IP address or hostname."
                )
            )

        url = f"{self.http_base}{command_url.format(command=command)}"
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.TX, f"GET {url}"
        )
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    response.raise_for_status()  # Check for 4xx/5xx errors
                    data = await response.text()
            debug_log_manager.add_entry(
                self.__class__.__name__, LogType.RX, data.encode("utf-8")
            )
            return data
        except aiohttp.ClientError as e:
            msg = _(
                "Could not connect to host '{host}'. Check the IP address"
                " and network connection."
            ).format(host=self.host)
            raise DeviceConnectionError(msg) from e

    async def _upload(self, gcode, filename):
        """
        Overrides the base GrblDriver's upload method with a standard
        multipart/form-data POST request.
        """
        form = aiohttp.FormData()
        form.add_field(
            "file", gcode, filename=filename, content_type="text/plain"
        )
        url = f"{self.http_base}{upload_url}?path=/"

        debug_log_manager.add_entry(
            self.__class__.__name__,
            LogType.TX,
            f"POST to {url} with file '{filename}' size {len(gcode)}",
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form) as response:
                response.raise_for_status()
                data = await response.text()

        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.RX, data.encode("utf-8")
        )
        return data

    async def _execute(self, filename):
        url = f"{self.http_base}{execute_url.format(filename=filename)}"
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.TX, f"GET {url}"
        )
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.text()
        debug_log_manager.add_entry(
            self.__class__.__name__, LogType.RX, data.encode("utf-8")
        )
        await session.close()
        return data

    async def connect(self):
        if not self.host:
            self._on_connection_status_changed(
                TransportStatus.DISCONNECTED, "No host configured"
            )
            return

        self.keep_running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        assert self.http and self.websocket
        while self.keep_running:
            self._on_connection_status_changed(TransportStatus.CONNECTING)
            try:
                self._log("Fetching hardware info...")
                await self._get_hardware_info()

                self._log("Fetching device info...")
                await self._get_device_info()

                self._log("Fetching EEPROM info...")
                await self._get_eeprom_info()

                self._log("Starting HTTP and WebSocket transports...")
                async with asyncio.TaskGroup() as tg:
                    tg.create_task(self.http.connect())
                    tg.create_task(self.websocket.connect())

            except DeviceConnectionError as e:
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
            except Exception as e:
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
            finally:
                if self.websocket:
                    await self.websocket.disconnect()
                if self.http:
                    await self.http.disconnect()

            self._on_connection_status_changed(TransportStatus.SLEEPING)
            await asyncio.sleep(5)

    async def run(
        self,
        ops: Ops,
        machine: "Machine",
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        if not self.host:
            raise ConnectionError("Driver not configured with a host.")
        encoder = GcodeEncoder.for_machine(machine)
        gcode, op_map = encoder.encode(ops, machine, doc)

        try:
            # For GRBL driver, we don't track individual commands
            # since we upload the entire file at once
            if on_command_done is not None:
                # Call the callback for each op to indicate completion
                for op_index in range(len(ops)):
                    result = on_command_done(op_index)
                    if inspect.isawaitable(result):
                        await result

            await self._upload(gcode, "rayforge.gcode")
            await self._execute("rayforge.gcode")
        except Exception as e:
            self._on_connection_status_changed(TransportStatus.ERROR, str(e))
            raise
        finally:
            self.job_finished.send(self)

    async def _execute_command(self, command: str) -> List[str]:
        """
        Sends a command via HTTP and waits for the full response from the
        WebSocket, including an 'ok' or 'error:'.
        """
        async with self._cmd_lock:
            if not self.websocket or not self.websocket.is_connected:
                raise DeviceConnectionError("Device is not connected.")

            request = CommandRequest(command=command)
            self._current_request = request
            try:
                # Trigger command via HTTP. We don't care about the response.
                await self._send_command(command)
                # Wait for the response to arrive on the WebSocket.
                await asyncio.wait_for(request.finished.wait(), timeout=10.0)
                return request.response_lines
            except asyncio.TimeoutError as e:
                msg = f"Command '{command}' timed out."
                raise DeviceConnectionError(msg) from e
            finally:
                self._current_request = None

    async def set_hold(self, hold: bool = True) -> None:
        await self._send_command("!" if hold else "~")

    async def cancel(self) -> None:
        # Cancel is a fire-and-forget soft reset, doesn't always
        # respond with 'ok'
        await self._send_command("%18")

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
        Jogs the machine along a specific axis using GRBL's $J command.

        Args:
            axis: The Axis enum value
            distance: The distance to jog in mm (positive or negative)
            speed: The jog speed in mm/min
        """
        assert axis.name
        axis_letter: str = axis.name.upper()
        cmd = f"$J=G91 G21 F{speed} {axis_letter}{distance}"
        await self._execute_command(cmd)

    def on_http_data_received(self, sender, data: bytes):
        pass

    def on_http_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        self._on_command_status_changed(status, message)

    def on_websocket_data_received(self, sender, data: bytes):
        source = f"{self.__class__.__name__}.WebSocket"
        debug_log_manager.add_entry(source, LogType.RX, data)
        try:
            data_str = data.decode("utf-8").strip()
        except UnicodeDecodeError:
            self._log(f"Received non-UTF8 data on WebSocket: {data!r}")
            return

        for line in data_str.splitlines():
            self._log(line)
            request = self._current_request

            # If a command is awaiting a response, collect the lines.
            if request and not request.finished.is_set():
                request.response_lines.append(line)

            # Process line for state updates, regardless of active request.
            if line.startswith("<") and line.endswith(">"):
                state = parse_state(line, self.state, self._log)
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

    def on_websocket_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        self._on_connection_status_changed(status, message)

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
        """Writes a setting by sending '$<key>=<value>'."""
        cmd = f"${key}={value}"
        await self._execute_command(cmd)

    def can_g0_with_speed(self) -> bool:
        """GRBL doesn't support speed parameter in G0 commands."""
        return False
