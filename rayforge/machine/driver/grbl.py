import re
import asyncio
import aiohttp
from copy import copy
from typing import Callable, List, Optional, cast, Any, TYPE_CHECKING
from ...debug import debug_log_manager, LogType
from ...shared.varset import VarSet, HostnameVar
from ...core.ops import Ops
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import HttpTransport, WebSocketTransport, TransportStatus
from ..transport.validators import is_valid_hostname_or_ip
from .driver import (
    Driver,
    DeviceStatus,
    DeviceState,
    Pos,
    DriverSetupError,
    DeviceConnectionError,
)

if TYPE_CHECKING:
    from ..models.machine import Machine


hw_info_url = "/command?plain=%5BESP420%5D&PAGEID="
fw_info_url = "/command?plain=%5BESP800%5D&PAGEID="
eeprom_info_url = "/command?plain=%5BESP400%5D&PAGEID="
command_url = "/command?commandText={command}&PAGEID="
upload_url = "/upload"
upload_list_url = "/upload?path=/&PAGEID=0"
execute_url = "/command?commandText=%5BESP220%5D/{filename}"
status_url = command_url.format(command="?")

pos_re = re.compile(r":(\d+\.\d+),(\d+\.\d+),(\d+\.\d+)")
fs_re = re.compile(r"FS:(\d+),(\d+)")


def _parse_pos_triplet(pos) -> Optional[Pos]:
    match = pos_re.search(pos)
    if not match:
        return None
    pos = tuple(float(i) for i in match.groups())
    if len(pos) != 3:
        return None
    return pos


def _parse_state(
    state_str: str, default: DeviceState, logger: Callable
) -> DeviceState:
    state = copy(default)
    try:
        status, *attribs = state_str.split("|")
        status = status.split(":")[0]
    except ValueError:
        return state

    try:
        state.status = DeviceStatus[status.upper()]
    except KeyError:
        logger(message=f"device sent an unupported status: {status}")

    for attrib in attribs:
        if attrib.startswith("MPos:"):
            state.machine_pos = _parse_pos_triplet(attrib) or state.machine_pos
        elif attrib.startswith("WPos:"):
            state.work_pos = _parse_pos_triplet(attrib) or state.work_pos
        elif attrib.startswith("FS:"):
            try:
                match = fs_re.match(attrib)
                if not match:
                    continue
                fs = [int(i) for i in match.groups()]
                state.feed_rate = int(fs[0])
            except (ValueError, IndexError):
                pass
    return state


class GrblDriver(Driver):
    label = _("GRBL (Network)")
    subtitle = _("Connect to a GRBL-compatible device over the network")
    supports_settings = False

    def __init__(self):
        super().__init__()
        self.host = None
        self.http = None
        self.websocket = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None

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

    def get_setting_vars(self) -> List["VarSet"]:
        return [VarSet()]

    def setup(self, **kwargs: Any):
        host = cast(str, kwargs.get("host", ""))
        if not is_valid_hostname_or_ip(host):
            raise DriverSetupError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )

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
                    "Host is not configured."
                    " Please set a valid IP address or hostname."
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
                "Could not connect to host '{host}'."
                " Check the IP address and network connection."
            ).format(host=self.host)
            raise DeviceConnectionError(msg) from e

    async def _upload(self, gcode, filename):
        form = aiohttp.FormData([])
        form.add_field("path", "/")
        form.add_field(f"/{filename}S", str(len(gcode)))
        form.add_field("myfile[]", gcode, filename=filename)
        url = f"{self.http_base}{upload_url}"
        debug_log_manager.add_entry(
            self.__class__.__name__,
            LogType.TX,
            f"POST to {url} with file '{filename}' size {len(gcode)}",
        )
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=form) as response:
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

    async def run(self, ops: Ops, machine: "Machine") -> None:
        if not self.host:
            raise ConnectionError("Driver not configured with a host.")
        encoder = GcodeEncoder.for_machine(machine)
        gcode = encoder.encode(ops, machine)

        try:
            await self._upload(gcode, "rayforge.gcode")
            await self._execute("rayforge.gcode")
        except Exception as e:
            self._on_connection_status_changed(TransportStatus.ERROR, str(e))
            raise

    async def set_hold(self, hold: bool = True) -> None:
        if not self.host:
            return
        if hold:
            await self._send_command("!")
        else:
            await self._send_command("~")

    async def cancel(self) -> None:
        if not self.host:
            return
        await self._send_command("%18")

    async def home(self) -> None:
        if not self.host:
            return
        await self._send_command("$H")

    async def move_to(self, pos_x, pos_y) -> None:
        if not self.host:
            return
        cmd = f"$J=G90 G21 F1500 X{float(pos_x)} Y{float(pos_y)}"
        await self._send_command(cmd)

    def on_http_data_received(self, sender, data: bytes):
        pass

    def on_http_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        self._on_command_status_changed(status, message)

    def on_websocket_data_received(self, sender, data: bytes):
        source = f"{self.__class__.__name__}.WebSocket"
        debug_log_manager.add_entry(source, LogType.RX, data)
        data_str = data.decode("utf-8")
        for line in data_str.splitlines():
            self._log(line)
            if not line.startswith("<") or not line.endswith(">"):
                continue
            state = _parse_state(line[1:-1], self.state, self._log)
            if state != self.state:
                self.state = state
                self._on_state_changed()

    def on_websocket_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        self._on_connection_status_changed(status, message)

    async def read_settings(self) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def write_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )
