import asyncio
from typing import List, Optional, cast, Any, TYPE_CHECKING
from ...debug import debug_log_manager, LogType
from ...shared.varset import VarSet, HostnameVar, IntVar
from ...core.ops import Ops
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import TelnetTransport, TransportStatus
from .driver import Driver, DeviceStatus, DriverSetupError
from .grbl import _parse_state

if TYPE_CHECKING:
    from ..models.machine import Machine


class SmoothieDriver(Driver):
    """
    Handles Smoothie-based devices via Telnet
    """

    label = _("Smoothie")
    subtitle = _("Smoothieware via a Telnet connection")
    supports_settings = False

    def __init__(self):
        super().__init__()
        self.telnet = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._ok_event = asyncio.Event()

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                HostnameVar(
                    key="host",
                    label=_("Hostname"),
                    description=_("The IP address or hostname of the device"),
                ),
                IntVar(
                    key="port",
                    label=_("Port"),
                    description=_("The Telnet port number"),
                    default=23,
                    min_val=1,
                    max_val=65535,
                ),
            ]
        )

    def get_setting_vars(self) -> List["VarSet"]:
        return [VarSet()]

    def setup(self, **kwargs: Any):
        host = cast(str, kwargs.get("host", ""))
        port = kwargs.get("port", 23)

        if not host:
            raise DriverSetupError(
                _("Invalid hostname or IP address: '{host}'").format(host=host)
            )
        super().setup()

        # Initialize transports
        self.telnet = TelnetTransport(host, port)
        self.telnet.received.connect(self.on_telnet_data_received)
        self.telnet.status_changed.connect(self.on_telnet_status_changed)

    async def cleanup(self):
        self.keep_running = False
        if self._connection_task:
            self._connection_task.cancel()
        if self.telnet:
            await self.telnet.disconnect()
            self.telnet.received.disconnect(self.on_telnet_data_received)
            self.telnet.status_changed.disconnect(
                self.on_telnet_status_changed
            )
            self.telnet = None
        await super().cleanup()

    async def connect(self):
        self.keep_running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        while self.keep_running:
            if not self.telnet:
                self._on_connection_status_changed(
                    TransportStatus.ERROR, "Driver not configured"
                )
                await asyncio.sleep(5)
                continue

            self._on_connection_status_changed(TransportStatus.CONNECTING)
            try:
                await self.telnet.connect()
                # The transport handles the connection loop.
                # We just need to wait here until cleanup.
                while self.keep_running:
                    await self._send_and_wait(b"?", wait_for_ok=False)
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                break  # cleanup is called
            except Exception as e:
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
            finally:
                if self.telnet:
                    await self.telnet.disconnect()

            if not self.keep_running:
                break

            self._on_connection_status_changed(TransportStatus.SLEEPING)
            await asyncio.sleep(5)

    async def _send_and_wait(self, cmd: bytes, wait_for_ok: bool = True):
        if not self.telnet:
            return
        if wait_for_ok:
            self._ok_event.clear()

        debug_log_manager.add_entry(self.__class__.__name__, LogType.TX, cmd)
        await self.telnet.send(cmd)

        if wait_for_ok:
            try:
                # Set a 10s timeout to avoid deadlocks
                await asyncio.wait_for(self._ok_event.wait(), 10.0)
            except asyncio.TimeoutError as e:
                raise ConnectionError(
                    f"Command '{cmd.decode()}' not confirmed"
                ) from e

    async def run(self, ops: Ops, machine: "Machine") -> None:
        encoder = GcodeEncoder.for_machine(machine)
        gcode = encoder.encode(ops, machine)

        try:
            for line in gcode.splitlines():
                line = line.strip()
                if line:
                    await self._send_and_wait(line.encode())
        except Exception as e:
            self._on_connection_status_changed(TransportStatus.ERROR, str(e))
            raise

    async def set_hold(self, hold: bool = True) -> None:
        if hold:
            await self._send_and_wait(b"!")
        else:
            await self._send_and_wait(b"~")

    async def cancel(self) -> None:
        # Send Ctrl+C
        await self._send_and_wait(b"\x03")

    async def home(self) -> None:
        await self._send_and_wait(b"$H")

    async def move_to(self, pos_x, pos_y) -> None:
        cmd = f"G90 G0 X{float(pos_x)} Y{float(pos_y)}"
        await self._send_and_wait(cmd.encode())

    def on_telnet_data_received(self, sender, data: bytes):
        debug_log_manager.add_entry(self.__class__.__name__, LogType.RX, data)
        data_str = data.decode("utf-8")
        for line in data_str.splitlines():
            self._log(line)
            if "ok" in line:
                self._ok_event.set()
                self._on_command_status_changed(TransportStatus.IDLE)

            if not line.startswith("<") or not line.endswith(">"):
                continue
            state = _parse_state(line[1:-1], self.state, self._log)
            if state != self.state:
                self.state = state
                self._on_state_changed()

    def on_telnet_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        self._on_connection_status_changed(status, message)
        if status in [TransportStatus.DISCONNECTED, TransportStatus.ERROR]:
            if self.state.status != DeviceStatus.UNKNOWN:
                self.state.status = DeviceStatus.UNKNOWN
                self._on_state_changed()

    async def read_settings(self) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def write_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )
