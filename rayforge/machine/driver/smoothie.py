import asyncio
import inspect
from typing import (
    List,
    Optional,
    cast,
    Any,
    TYPE_CHECKING,
    Callable,
    Union,
    Awaitable,
)
from ...context import RayforgeContext
from ...core.ops import Ops
from ...debug import LogType
from ...pipeline.encoder.gcode import GcodeEncoder
from ...shared.varset import VarSet, HostnameVar, IntVar
from ..transport import TelnetTransport, TransportStatus
from ..transport.validators import is_valid_hostname_or_ip
from .driver import (
    Driver,
    DeviceStatus,
    DriverSetupError,
    DriverPrecheckError,
    Axis,
)
from .grbl_util import parse_state

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ..models.machine import Machine
    from ..models.laser import Laser


class SmoothieDriver(Driver):
    """
    Handles Smoothie-based devices via Telnet
    """

    label = _("Smoothie")
    subtitle = _("Smoothieware via a Telnet connection")
    supports_settings = False
    reports_granular_progress = True

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self.telnet = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._ok_event = asyncio.Event()

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        """Checks if the hostname is a valid format."""
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
            raise DriverSetupError(_("Hostname must be configured."))
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

        self._context.debug_log_manager.add_entry(
            self.__class__.__name__, LogType.TX, cmd
        )
        await self.telnet.send(cmd)

        if wait_for_ok:
            try:
                # Set a 10s timeout to avoid deadlocks
                await asyncio.wait_for(self._ok_event.wait(), 10.0)
            except asyncio.TimeoutError as e:
                raise ConnectionError(
                    f"Command '{cmd.decode()}' not confirmed"
                ) from e

    async def run(
        self,
        ops: Ops,
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        encoder = GcodeEncoder.for_machine(self._machine)
        gcode, op_map = encoder.encode(ops, self._machine, doc)
        gcode_lines = gcode.splitlines()

        try:
            for op_index in range(len(ops)):
                # Find all g-code lines for this specific op_index
                line_indices = op_map.op_to_gcode.get(op_index, [])
                if not line_indices:
                    # If an op generates no g-code, still report it as done.
                    if on_command_done:
                        result = on_command_done(op_index)
                        if inspect.isawaitable(result):
                            await result
                    continue

                for line_idx in sorted(line_indices):
                    line = gcode_lines[line_idx].strip()
                    if line:
                        await self._send_and_wait(line.encode())

                # After all lines for this op are sent and confirmed,
                # fire the callback.
                if on_command_done:
                    result = on_command_done(op_index)
                    if inspect.isawaitable(result):
                        await result

        except Exception as e:
            self._on_connection_status_changed(TransportStatus.ERROR, str(e))
            raise
        finally:
            self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        if hold:
            await self._send_and_wait(b"!")
        else:
            await self._send_and_wait(b"~")

    async def cancel(self) -> None:
        # Send Ctrl+C
        await self._send_and_wait(b"\x03")

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        """Smoothie supports homing for all axes."""
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
            await self._send_and_wait(b"$H")
            return

        # Handle multiple axes - home them one by one
        for axis in Axis:
            if axes & axis:
                assert axis.name
                axis_letter: str = axis.name.upper()
                cmd = f"G28 {axis_letter}0"
                await self._send_and_wait(cmd.encode())

    async def move_to(self, pos_x, pos_y) -> None:
        cmd = f"G90 G0 X{float(pos_x)} Y{float(pos_y)}"
        await self._send_and_wait(cmd.encode())

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """Smoothie supports jogging for all axes."""
        return True

    async def jog(self, axis: Axis, distance: float, speed: int) -> None:
        """
        Jogs the machine along a specific axis using G91 incremental mode.

        Args:
            axis: The Axis enum value
            distance: The distance to jog in mm (positive or negative)
            speed: The jog speed in mm/min
        """
        assert axis.name
        axis_letter = axis.name.upper()
        cmd = f"G91 G0 F{speed} {axis_letter}{distance}"
        await self._send_and_wait(cmd.encode())

    async def select_tool(self, tool_number: int) -> None:
        """Sends a tool change command for the given tool number."""
        cmd = f"T{tool_number}"
        await self._send_and_wait(cmd.encode())

    async def clear_alarm(self) -> None:
        await self._send_and_wait(b"M999")

    async def set_power(self, head: "Laser", percent: int) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control.
            percent: Power percentage (0-100). 0 disables power.
        """
        # Get the dialect for power control commands
        dialect = self._machine.dialect

        if percent <= 0:
            # Disable power
            cmd = dialect.laser_off
        else:
            # Enable power with specified percentage
            power_abs = (percent / 100.0) * head.max_power
            power_val = dialect.format_laser_power(power_abs)
            cmd = dialect.laser_on.format(power=power_val)

        await self._send_and_wait(cmd.encode("utf-8"))

    def on_telnet_data_received(self, sender, data: bytes):
        self._context.debug_log_manager.add_entry(
            self.__class__.__name__, LogType.RX, data
        )
        data_str = data.decode("utf-8")
        for line in data_str.splitlines():
            self._log(line)
            if "ok" in line:
                self._ok_event.set()
                self._on_command_status_changed(TransportStatus.IDLE)

            if not line.startswith("<") or not line.endswith(">"):
                continue
            state = parse_state(line[1:-1], self.state, self._log)
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

    def can_g0_with_speed(self) -> bool:
        """Smoothie supports speed parameter in G0 commands."""
        return True
