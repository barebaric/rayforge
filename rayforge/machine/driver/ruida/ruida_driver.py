import logging
import asyncio
from typing import (
    Any,
    TYPE_CHECKING,
    Optional,
    Callable,
    Union,
    Awaitable,
    Dict,
    List,
)
from gettext import gettext as _

from ....context import RayforgeContext
from ....core.varset import VarSet, HostnameVar, PortVar
from ....pipeline.encoder.base import OpsEncoder, MachineCodeOpMap
from ....pipeline.encoder.gcode import GcodeEncoder
from ...transport import TransportStatus
from ..driver import (
    Driver,
    DriverSetupError,
    DriverPrecheckError,
    Axis,
    Pos,
    DeviceStatus,
)
from ...transport.validators import is_valid_hostname_or_ip
from ...transport.udp import UdpTransport
from .ruida_transport import RuidaTransport
from .ruida_client import RuidaClient

if TYPE_CHECKING:
    from ....core.doc import Doc
    from ...models.machine import Machine
    from ...models.laser import Laser


logger = logging.getLogger(__name__)


class RuidaDriver(Driver):
    """
    Driver for Ruida laser controllers using UDP protocol.

    Implements the Driver interface with unit conversion (mm ↔ µm)
    and uses RuidaClient for communication with the controller.
    """

    label = _("Ruida (UDP)")
    subtitle = _("Connect to a Ruida laser controller over UDP")
    supports_settings = False
    reports_granular_progress = False

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self.host = None
        self.port = None
        self.jog_port = None
        self._udp_transport = None
        self._ruida_transport = None
        self._jog_udp_transport = None
        self._client = None

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @property
    def resource_uri(self) -> Optional[str]:
        if self.host:
            return f"udp://{self.host}:{self.port} (jog: {self.jog_port})"
        return None

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        host = kwargs.get("host", "")
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
                    description=_(
                        "The IP address or hostname of the Ruida controller"
                    ),
                ),
                PortVar(
                    key="port",
                    label=_("Main Port"),
                    description=_(
                        "The UDP port for main commands (default: 50200)"
                    ),
                    default=50200,
                ),
                PortVar(
                    key="jog_port",
                    label=_("Jog Port"),
                    description=_(
                        "The UDP port for jog commands (default: 50207)"
                    ),
                    default=50207,
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        return GcodeEncoder(machine.dialect)

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = kwargs.get("host", "")
        port = kwargs.get("port", 50200)
        jog_port = kwargs.get("jog_port", 50207)
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        self.host = host
        self.port = port
        self.jog_port = jog_port

        self._udp_transport = UdpTransport(host, port)
        self._ruida_transport = RuidaTransport(self._udp_transport)
        self._jog_udp_transport = UdpTransport(host, jog_port)
        self._client = RuidaClient(
            self._ruida_transport, jog_transport=self._jog_udp_transport
        )

        self._ruida_transport.decoded_received.connect(self._on_data_received)
        self._ruida_transport.status_changed.connect(self._on_status_changed)

    async def cleanup(self):
        if self._ruida_transport:
            self._ruida_transport.decoded_received.disconnect(
                self._on_data_received
            )
            self._ruida_transport.status_changed.disconnect(
                self._on_status_changed
            )
        if self._client:
            await self._client.disconnect()
        if self._jog_udp_transport:
            await self._jog_udp_transport.disconnect()
        if self._ruida_transport:
            await self._ruida_transport.disconnect()
        self._jog_udp_transport = None
        self._ruida_transport = None
        self._udp_transport = None
        self._update_connection_status(TransportStatus.DISCONNECTED, "")
        await super().cleanup()

    async def _connect_implementation(self) -> None:
        if not self.host:
            self._update_connection_status(
                TransportStatus.DISCONNECTED, "No host configured"
            )
            return

        assert self._client
        try:
            await self._client.connect()
            self._update_connection_status(TransportStatus.CONNECTED, "")
            self.state.status = DeviceStatus.IDLE
            self.state_changed.send(self, state=self.state)
            logger.info(
                f"Connected to Ruida controller at {self.host}:{self.port}"
            )
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            self._update_connection_status(TransportStatus.ERROR, str(e))
            raise

    async def run(
        self,
        machine_code: Any,
        op_map: "MachineCodeOpMap",
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        gcode = str(machine_code)
        lines = [line.strip() for line in gcode.splitlines() if line.strip()]

        if on_command_done is not None:
            num_ops = 0
            if op_map and op_map.op_to_machine_code:
                num_ops = max(op_map.op_to_machine_code.keys()) + 1

            for op_index in range(num_ops):
                result = on_command_done(op_index)
                import inspect

                if inspect.isawaitable(result):
                    await result

        logger.info(
            f"Executing {len(lines)} G-code lines",
            extra=self._log_extra("USER_COMMAND"),
        )

        for line in lines:
            logger.info(line, extra=self._log_extra("USER_COMMAND"))

        self.job_finished.send(self)

    async def run_raw(self, gcode: str) -> None:
        lines = [line.strip() for line in gcode.splitlines() if line.strip()]
        for line in lines:
            logger.info(line, extra=self._log_extra("USER_COMMAND"))
        self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        assert self._client
        if hold:
            await self._client.pause_process()
        else:
            await self._client.resume_process()

    async def cancel(self) -> None:
        assert self._client
        await self._client.stop_process()

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def home(self, axes: Optional[Axis] = None) -> None:
        assert self._client
        if axes is None:
            await self._client.home_xy()
            await self._client.home_z()
        else:
            if axes & (Axis.X | Axis.Y):
                await self._client.home_xy()
            if axes & Axis.Z:
                await self._client.home_z()

    async def move_to(self, pos_x: float, pos_y: float) -> None:
        assert self._client
        x_um = int(pos_x * 1000)
        y_um = int(pos_y * 1000)
        await self._client.move_abs(x_um, y_um)

    async def select_tool(self, tool_number: int) -> None:
        pass

    async def read_settings(self) -> None:
        await asyncio.sleep(0)
        self.settings_read.send(self, settings=[])

    def get_setting_vars(self) -> List["VarSet"]:
        return [VarSet(title=_("No settings"))]

    async def write_setting(self, key: str, value: Any) -> None:
        pass

    async def clear_alarm(self) -> None:
        assert self._client
        await self._client.stop_process()

    async def set_power(self, head: "Laser", percent: float) -> None:
        assert self._client
        power_percent = percent * 100
        laser_num = head.tool_number + 1
        await self._client.set_power_immediate(laser_num, power_percent)

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        assert self._client
        for axis_name, delta in deltas.items():
            axis_lower = axis_name.lower()
            if axis_lower == "x":
                await self._client.jog_rel_x(int(delta * 1000))
            elif axis_lower == "y":
                await self._client.jog_rel_y(int(delta * 1000))

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        pass

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        self.wcs_updated.send(self, offsets={})
        return {}

    async def read_parser_state(self) -> Optional[str]:
        return None

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        self.probe_status_changed.send(self, message="Probe not supported")
        return None

    def _on_data_received(self, sender, data: bytes) -> None:
        pass

    def _on_status_changed(
        self, sender, status: TransportStatus, message: str = ""
    ) -> None:
        self._update_connection_status(status, message)

    def _update_connection_status(
        self, status: TransportStatus, message: str = ""
    ) -> None:
        self.connection_status_changed.send(
            self, status=status, message=message
        )

    @property
    def is_connected(self) -> bool:
        """Check if driver is connected."""
        return self._client is not None and self._client.is_connected
