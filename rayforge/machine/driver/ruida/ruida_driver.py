import logging
import asyncio
import inspect
from dataclasses import replace
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
from ....pipeline.encoder.base import OpsEncoder, EncodedOutput
from ...models.coordinate_system import CoordinateSystem
from ...transport import TransportStatus
from ...transport.validators import is_valid_hostname_or_ip
from ...transport.udp import UdpTransport
from ..driver import (
    Driver,
    DriverSetupError,
    DriverPrecheckError,
    DriverMaturity,
    Axis,
    Pos,
    DeviceStatus,
)
from .ruida_client import RuidaClient
from .ruida_encoder import RuidaEncoder
from .ruida_transport import RuidaTransport

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
    uses_gcode = False
    maturity = DriverMaturity.KNOWN_BUGGY
    CONNECTION_TIMEOUT = 2.0
    RECONNECT_INTERVAL = 5.0
    KEEPALIVE_INTERVAL = 1.0
    POSITION_POLL_INTERVAL = 0.5
    RESPONSE_PORT = 40200

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self.host = None
        self.port = None
        self.jog_port = None
        self._udp_transport = None
        self._ruida_transport = None
        self._jog_udp_transport = None
        self._client = None
        self._response_received = asyncio.Event()
        self._connection_task: Optional[asyncio.Task] = None
        self._keep_running = False
        self._is_connected = False

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @property
    def supported_wcs(self) -> List[str]:
        if not self._client:
            return [self.machine_space_wcs]
        return list(self._client.ref_points)

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
        return RuidaEncoder()

    def _setup_implementation(self, **kwargs: Any) -> None:
        host = kwargs.get("host", "")
        port = kwargs.get("port", 50200)
        jog_port = kwargs.get("jog_port", 50207)
        if not host:
            raise DriverSetupError(_("Hostname must be configured."))

        self.host = host
        self.port = port
        self.jog_port = jog_port

        self._udp_transport = UdpTransport(
            host, port, local_port=self.RESPONSE_PORT
        )
        self._ruida_transport = RuidaTransport(self._udp_transport)
        self._jog_udp_transport = UdpTransport(host, jog_port)
        self._jog_ruida_transport = RuidaTransport(self._jog_udp_transport)
        self._client = RuidaClient(
            self._ruida_transport,
            jog_transport=self._jog_ruida_transport,
        )

        self._client.state_changed.connect(self._on_state_changed)
        self._ruida_transport.status_changed.connect(self._on_status_changed)
        self._client.position_updated.connect(self._on_position_updated)

        self._init_coordinate_systems()

    def _init_coordinate_systems(self) -> None:
        """
        Initialize the machine's coordinate systems to match the
        Ruida controller's ref point model (MACHINE, REF0, REF1).
        """
        m = self._machine
        supported = self.supported_wcs
        existing = m.coordinate_systems

        new_systems = {}
        for name in supported:
            if name in existing:
                new_systems[name] = existing[name]
            else:
                new_systems[name] = CoordinateSystem(name=name)

        m.coordinate_systems = new_systems
        if m.active_wcs not in new_systems:
            m.active_wcs = supported[0]

    async def cleanup(self):
        self._keep_running = False
        self._is_connected = False

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            self._connection_task = None

        if self._ruida_transport:
            self._ruida_transport.status_changed.disconnect(
                self._on_status_changed
            )
        if self._client:
            self._client.state_changed.disconnect(self._on_state_changed)
            self._client.position_updated.disconnect(self._on_position_updated)
            await self._client.disconnect()
        if self._jog_udp_transport:
            await self._jog_udp_transport.disconnect()
        if self._ruida_transport:
            await self._ruida_transport.disconnect()
        self._jog_udp_transport = None
        self._ruida_transport = None
        self._udp_transport = None
        self._client = None
        self._update_connection_status(TransportStatus.DISCONNECTED, "")
        await super().cleanup()

    async def _connect_implementation(self) -> None:
        if not self.host:
            self._update_connection_status(
                TransportStatus.DISCONNECTED, "No host configured"
            )
            return

        if self._connection_task and not self._connection_task.done():
            logger.warning("Connect called with active connection task")
            return

        self._keep_running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        logger.debug("Entering Ruida connection loop")
        while self._keep_running:
            self._update_connection_status(TransportStatus.CONNECTING)
            self._is_connected = False

            try:
                if not self._client:
                    raise DriverSetupError("Client not initialized")

                await self._client.connect()

                self._response_received.clear()
                await self._client.keep_alive()

                try:
                    await asyncio.wait_for(
                        self._response_received.wait(),
                        timeout=self.CONNECTION_TIMEOUT,
                    )
                except asyncio.TimeoutError:
                    self._update_connection_status(
                        TransportStatus.ERROR,
                        _("No response from controller"),
                    )
                    await self._disconnect_transports()
                    self._update_connection_status(TransportStatus.SLEEPING)
                    await asyncio.sleep(self.RECONNECT_INTERVAL)
                    continue

                self._is_connected = True
                self._update_connection_status(TransportStatus.CONNECTED, "")
                self.state.status = DeviceStatus.IDLE
                self.state_changed.send(self, state=self.state)

                logger.info(
                    f"Connected to Ruida controller "
                    f"at {self.host}:{self.port}",
                    extra=self._log_extra("MACHINE_EVENT"),
                )

                asyncio.create_task(
                    self._fetch_card_info(),
                    name="ruida-fetch-card-info",
                )

                last_poll_time = 0.0
                last_ref_poll_time = 0.0

                while self._keep_running and self._is_connected:
                    current_time = asyncio.get_event_loop().time()

                    if (
                        current_time - last_poll_time
                        >= self.POSITION_POLL_INTERVAL
                    ):
                        self._response_received.clear()
                        await self._poll_position()
                        last_poll_time = current_time

                        try:
                            await asyncio.wait_for(
                                self._response_received.wait(),
                                timeout=self.CONNECTION_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            logger.warning(
                                "Controller stopped responding, reconnecting",
                                extra=self._log_extra("MACHINE_EVENT"),
                            )
                            self._is_connected = False
                            await self._disconnect_transports()
                            break

                    if current_time - last_ref_poll_time >= 2.0:
                        await self._poll_ref_point_mode()
                        last_ref_poll_time = current_time

                    await asyncio.sleep(self.KEEPALIVE_INTERVAL)

            except asyncio.CancelledError:
                logger.debug("Connection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self._update_connection_status(TransportStatus.ERROR, str(e))
                await self._disconnect_transports()

            if self._keep_running:
                self._update_connection_status(TransportStatus.SLEEPING)
                await asyncio.sleep(self.RECONNECT_INTERVAL)

        logger.debug("Exiting Ruida connection loop")

    async def _disconnect_transports(self) -> None:
        if self._client:
            try:
                await self._client.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting client: {e}")
        if self._jog_udp_transport:
            try:
                await self._jog_udp_transport.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting jog transport: {e}")
        if self._ruida_transport:
            try:
                await self._ruida_transport.disconnect()
            except Exception as e:
                logger.debug(f"Error disconnecting ruida transport: {e}")

    async def _poll_position(self) -> None:
        """Poll current position from controller."""
        if not self._client or not self._is_connected:
            return

        try:
            logger.debug("Polling position from controller")
            await self._client.get_position()
        except Exception as e:
            logger.debug(f"Error polling position: {e}")

    async def _poll_ref_point_mode(self) -> None:
        """Poll current ref point mode from controller."""
        if not self._client or not self._is_connected:
            return

        try:
            mode = await self._client.get_ref_point_mode()
            if mode and mode != self._machine.active_wcs:
                logger.debug(f"Ref point mode changed: {mode}")
                self._machine.set_active_wcs(mode)
        except Exception as e:
            logger.debug(f"Error polling ref point mode: {e}")

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        binary_data = encoded.driver_data.get("binary", b"")
        text_lines = [
            line.strip() for line in encoded.text.splitlines() if line.strip()
        ]
        op_map = encoded.op_map

        if on_command_done is not None:
            num_ops = 0
            if op_map and op_map.op_to_machine_code:
                num_ops = max(op_map.op_to_machine_code.keys()) + 1

            for op_index in range(num_ops):
                result = on_command_done(op_index)
                if inspect.isawaitable(result):
                    await result

        logger.info(
            f"Executing {len(text_lines)} commands",
            extra=self._log_extra("USER_COMMAND"),
        )

        for line in text_lines:
            logger.info(line, extra=self._log_extra("USER_COMMAND"))

        if binary_data and self._client:
            CHUNK_SIZE = 1024
            for i in range(0, len(binary_data), CHUNK_SIZE):
                chunk = binary_data[i : i + CHUNK_SIZE]
                await self._client.send_command(chunk)

        self.job_finished.send(self)

    async def run_raw(self, machine_code: str) -> None:
        """
        Ruida controllers use binary protocol, not text-based machine code.

        This method logs a warning and does nothing. Use run() with
        properly encoded Ruida binary data instead.
        """
        if machine_code and machine_code.strip():
            logger.warning(
                "Ruida controllers do not support text-based machine code. "
                "Use run() with EncodedOutput instead."
            )
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
            logger.info("Home All", extra=self._log_extra("MACHINE_EVENT"))
        else:
            cmd_parts = []
            if axes & (Axis.X | Axis.Y):
                cmd_parts.append("XY")
            if axes & Axis.Z:
                cmd_parts.append("Z")
            cmd_name = f"Home {'/'.join(cmd_parts)}"
            logger.info(cmd_name, extra=self._log_extra("MACHINE_EVENT"))
        await self._rapid_move_to(0, 0)

    async def move_to(self, pos_x: float, pos_y: float) -> None:
        assert self._client
        logger.info(
            f"move_to x={pos_x:.2f} y={pos_y:.2f}",
            extra=self._log_extra("MACHINE_EVENT"),
        )
        x_um = int(pos_x * 1000)
        y_um = int(pos_y * 1000)
        await self._rapid_move_to(x_um, y_um)

    async def _rapid_move_to(self, target_x: int, target_y: int) -> None:
        assert self._client
        cur_x = await self._client._read_memory_wait(0x0421)
        cur_y = await self._client._read_memory_wait(0x0431)
        dx = target_x - (cur_x or 0)
        dy = target_y - (cur_y or 0)
        if dx != 0:
            await self._client.rapid_move_axis(0x00, dx)
        if dy != 0:
            await self._client.rapid_move_axis(0x01, dy)

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

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        await self.set_power(head, percent)

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        assert self._client
        for axis_name, delta in deltas.items():
            axis_lower = axis_name.lower()
            delta_um = int(delta * 1000)
            if axis_lower == "x":
                await self._client.rapid_move_axis(0x00, delta_um)
            elif axis_lower == "y":
                await self._client.rapid_move_axis(0x01, delta_um)

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        """
        Set a reference point offset on the controller.

        Args:
            wcs_slot: "REF0" or "REF1"
            x, y, z: Offset in mm (z ignored, Ruida is 2D)
        """
        if wcs_slot == "MACHINE":
            return

        if not self._client:
            return

        if wcs_slot not in self._client.ref_points:
            logger.warning(f"Unknown WCS slot: {wcs_slot}")
            return

        x_um = int(x * 1000)
        y_um = int(y * 1000)

        await self._client.set_ref_point_offset(wcs_slot, x_um, y_um)
        self.wcs_updated.send(self, offsets={wcs_slot: (x, y, z)})

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        """
        Read reference point offsets from the controller.

        Returns offsets for REF0 and REF1 in mm. MACHINE is always zero.
        """
        offsets: Dict[str, Pos] = {"MACHINE": (0.0, 0.0, 0.0)}

        if not self._client or not self._is_connected:
            self.wcs_updated.send(self, offsets=offsets)
            return offsets

        for ref_point in self._client.ref_points:
            if ref_point == "MACHINE":
                continue
            result = await self._client.get_ref_point_offset(ref_point)
            if result is not None:
                x_um, y_um = result
                offsets[ref_point] = (x_um / 1000.0, y_um / 1000.0, 0.0)

        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> Optional[str]:
        if not self._client or not self._is_connected:
            return None
        return await self._client.get_ref_point_mode()

    async def select_wcs(self, wcs: str) -> None:
        """
        Select a reference point mode on the controller.

        Args:
            wcs: "REF0", "REF1", or "MACHINE"
        """
        if not self._client:
            return
        await self._client.select_ref_point(wcs)

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        self.probe_status_changed.send(self, message="Probe not supported")
        return None

    async def _fetch_card_info(self) -> None:
        if not self._client:
            return
        try:
            card_info = await self._client.get_card_info()
            if card_info:
                card_id, model_name = card_info
                device = (
                    f"{model_name or 'Ruida controller'} "
                    f"(Card ID: 0x{card_id:08X})"
                )
            else:
                device = "Ruida controller"
            logger.info(
                f"Identified: {device}",
                extra=self._log_extra("MACHINE_EVENT"),
            )
        except Exception as e:
            logger.debug(f"Could not fetch card info: {e}")

    def _on_state_changed(self, sender) -> None:
        self._response_received.set()

    def _on_position_updated(self, sender, axis: str, value_um: int) -> None:
        """Handle position update from client."""
        pos_mm = value_um / 1000.0
        current_pos = self.state.machine_pos

        if axis == "x":
            new_pos = (pos_mm, current_pos[1], current_pos[2])
        elif axis == "y":
            new_pos = (current_pos[0], pos_mm, current_pos[2])
        elif axis == "z":
            new_pos = (current_pos[0], current_pos[1], pos_mm)
        else:
            return

        if new_pos != current_pos:
            self.state = replace(self.state, machine_pos=new_pos)
            logger.debug(
                f"Position update: {axis}={pos_mm:.3f}mm, "
                f"machine_pos={self.state.machine_pos}"
            )
            self.state_changed.send(self, state=self.state)

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
        return self._is_connected
