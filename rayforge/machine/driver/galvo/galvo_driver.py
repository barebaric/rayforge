"""
Galvo Driver - High-level driver for EzCad2/LMC galvo laser controllers.

Implements the Driver interface with coordinate conversion
(mm -> galvo native) and uses GalvoController for low-level
communication.
"""

import asyncio
import inspect
import logging
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Union,
)

from raygeo.ops.axis import Axis

from ....context import RayforgeContext
from ....core.capability import PWMCapability
from ....core.varset import ChoiceVar, VarSet, BoolVar
from ....pipeline.encoder.base import EncodedOutput, OpsEncoder
from ...models.laser import LaserType
from ...transport import TransportStatus
from ..driver import (
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverSetupError,
    Pos,
)
from .galvo_controller import GalvoController
from .galvo_encoder import GalvoEncoder
from .galvo_mock_connection import MockGalvoConnection

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


class GalvoDriver(Driver):
    """
    Driver for EzCad2/LMC galvo laser controllers (USB).

    Supports galvo-based laser systems like AtomStack M4.
    Coordinates are converted from mm to native galvo values
    (0-65535, center at 0x8000).
    """

    label = _("Galvo (EzCad2)")
    subtitle = _("Connect to a Galvo/EzCad2 laser controller over USB")
    supports_settings = False
    reports_granular_progress = False
    uses_gcode = False
    maturity = DriverMaturity.EXPERIMENTAL

    CONNECTION_TIMEOUT = 5.0
    RECONNECT_INTERVAL = 3.0
    POSITION_POLL_INTERVAL = 1.0

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._controller: Optional[GalvoController] = None
        self._connection_task: Optional[asyncio.Task] = None
        self._keep_running = False
        self._is_connected = False
        self._source: str = "fiber"
        self._mock: bool = False

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @property
    def supported_wcs(self) -> List[str]:
        return [self.machine_space_wcs]

    @property
    def resource_uri(self) -> Optional[str]:
        return "usb://galvo/ezcad2"

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        pass

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                ChoiceVar(
                    key="source",
                    label=_("Laser Source"),
                    description=_("Type of laser source"),
                    choices=["fiber", "co2", "uv"],
                    default="fiber",
                ),
                BoolVar(
                    key="mock",
                    label=_("Mock Connection"),
                    description=_(
                        "Use mock connection for "
                        "testing (no hardware required)"
                    ),
                    default=False,
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        return GalvoEncoder()

    def get_laser_capabilities(self, laser: "Laser"):
        if laser.laser_type != LaserType.DIODE:
            return (
                PWMCapability(
                    frequency=laser.pwm_frequency,
                    max_frequency=laser.max_pwm_frequency,
                    pulse_width=laser.pulse_width,
                    min_pulse_width=laser.min_pulse_width,
                    max_pulse_width=laser.max_pulse_width,
                ),
            )
        return ()

    def _setup_implementation(self, **kwargs: Any) -> None:
        self._source = kwargs.get("source", "fiber")
        self._mock = kwargs.get("mock", False)

        if self._mock:
            connection = MockGalvoConnection()
        else:
            from .galvo_usb_connection import GalvoUSBConnection

            connection = GalvoUSBConnection()

        self._controller = GalvoController(
            connection=connection,
            source=self._source,
        )
        self._init_coordinate_systems()

    def _init_coordinate_systems(self) -> None:
        m = self._machine
        existing = m.coordinate_systems
        supported = self.supported_wcs
        new_systems = {}
        for name in supported:
            if name in existing:
                new_systems[name] = existing[name]
            else:
                from ...models.coordinate_system import CoordinateSystem

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

        if self._controller:
            await self._controller.disconnect()
            self._controller = None

        self._update_connection_status(TransportStatus.DISCONNECTED, "")
        await super().cleanup()

    async def _connect_implementation(self) -> None:
        if self._connection_task and not self._connection_task.done():
            logger.warning("Connect called with active connection task")
            return

        self._keep_running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        logger.debug("Entering Galvo connection loop")
        while self._keep_running:
            self._update_connection_status(TransportStatus.CONNECTING)
            self._is_connected = False

            try:
                if not self._controller:
                    raise DriverSetupError("Controller not initialized")

                await self._controller.connect()
                self._is_connected = True
                self._update_connection_status(TransportStatus.CONNECTED, "")
                self.state.status = DeviceStatus.IDLE
                self.state_changed.send(self, state=self.state)

                logger.info(
                    "Connected to Galvo controller",
                    extra=self._log_extra("MACHINE_EVENT"),
                )

                while self._keep_running and self._is_connected:
                    await asyncio.sleep(self.POSITION_POLL_INTERVAL)

            except asyncio.CancelledError:
                logger.debug("Connection loop cancelled")
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self._update_connection_status(TransportStatus.ERROR, str(e))
                if self._controller:
                    await self._controller.disconnect()

            if self._keep_running:
                self._update_connection_status(TransportStatus.SLEEPING)
                await asyncio.sleep(self.RECONNECT_INTERVAL)

        logger.debug("Exiting Galvo connection loop")

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        ops: "Ops",
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

        if binary_data and self._controller:
            self._controller._reset_list()
            self._controller._list_write(0x8051)
            chunk_size = 12
            for i in range(0, len(binary_data), chunk_size):
                chunk = binary_data[i : i + chunk_size]
                cmd = int.from_bytes(chunk[0:2], "little")
                v1 = int.from_bytes(chunk[2:4], "little")
                v2 = int.from_bytes(chunk[4:6], "little")
                v3 = int.from_bytes(chunk[6:8], "little")
                v4 = int.from_bytes(chunk[8:10], "little")
                v5 = int.from_bytes(chunk[10:12], "little")
                self._controller._list_write(cmd, v1, v2, v3, v4, v5)

        self.job_finished.send(self)

    async def run_raw(self, machine_code: str) -> None:
        if machine_code and machine_code.strip():
            logger.warning(
                "Galvo controllers do not support "
                "text-based machine code. "
                "Use run() with EncodedOutput instead."
            )
        self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        assert self._controller
        if hold:
            self._controller._send_single(0x0020)
        else:
            self._controller._send_single(0x0013)

    async def cancel(self) -> None:
        assert self._controller
        self._controller._send_single(0x001F)

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        return False

    async def home(self, axes: Optional[Axis] = None) -> None:
        logger.info("Home not supported on galvo controllers")

    async def move_to(self, pos_x: float, pos_y: float) -> None:
        assert self._controller
        logger.info(
            f"move_to x={pos_x:.2f} y={pos_y:.2f}",
            extra=self._log_extra("MACHINE_EVENT"),
        )
        x = int((pos_x / 200.0) * 32768.0 + 32768.0)
        y = int((pos_y / 200.0) * 32768.0 + 32768.0)
        x = max(0, min(0xFFFF, x))
        y = max(0, min(0xFFFF, y))
        self._controller.goto_xy(x, y)

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
        assert self._controller
        self._controller._send_single(0x001F)

    async def set_power(self, head: "Laser", percent: float) -> None:
        assert self._controller
        power_pct = percent * 100.0
        self._controller.set_power(power_pct)
        self._controller.set_frequency(30.0)

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        await self.set_power(head, percent)

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        assert self._controller
        for axis_name, delta in deltas.items():
            axis_lower = axis_name.lower()
            delta_galvo = int((delta / 200.0) * 32768.0)
            if axis_lower == "x":
                new_x = self._controller._last_x + delta_galvo
                new_x = max(0, min(0xFFFF, new_x))
                self._controller.goto_xy(new_x, self._controller._last_y)
            elif axis_lower == "y":
                new_y = self._controller._last_y + delta_galvo
                new_y = max(0, min(0xFFFF, new_y))
                self._controller.goto_xy(self._controller._last_x, new_y)

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        pass

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        offsets: Dict[str, Pos] = {"MACHINE": (0.0, 0.0, 0.0)}
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> Optional[str]:
        return None

    async def select_wcs(self, wcs: str) -> None:
        pass

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        self.probe_status_changed.send(
            self, message="Probe not supported on galvo"
        )
        return None

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    def _update_connection_status(
        self, status: TransportStatus, message: str = ""
    ) -> None:
        self.connection_status_changed.send(
            self, status=status, message=message
        )
