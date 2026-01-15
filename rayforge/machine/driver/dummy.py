import asyncio
import inspect
import logging
from typing import (
    Any,
    TYPE_CHECKING,
    List,
    Optional,
    Callable,
    Union,
    Awaitable,
    Dict,
    cast,
)
from ...context import RayforgeContext
from ...core.varset import VarSet
from ...pipeline.encoder.base import OpsEncoder, MachineCodeOpMap
from ...pipeline.encoder.gcode import GcodeEncoder
from ..transport import TransportStatus
from .driver import Driver, Axis, Pos, DeviceStatus

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ..models.machine import Machine
    from ..models.laser import Laser


logger = logging.getLogger(__name__)


class NoDeviceDriver(Driver):
    """
    A dummy driver that is used if the user has no machine.
    """

    label = _("No driver")
    subtitle = _("No connection")
    supports_settings = False
    reports_granular_progress = True

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        # Internal state for WCS offsets to behave like a stateful machine
        # Initialize from machine's persisted state to prevent overwriting
        # loaded configuration with defaults upon connection.
        self._offsets: Dict[str, Pos] = cast(
            Dict[str, Pos], machine.wcs_offsets.copy()
        )

        # Ensure standard keys exist
        defaults = ["G54", "G55", "G56", "G57", "G58", "G59"]
        for key in defaults:
            if key not in self._offsets:
                self._offsets[key] = (0.0, 0.0, 0.0)

    @property
    def machine_space_wcs(self) -> str:
        """
        Returns the machine space coordinate system identifier.
        This is an immutable coordinate system with zero offset.
        """
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        """
        Returns a human-readable display name for the machine space
        coordinate system.
        """
        return _("Machine Coordinates")

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        pass

    def _setup_implementation(self, **kwargs: Any) -> None:
        pass

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(title=_("No settings"))

    def get_encoder(self) -> "OpsEncoder":
        """Returns a GcodeEncoder configured for the machine's dialect."""
        return GcodeEncoder(self._machine.dialect)

    def get_setting_vars(self) -> List["VarSet"]:
        return [VarSet(title=_("No settings"))]

    async def _connect_implementation(self) -> None:
        # Simulate connection sequence
        self.connection_status_changed.send(
            self, status=TransportStatus.CONNECTING
        )
        await asyncio.sleep(0.1)

        # Set IDLE state so the UI knows we are ready and not "busy"
        self.state.status = DeviceStatus.IDLE
        self.state_changed.send(self, state=self.state)

        self.connection_status_changed.send(
            self, status=TransportStatus.CONNECTED
        )

        # Upon connect, broadcast WCS state (matches loaded machine state)
        self.wcs_updated.send(self, offsets=self._offsets)

    async def run(
        self,
        machine_code: Any,
        op_map: "MachineCodeOpMap",
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        """
        Dummy implementation that simulates command execution.

        This implementation iterates through the ops defined in op_map and
        simulates execution by calling the on_command_done callback for each
        command with a small delay.
        """
        # We assume ops are indexed 0..N-1.
        num_ops = 0
        if op_map and op_map.op_to_machine_code:
            num_ops = max(op_map.op_to_machine_code.keys()) + 1

        # Simulate command execution with delays
        for op_index in range(num_ops):
            # Small delay to simulate execution time
            await asyncio.sleep(0.01)

            # Call the callback if provided, awaiting it if it's a coroutine
            if on_command_done is not None:
                try:
                    result = on_command_done(op_index)
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    # Don't let callback exceptions stop execution
                    pass
        self.job_finished.send(self)

    async def run_raw(self, gcode: str) -> None:
        """
        Dummy implementation that simulates raw G-code execution.
        """
        gcode_lines = gcode.splitlines()
        for _ in gcode_lines:
            # Small delay to simulate execution time
            await asyncio.sleep(0.01)
        self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        pass

    async def cancel(self) -> None:
        pass

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        """Dummy driver supports homing for all axes."""
        return True

    async def home(self, axes: Optional[Axis] = None) -> None:
        pass

    async def move_to(self, pos_x, pos_y) -> None:
        pass

    async def select_tool(self, tool_number: int) -> None:
        pass

    async def read_settings(self) -> None:
        pass

    async def write_setting(self, key: str, value: Any) -> None:
        pass

    async def clear_alarm(self) -> None:
        pass

    async def set_power(self, head: "Laser", percent: float) -> None:
        """
        Sets the laser power to the specified percentage of max power.

        Args:
            head: The laser head to control.
            percent: Power percentage (0.0-1.0). 0 disables power.
        """
        # Dummy driver doesn't control any hardware, so just log the call
        logger.info(
            f"set_power called with head {head.uid} at {percent * 100:.1f}%",
            extra={"log_category": "DRIVER_CMD"},
        )

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """Dummy driver supports jogging for all axes."""
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        pass

    def can_g0_with_speed(self) -> bool:
        """Dummy driver supports G0 with speed."""
        return True

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        """Dummy implementation, updates internal state."""
        self._offsets[wcs_slot] = (x, y, z)
        # Notify machine that the driver updated offsets
        self.wcs_updated.send(self, offsets=self._offsets)

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        """Dummy implementation, returns internal state."""
        self.wcs_updated.send(self, offsets=self._offsets)
        return self._offsets

    async def read_parser_state(self) -> Optional[str]:
        """
        Simulate reading the active WCS state.
        Returns the machine's active WCS to treat the client's selection
        as the source of truth for the dummy driver.
        """
        return self._machine.active_wcs

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        """
        Dummy implementation, simulates a successful probe after a short delay.
        """
        self.probe_status_changed.send(
            self, message=f"Simulating probe cycle for axis {axis.name}..."
        )
        await asyncio.sleep(0.5)
        # Simulate a successful probe at a fixed position
        simulated_pos = (10.0, 15.0, -1.0)
        self.probe_status_changed.send(
            self, message=f"Probe triggered at {simulated_pos}"
        )
        return simulated_pos
