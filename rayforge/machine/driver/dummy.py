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
)
from ...context import RayforgeContext
from ...core.ops import Ops
from ...core.varset import VarSet
from ...pipeline.encoder.base import OpsEncoder
from ...pipeline.encoder.gcode import GcodeEncoder
from .driver import Driver, Axis

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

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
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
        pass

    async def run(
        self,
        ops: Ops,
        doc: "Doc",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        """
        Dummy implementation that simulates command execution.

        This implementation creates a MachineCodeOpMap to track which commands
        correspond to which Ops, then simulates execution by calling the
        on_command_done callback for each command with a small delay.
        """
        # Get the operation map for tracking
        _ = self._track_command_execution(ops, doc, on_command_done)

        # Simulate command execution with delays
        for op_index in range(len(ops)):
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

    async def jog(self, axis: Axis, distance: float, speed: int) -> None:
        pass

    def can_g0_with_speed(self) -> bool:
        """Dummy driver supports G0 with speed."""
        return True
