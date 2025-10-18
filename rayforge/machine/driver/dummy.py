from .driver import Driver, Axis
from ...core.ops import Ops
from ...shared.varset import VarSet
from typing import Any, TYPE_CHECKING, List, Optional, Callable
import asyncio

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ..models.machine import Machine


class NoDeviceDriver(Driver):
    """
    A dummy driver that is used if the user has no machine.
    """

    label = _("No driver")
    subtitle = _("No connection")
    supports_settings = False
    reports_granular_progress = True

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        pass

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(title=_("No settings"))

    def get_setting_vars(self) -> List["VarSet"]:
        return [VarSet(title=_("No settings"))]

    async def connect(self) -> None:
        pass

    async def run(
        self,
        ops: Ops,
        machine: "Machine",
        doc: "Doc",
        on_command_done: Optional[Callable[[int], None]] = None,
    ) -> None:
        """
        Dummy implementation that simulates command execution.

        This implementation creates a GcodeOpMap to track which commands
        correspond to which Ops, then simulates execution by calling the
        on_command_done callback for each command with a small delay.
        """
        # Get the operation map for tracking
        _ = self._track_command_execution(ops, machine, doc, on_command_done)

        # Simulate command execution with delays
        for op_index in range(len(ops)):
            # Small delay to simulate execution time
            await asyncio.sleep(0.01)

            # Call the callback if provided
            if on_command_done is not None:
                try:
                    on_command_done(op_index)
                except Exception:
                    # Don't let callback exceptions stop execution
                    pass

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

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        """Dummy driver supports jogging for all axes."""
        return True

    async def jog(self, axis: Axis, distance: float, speed: int) -> None:
        pass

    def can_g0_with_speed(self) -> bool:
        """Dummy driver supports G0 with speed."""
        return True
