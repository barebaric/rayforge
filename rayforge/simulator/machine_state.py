from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional

from ..core.ops import Axis, State
from ..core.ops.commands import (
    Command,
    LayerStartCommand,
    MovingCommand,
    ScanLinePowerCommand,
)

if TYPE_CHECKING:
    from ..machine.models.axis import AxisSet


class MachineState(State):
    def __init__(
        self,
        axis_letters: Optional[Iterable[Axis]] = None,
    ):
        super().__init__()
        if axis_letters is not None:
            self.axes: Dict[Axis, float] = {a: 0.0 for a in axis_letters}
        else:
            self.axes = {
                Axis.X: 0.0,
                Axis.Y: 0.0,
                Axis.Z: 0.0,
            }
        self.laser_on = False
        self.reached_textures: set = set()
        self.current_layer_uid: Optional[str] = None

    @classmethod
    def from_axis_set(cls, axis_set: AxisSet) -> MachineState:
        axes = (cfg.letter for cfg in axis_set.configs)
        return cls(axis_letters=axes)

    def apply_command(self, cmd: Command, index: int):
        if cmd.is_state_command():
            cmd.apply_to_state(self)
        elif isinstance(cmd, MovingCommand):
            if cmd.end is not None:
                self.axes[Axis.X] = cmd.end[0]
                self.axes[Axis.Y] = cmd.end[1]
                self.axes[Axis.Z] = cmd.end[2]
            for axis, value in cmd.extra_axes.items():
                self.axes[axis] = value
            if isinstance(cmd, ScanLinePowerCommand):
                self.reached_textures.add(index)
            self.laser_on = cmd.is_cutting_command()
        elif isinstance(cmd, LayerStartCommand):
            self.current_layer_uid = cmd.layer_uid
        elif cmd.is_marker():
            return
        else:
            raise TypeError(f"Unknown command type: {type(cmd).__name__}")

    def copy(self) -> MachineState:
        new = MachineState.__new__(MachineState)
        new.power = self.power
        new.air_assist = self.air_assist
        new.cut_speed = self.cut_speed
        new.travel_speed = self.travel_speed
        new.active_laser_uid = self.active_laser_uid
        new.axes = dict(self.axes)
        new.laser_on = self.laser_on
        new.reached_textures = set(self.reached_textures)
        new.current_layer_uid = self.current_layer_uid
        return new
