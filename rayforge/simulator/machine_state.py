from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional

from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandCategory, CommandType

if TYPE_CHECKING:
    from ..machine.models.axis import AxisSet


class MachineState:
    def __init__(
        self,
        axis_letters: Optional[Iterable[Axis]] = None,
    ):
        self.power: float = 0.0
        self.air_assist: bool = False
        self.cut_speed: Optional[int] = None
        self.travel_speed: Optional[int] = None
        self.active_laser_uid: Optional[str] = None
        self.frequency: Optional[int] = None
        self.pulse_width: Optional[float] = None
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

    def apply_command(self, ops: Ops, idx: int):
        ct = ops.command_type(idx)
        cat = ops.category(idx)

        if cat == CommandCategory.STATE:
            if ct == CommandType.SET_POWER:
                self.power = ops.power(idx)
            elif ct == CommandType.SET_FEED_RATE:
                self.cut_speed = int(ops.rate(idx))
            elif ct == CommandType.SET_RAPID_RATE:
                self.travel_speed = int(ops.rate(idx))
            elif ct == CommandType.SET_AIR_ASSIST:
                self.air_assist = ops.air_assist(idx) == "On"
            elif ct == CommandType.SET_HEAD:
                self.active_laser_uid = ops.head_uid(idx)
            elif ct == CommandType.SET_FREQUENCY:
                self.frequency = ops.frequency(idx)
            elif ct == CommandType.SET_PULSE_WIDTH:
                self.pulse_width = ops.pulse_width(idx)
        elif cat == CommandCategory.MOVING:
            end = ops.endpoint(idx)
            self.axes[Axis.X] = end[0]
            self.axes[Axis.Y] = end[1]
            self.axes[Axis.Z] = end[2]
            ea = ops.extra_axes(idx)
            if ea:
                for axis, value in ea.items():
                    self.axes[axis] = value
            if ct == CommandType.SCAN_LINE:
                self.reached_textures.add(idx)
            self.laser_on = ct != CommandType.MOVE_TO
        elif ct == CommandType.LAYER_START:
            self.current_layer_uid = ops.layer_uid(idx)

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
