import math
from typing import TYPE_CHECKING

from ..core.geo import Point3D
from .driver.driver import Axis

if TYPE_CHECKING:
    from ..simulator.machine_state import MachineState


class Kinematics:
    """Base class. Translates MachineState into spatial data."""

    def head_position(self, state: "MachineState") -> Point3D:
        raise NotImplementedError

    def cylinder_angle(self, state: "MachineState") -> float:
        return 0.0


class CartesianKinematics(Kinematics):
    def head_position(self, state: "MachineState") -> Point3D:
        return (
            state.axes[Axis.X],
            state.axes[Axis.Y],
            state.axes[Axis.Z],
        )


class RotaryKinematics(CartesianKinematics):
    def __init__(self, rotary_diameter: float):
        if rotary_diameter <= 0:
            raise ValueError(f"Invalid rotary diameter: {rotary_diameter}")
        self.rotary_diameter = rotary_diameter

    def cylinder_angle(self, state: "MachineState") -> float:
        circumference = self.rotary_diameter * math.pi
        return (state.axes[Axis.Y] / circumference) * 2 * math.pi
