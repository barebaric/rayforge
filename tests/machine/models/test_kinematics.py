import math

import pytest

from rayforge.machine.driver.driver import Axis
from rayforge.machine.kinematics import (
    CartesianKinematics,
    Kinematics,
    RotaryKinematics,
)
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryModule
from rayforge.simulator.machine_state import MachineState


def _make_state(x=0.0, y=0.0, z=0.0):
    state = MachineState()
    state.axes[Axis.X] = x
    state.axes[Axis.Y] = y
    state.axes[Axis.Z] = z
    return state


def test_base_head_position_raises():
    kinematics = Kinematics()
    with pytest.raises(NotImplementedError):
        kinematics.head_position(MachineState())


def test_base_cylinder_angle_returns_zero():
    kinematics = Kinematics()
    assert kinematics.cylinder_angle(MachineState()) == 0.0


def test_cartesian_head_position():
    kinematics = CartesianKinematics()
    state = _make_state(x=10.0, y=20.0, z=0.0)
    assert kinematics.head_position(state) == (10.0, 20.0, 0.0)


def test_cartesian_cylinder_angle_returns_zero():
    kinematics = CartesianKinematics()
    assert kinematics.cylinder_angle(_make_state(y=100.0)) == 0.0


def test_rotary_cylinder_angle_quarter_turn():
    diameter = 50.0
    circumference = diameter * math.pi
    kinematics = RotaryKinematics(rotary_diameter=diameter)
    state = _make_state(y=circumference / 4)
    angle = kinematics.cylinder_angle(state)
    assert abs(angle - math.pi / 2) < 1e-9


def test_rotary_head_position_inherited():
    kinematics = RotaryKinematics(rotary_diameter=40.0)
    state = _make_state(x=5.0, y=10.0, z=3.0)
    assert kinematics.head_position(state) == (5.0, 10.0, 3.0)


def test_rotary_zero_diameter_raises():
    with pytest.raises(ValueError, match="Invalid rotary diameter"):
        RotaryKinematics(rotary_diameter=0)


def test_rotary_negative_diameter_raises():
    with pytest.raises(ValueError, match="Invalid rotary diameter"):
        RotaryKinematics(rotary_diameter=-10)


def test_machine_kinematics_no_rotary():
    from rayforge.context import RayforgeContext

    ctx = RayforgeContext()
    machine = Machine(ctx)
    kin = machine.kinematics
    assert isinstance(kin, CartesianKinematics)


def test_machine_kinematics_with_rotary():
    from rayforge.context import RayforgeContext

    ctx = RayforgeContext()
    machine = Machine(ctx)
    module = RotaryModule()
    module.default_diameter = 40.0
    machine.rotary_modules["test"] = module
    kin = machine.kinematics
    assert isinstance(kin, RotaryKinematics)
    assert kin.rotary_diameter == 40.0
