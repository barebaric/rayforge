import math

from rayforge.core.ops.axis import Axis
from rayforge.machine.kinematics import (
    Kinematics,
    create_kinematics,
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


def test_cartesian_head_positions():
    kinematics = create_kinematics()
    state = _make_state(x=10.0, y=20.0, z=0.0)
    heads = kinematics.head_positions(state)
    assert heads == {"head_0": (10.0, 20.0, 0.0)}


def test_cartesian_chuck_angles_empty():
    kinematics = create_kinematics()
    angles = kinematics.chuck_angles(_make_state(y=100.0))
    assert angles == {}


def test_cartesian_no_rotary():
    kinematics = create_kinematics()
    assert not kinematics.has_rotary


def test_rotary_chuck_angles_quarter_turn():
    diameter = 50.0
    circumference = diameter * math.pi
    kinematics = create_kinematics(rotary_diameter=diameter)
    state = _make_state(y=circumference / 4)
    angles = kinematics.chuck_angles(state)
    assert abs(angles["rotary_chuck_0"] - math.pi / 2) < 1e-9


def test_rotary_head_positions():
    kinematics = create_kinematics(rotary_diameter=40.0)
    state = _make_state(x=5.0, y=10.0, z=3.0)
    heads = kinematics.head_positions(state)
    assert heads == {"head_0": (5.0, 10.0, 3.0)}


def test_rotary_has_rotary():
    kinematics = create_kinematics(rotary_diameter=40.0)
    assert kinematics.has_rotary
    assert kinematics.rotary_diameter == 40.0


def test_machine_kinematics_no_rotary():
    from rayforge.context import RayforgeContext

    ctx = RayforgeContext()
    machine = Machine(ctx)
    kin = machine.kinematics
    assert isinstance(kin, Kinematics)
    assert not kin.has_rotary


def test_machine_kinematics_with_rotary():
    from rayforge.context import RayforgeContext

    ctx = RayforgeContext()
    machine = Machine(ctx)
    module = RotaryModule()
    module.default_diameter = 40.0
    machine.add_rotary_module(module)
    kin = machine.kinematics
    assert isinstance(kin, Kinematics)
    assert kin.has_rotary
    assert kin.rotary_diameter == 40.0
