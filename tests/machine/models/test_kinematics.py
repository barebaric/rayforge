import math

from rayforge.core.ops.axis import Axis
from rayforge.machine.kinematics import (
    Kinematics,
    create_kinematics,
)
from rayforge.machine.models.axis import (
    AxisConfig,
    AxisSet,
    AxisType,
)
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.rotary_module import RotaryModule
from rayforge.simulator.machine_state import MachineState


def _make_3axis_set():
    return AxisSet(
        [
            AxisConfig(
                letter=Axis.X,
                axis_type=AxisType.LINEAR,
                extents=(0, 200),
            ),
            AxisConfig(
                letter=Axis.Y,
                axis_type=AxisType.LINEAR,
                extents=(0, 200),
            ),
            AxisConfig(
                letter=Axis.Z,
                axis_type=AxisType.LINEAR,
                extents=(-50, 50),
            ),
        ]
    )


def _make_4axis_set(diameter=50.0):
    return AxisSet(
        [
            AxisConfig(
                letter=Axis.X,
                axis_type=AxisType.LINEAR,
                extents=(0, 400),
            ),
            AxisConfig(
                letter=Axis.Y,
                axis_type=AxisType.LINEAR,
                extents=(0, 200),
            ),
            AxisConfig(
                letter=Axis.Z,
                axis_type=AxisType.LINEAR,
                extents=(-50, 50),
            ),
            AxisConfig(
                letter=Axis.A,
                axis_type=AxisType.ROTARY,
                extents=(0, 360),
                rotary_diameter=diameter,
            ),
        ]
    )


def _make_state(x=0.0, y=0.0, z=0.0, a=0.0):
    state = MachineState()
    state.axes[Axis.X] = x
    state.axes[Axis.Y] = y
    state.axes[Axis.Z] = z
    state.axes[Axis.A] = a
    return state


def test_cartesian_head_positions():
    kinematics = create_kinematics(axis_set=_make_3axis_set())
    state = _make_state(x=10.0, y=20.0, z=0.0)
    heads = kinematics.head_positions(state)
    assert heads == {"head_0": (10.0, 20.0, 0.0)}


def test_cartesian_chuck_angles_empty():
    kinematics = create_kinematics(axis_set=_make_3axis_set())
    angles = kinematics.chuck_angles(_make_state(a=100.0))
    assert angles == {}


def test_cartesian_no_rotary():
    kinematics = create_kinematics(axis_set=_make_3axis_set())
    assert not kinematics.has_rotary


def test_rotary_chuck_angles_quarter_turn():
    diameter = 50.0
    kinematics = create_kinematics(
        axis_set=_make_4axis_set(diameter),
        rotary_modules={"a": _make_module_a(diameter)},
    )
    state = _make_state(a=90.0)
    angles = kinematics.chuck_angles(state)
    assert abs(angles["rotary_chuck_0"] - math.pi / 2) < 1e-9


def test_rotary_head_positions():
    kinematics = create_kinematics(
        axis_set=_make_4axis_set(40.0),
        rotary_modules={"a": _make_module_a(40.0)},
    )
    state = _make_state(x=5.0, y=10.0, z=3.0)
    heads = kinematics.head_positions(state)
    assert heads == {"head_0": (5.0, 10.0, 3.0)}


def test_rotary_has_rotary():
    kinematics = create_kinematics(
        axis_set=_make_4axis_set(40.0),
        rotary_modules={"a": _make_module_a(40.0)},
    )
    assert kinematics.has_rotary


def test_4axis_set_no_rotary_modules_is_cartesian():
    kinematics = create_kinematics(axis_set=_make_4axis_set(40.0))
    assert not kinematics.has_rotary


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


def _make_module_a(diameter=25.0):
    rm = RotaryModule()
    rm.axis = Axis.A
    rm.default_diameter = diameter
    return rm
