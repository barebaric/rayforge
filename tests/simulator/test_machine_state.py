from rayforge.simulator.machine_state import MachineState
from rayforge.core.ops import Ops, Axis
from rayforge.core.ops.commands import (
    MoveToCommand,
    ScanLinePowerCommand,
)
from rayforge.machine.models.axis import (
    AxisConfig,
    AxisSet,
    AxisType,
    AxisDirection,
)


def test_walk_movement_updates_axes():
    state = MachineState()
    ops = Ops()
    ops.move_to(10.0, 20.0, 0.0)
    ops.line_to(30.0, 40.0, 5.0)

    for i, cmd in enumerate(ops):
        state.apply_command(cmd, i)

    assert state.axes[Axis.X] == 30.0
    assert state.axes[Axis.Y] == 40.0
    assert state.axes[Axis.Z] == 5.0


def test_state_commands_no_axis_change():
    state = MachineState()
    ops = Ops()
    ops.set_power(0.8)
    ops.set_cut_speed(500)
    ops.set_travel_speed(3000)
    ops.enable_air_assist()

    for i, cmd in enumerate(ops):
        state.apply_command(cmd, i)

    assert state.axes[Axis.X] == 0.0
    assert state.axes[Axis.Y] == 0.0
    assert state.axes[Axis.Z] == 0.0
    assert state.power == 0.8
    assert state.cut_speed == 500
    assert state.travel_speed == 3000
    assert state.air_assist is True


def test_scanline_tracked_in_reached_textures():
    state = MachineState()
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([100, 200])))

    for i, cmd in enumerate(ops):
        state.apply_command(cmd, i)

    assert 1 in state.reached_textures
    assert 0 not in state.reached_textures


def test_laser_on_during_cutting():
    state = MachineState()
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)

    for i, cmd in enumerate(ops):
        state.apply_command(cmd, i)

    assert state.laser_on is True


def test_laser_off_during_travel():
    state = MachineState()
    ops = Ops()
    ops.move_to(10.0, 10.0, 0.0)

    for i, cmd in enumerate(ops):
        state.apply_command(cmd, i)

    assert state.laser_on is False


def test_markers_ignored():
    state = MachineState()
    ops = Ops()
    ops.set_power(0.5)
    ops.line_to(5.0, 5.0, 0.0)

    for i, cmd in enumerate(ops):
        state.apply_command(cmd, i)

    assert state.power == 0.5
    assert state.axes[Axis.X] == 5.0


def test_copy_is_independent():
    state = MachineState()
    state.power = 0.7
    state.axes[Axis.X] = 42.0
    state.reached_textures.add(3)

    snapshot = state.copy()

    state.power = 0.0
    state.axes[Axis.X] = 0.0
    state.reached_textures.clear()

    assert snapshot.power == 0.7
    assert snapshot.axes[Axis.X] == 42.0
    assert 3 in snapshot.reached_textures


def test_copy_preserves_state_fields():
    state = MachineState()
    state.power = 0.9
    state.air_assist = True
    state.cut_speed = 1200
    state.travel_speed = 4000
    state.active_laser_uid = "laser-1"

    snapshot = state.copy()

    assert snapshot.power == 0.9
    assert snapshot.air_assist is True
    assert snapshot.cut_speed == 1200
    assert snapshot.travel_speed == 4000
    assert snapshot.active_laser_uid == "laser-1"


def test_unknown_command_raises():
    state = MachineState()

    class WeirdCommand:
        def is_state_command(self):
            return False

        def is_marker(self):
            return False

    try:
        state.apply_command(WeirdCommand(), 0)  # type: ignore
        assert False, "Should have raised TypeError"
    except TypeError:
        pass


def test_mixed_ops_walk():
    ops = Ops()
    ops.set_power(0.5)
    ops.set_cut_speed(800)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.set_power(1.0)
    ops.line_to(10.0, 10.0, 0.0)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(5.0, 5.0, 2.0)

    state = MachineState()
    for i, cmd in enumerate(ops):
        state.apply_command(cmd, i)

    assert state.axes[Axis.X] == 5.0
    assert state.axes[Axis.Y] == 5.0
    assert state.axes[Axis.Z] == 2.0
    assert state.power == 1.0
    assert state.cut_speed == 800
    assert state.laser_on is True


def test_default_axes_xyz():
    state = MachineState()
    assert set(state.axes.keys()) == {Axis.X, Axis.Y, Axis.Z}


def test_axis_letters_initializes_all_axes():
    state = MachineState(axis_letters=[Axis.X, Axis.Y, Axis.Z, Axis.A])
    assert set(state.axes.keys()) == {Axis.X, Axis.Y, Axis.Z, Axis.A}
    assert state.axes[Axis.A] == 0.0


def test_from_axis_set():
    axis_set = AxisSet(
        [
            AxisConfig(
                Axis.X, AxisType.LINEAR, (0, 400), AxisDirection.NORMAL
            ),
            AxisConfig(
                Axis.Y, AxisType.LINEAR, (0, 200), AxisDirection.NORMAL
            ),
            AxisConfig(
                Axis.Z, AxisType.LINEAR, (-50, 50), AxisDirection.NORMAL
            ),
            AxisConfig(
                Axis.A, AxisType.ROTARY, (0, 3600), AxisDirection.NORMAL
            ),
        ]
    )
    state = MachineState.from_axis_set(axis_set)
    assert set(state.axes.keys()) == {Axis.X, Axis.Y, Axis.Z, Axis.A}
    assert state.axes[Axis.A] == 0.0


def test_extra_axes_applied():
    state = MachineState()
    cmd = MoveToCommand(
        (10.0, 20.0, 5.0),
        extra_axes={Axis.A: 45.0, Axis.B: 0.0},
    )
    state.apply_command(cmd, 0)

    assert state.axes[Axis.X] == 10.0
    assert state.axes[Axis.Y] == 20.0
    assert state.axes[Axis.Z] == 5.0
    assert state.axes[Axis.A] == 45.0
    assert state.axes[Axis.B] == 0.0


def test_empty_extra_axes_no_extra_keys():
    state = MachineState()
    cmd = MoveToCommand((10.0, 20.0, 5.0))
    state.apply_command(cmd, 0)

    assert set(state.axes.keys()) == {Axis.X, Axis.Y, Axis.Z}


def test_copy_preserves_extra_axes():
    state = MachineState(axis_letters=[Axis.X, Axis.Y, Axis.Z, Axis.A])
    cmd = MoveToCommand(
        (10.0, 20.0, 5.0),
        extra_axes={Axis.A: 90.0},
    )
    state.apply_command(cmd, 0)

    snapshot = state.copy()

    state.axes[Axis.A] = 0.0
    assert snapshot.axes[Axis.A] == 90.0
    assert set(snapshot.axes.keys()) == {Axis.X, Axis.Y, Axis.Z, Axis.A}
