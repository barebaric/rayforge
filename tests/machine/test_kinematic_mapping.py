import math

import pytest

from rayforge.core.ops.axis import Axis
from rayforge.core.ops import Ops
from rayforge.core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    MovingCommand,
    QuadraticBezierToCommand,
)
from rayforge.machine.kinematic_mapping import KinematicMapping
from rayforge.machine.models.rotary_module import (
    RotaryModule,
    RotaryMode,
    RotaryType,
)


class TestKinematicMappingApply:
    def test_y_to_a_degrees(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 0, 0)
        ops.line_to(30, 50, 0)

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[1] == pytest.approx(0.0)
        assert cmds[1].end[1] == pytest.approx(0.0)

        expected_deg_0 = (0.0 / (diameter * math.pi)) * 360.0
        expected_deg_50 = (50.0 / (diameter * math.pi)) * 360.0
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(expected_deg_0)
        assert cmds[1].extra_axes[Axis.A] == pytest.approx(expected_deg_50)

    def test_x_to_b_degrees(self):
        diameter = 40.0
        ops = Ops()
        ops.move_to(0, 10, 0)
        ops.line_to(100, 10, 0)

        mapping = KinematicMapping(
            source_axis=Axis.X,
            rotary_axis=Axis.B,
            diameter=diameter,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[0] == pytest.approx(0.0)
        assert cmds[1].end[0] == pytest.approx(0.0)

        circ = diameter * math.pi
        expected = (100.0 / circ) * 360.0
        assert cmds[1].extra_axes[Axis.B] == pytest.approx(expected)

    def test_z_affects_effective_diameter(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 0, 5)
        ops.line_to(10, 50, 5)

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        eff_d = diameter + 2 * 5
        expected = (50.0 / (eff_d * math.pi)) * 360.0
        assert cmds[1].extra_axes[Axis.A] == pytest.approx(expected)

    def test_gear_ratio(self):
        diameter = 20.0
        roller_diameter = 10.0
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(0, 50, 0)

        gear_ratio = diameter / roller_diameter
        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
            gear_ratio=gear_ratio,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        circ = diameter * math.pi
        expected = (50.0 / circ) * 360.0 * gear_ratio
        assert cmds[1].extra_axes[Axis.A] == pytest.approx(expected)

    def test_reverse_axis(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(0, 50, 0)

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
            reverse=True,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        circ = diameter * math.pi
        expected = -(50.0 / circ) * 360.0
        assert cmds[1].extra_axes[Axis.A] == pytest.approx(expected)

    def test_arc_center_offset_converted(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 0, 0)
        ops.arc_to(0, 10, i=-10, j=0, clockwise=False)

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
        )
        mapping.apply(ops)

        arc = [c for c in ops if isinstance(c, ArcToCommand)][0]
        src_offset_y = 0
        expected_deg = (src_offset_y / (diameter * math.pi)) * 360.0
        assert arc.center_offset[1] == pytest.approx(expected_deg)

    def test_bezier_control_points_converted(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.bezier_to(c1=(10, 20, 0), c2=(10, 40, 0), end=(0, 50, 0))

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
        )
        mapping.apply(ops)

        cmd = [c for c in ops if isinstance(c, BezierToCommand)][0]
        circ = diameter * math.pi
        assert cmd.control1[1] == pytest.approx((20.0 / circ) * 360.0)
        assert cmd.control2[1] == pytest.approx((40.0 / circ) * 360.0)

    def test_quadratic_bezier_control_converted(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(0, 0, 0)
        cmd = QuadraticBezierToCommand(end=(20, 50, 0), control=(10, 30, 0))
        ops.add(cmd)

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
        )
        mapping.apply(ops)

        cmd = [c for c in ops if isinstance(c, QuadraticBezierToCommand)][0]
        circ = diameter * math.pi
        assert cmd.control[1] == pytest.approx((30.0 / circ) * 360.0)

    def test_non_moving_commands_untouched(self):
        ops = Ops()
        ops.move_to(10, 20, 0)
        ops.set_power(0.5)
        ops.line_to(30, 40, 0)

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=25.0,
        )
        mapping.apply(ops)
        assert len(ops._commands) == 3

    def test_axis_position_parks_source_axis(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 100, 0)
        ops.line_to(30, 150, 0)

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
            axis_position=100.0,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[1] == pytest.approx(100.0)
        assert cmds[1].end[1] == pytest.approx(100.0)

        circ = diameter * math.pi
        expected_deg_0 = (100.0 / circ) * 360.0
        expected_deg_50 = (150.0 / circ) * 360.0
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(expected_deg_0)
        assert cmds[1].extra_axes[Axis.A] == pytest.approx(expected_deg_50)

    def test_axis_position_bezier_controls(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(0, 100, 0)
        ops.bezier_to(c1=(10, 120, 0), c2=(10, 140, 0), end=(0, 150, 0))

        mapping = KinematicMapping(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            diameter=diameter,
            axis_position=100.0,
        )
        mapping.apply(ops)

        cmd = [c for c in ops if isinstance(c, BezierToCommand)][0]
        circ = diameter * math.pi
        assert cmd.control1[1] == pytest.approx((120.0 / circ) * 360.0)
        assert cmd.control2[1] == pytest.approx((140.0 / circ) * 360.0)
        assert cmd.end[1] == pytest.approx(100.0)

    def test_axis_position_x_source(self):
        diameter = 40.0
        ops = Ops()
        ops.move_to(50, 10, 0)
        ops.line_to(150, 10, 0)

        mapping = KinematicMapping(
            source_axis=Axis.X,
            rotary_axis=Axis.B,
            diameter=diameter,
            axis_position=50.0,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[0] == pytest.approx(50.0)
        assert cmds[1].end[0] == pytest.approx(50.0)

        circ = diameter * math.pi
        expected = (150.0 / circ) * 360.0
        assert cmds[1].extra_axes[Axis.B] == pytest.approx(expected)


class TestKinematicMappingFromModule:
    def test_true_4th_axis(self):
        rm = RotaryModule()
        rm.set_axis(Axis.A)
        rm.set_source_axis(Axis.Y)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.source_axis == Axis.Y
        assert mapping.rotary_axis == Axis.A
        assert mapping.diameter == 25.0

    def test_axis_replacement(self):
        rm = RotaryModule()
        rm.set_mode(RotaryMode.AXIS_REPLACEMENT)
        rm.set_source_axis(Axis.Y)

        mapping = KinematicMapping.from_rotary_module(rm, 30.0)
        assert mapping is not None
        assert mapping.source_axis == Axis.Y
        assert mapping.rotary_axis == Axis.Y
        assert mapping.diameter == 30.0

    def test_rollers_gear_ratio(self):
        rm = RotaryModule()
        rm.set_source_axis(Axis.Y)
        rm.rotary_type = RotaryType.ROLLERS
        rm.roller_diameter = 10.0

        mapping = KinematicMapping.from_rotary_module(rm, 20.0)
        assert mapping is not None
        assert mapping.gear_ratio == pytest.approx(2.0)

    def test_reverse(self):
        rm = RotaryModule()
        rm.set_source_axis(Axis.Y)
        rm.reverse_axis = True

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.reverse is True

    def test_axis_position_from_module_transform(self):
        rm = RotaryModule()
        rm.set_source_axis(Axis.Y)
        rm.set_position(10, 200, 5)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.axis_position == pytest.approx(200.0)

    def test_axis_position_with_offset(self):
        rm = RotaryModule()
        rm.set_source_axis(Axis.Y)
        rm.set_position(10, 200, 5)
        rm.set_axis_position(0, 50, 0)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.axis_position == pytest.approx(250.0)

    def test_axis_position_x_source(self):
        rm = RotaryModule()
        rm.set_source_axis(Axis.X)
        rm.set_position(100, 50, 0)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.axis_position == pytest.approx(100.0)
