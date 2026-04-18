import math

import numpy as np
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

    def test_z_does_not_affect_degrees(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 0, 5)
        ops.line_to(10, 50, 5)

        mapping = KinematicMapping(
            rotary_axis=Axis.A,
            diameter=diameter,
        )
        mapping.apply(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        expected = (50.0 / (diameter * math.pi)) * 360.0
        assert cmds[1].extra_axes[Axis.A] == pytest.approx(expected)

    def test_gear_ratio(self):
        diameter = 20.0
        roller_diameter = 10.0
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(0, 50, 0)

        gear_ratio = diameter / roller_diameter
        mapping = KinematicMapping(
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


class TestKinematicMappingFromModule:
    def test_true_4th_axis(self):
        rm = RotaryModule()
        rm.set_axis(Axis.A)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.rotary_axis == Axis.A
        assert mapping.diameter == 25.0

    def test_axis_replacement(self):
        rm = RotaryModule()
        rm.set_mode(RotaryMode.AXIS_REPLACEMENT)

        mapping = KinematicMapping.from_rotary_module(rm, 30.0)
        assert mapping is not None
        assert mapping.rotary_axis == Axis.Y
        assert mapping.diameter == 30.0

    def test_rollers_gear_ratio(self):
        rm = RotaryModule()
        rm.rotary_type = RotaryType.ROLLERS
        rm.roller_diameter = 10.0

        mapping = KinematicMapping.from_rotary_module(rm, 20.0)
        assert mapping is not None
        assert mapping.gear_ratio == pytest.approx(2.0)

    def test_reverse(self):
        rm = RotaryModule()
        rm.reverse_axis = True

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.reverse is True

    def test_axis_position_from_module_transform(self):
        rm = RotaryModule()
        rm.set_position(10, 200, 5)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.axis_position == pytest.approx(200.0)

    def test_axis_position_with_offset(self):
        rm = RotaryModule()
        rm.set_position(10, 200, 5)
        rm.set_axis_position(0, 50, 0)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.axis_position == pytest.approx(250.0)

    def test_tilted_cylinder_tracks_center(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.line_to(50, 100, 0)

        ap3d = np.array([0.0, 200.0, 5.0])
        cdir = np.array([0.3, 0.0, 0.0])
        cdir /= np.linalg.norm(cdir)

        mapping = KinematicMapping(
            rotary_axis=Axis.A,
            diameter=diameter,
            axis_position_3d=ap3d,
            cylinder_dir=cdir,
        )

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        mapping.apply(ops)

        expected_si_0 = 200.0 + 0.0 * cdir[1]
        expected_si_1 = 200.0 + 100.0 * cdir[1]
        assert cmds[0].end[1] == pytest.approx(expected_si_0)
        assert cmds[1].end[1] == pytest.approx(expected_si_1)

    def test_cylinder_dir_from_rotated_module(self):
        rm = RotaryModule()
        rm.set_position(10, 200, 5)
        rm.set_rotation(0, 0, 45)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.cylinder_dir is not None
        norm = np.linalg.norm(mapping.cylinder_dir)
        assert norm == pytest.approx(1.0, abs=1e-6)

    def test_default_cylinder_dir_no_tilt(self):
        rm = RotaryModule()
        rm.set_position(10, 200, 5)

        mapping = KinematicMapping.from_rotary_module(rm, 25.0)
        assert mapping is not None
        assert mapping.cylinder_dir[0] == pytest.approx(1.0)
        assert mapping.cylinder_dir[1] == pytest.approx(0.0)
        assert mapping.cylinder_dir[2] == pytest.approx(0.0)
