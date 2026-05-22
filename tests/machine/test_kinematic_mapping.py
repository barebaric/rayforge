import math

import numpy as np
import pytest
from raygeo.ops import Ops
from raygeo.ops.axis import Axis
from raygeo.ops.types import CommandType

from rayforge.machine.kinematic_mapping import KinematicMapping
from rayforge.machine.models.rotary_module import (
    RotaryMode,
    RotaryModule,
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

        move_idx = ops.indices_of(CommandType.MOVE_TO)[0]
        line_idx = ops.indices_of(CommandType.LINE_TO)[0]
        assert ops.endpoint(move_idx)[1] == pytest.approx(0.0)
        assert ops.endpoint(line_idx)[1] == pytest.approx(0.0)

        expected_deg_0 = (0.0 / (diameter * math.pi)) * 360.0
        expected_deg_50 = (50.0 / (diameter * math.pi)) * 360.0
        ea0 = ops.extra_axes(move_idx)
        ea1 = ops.extra_axes(line_idx)
        assert ea0 is not None
        assert ea1 is not None
        assert ea0[Axis.A] == pytest.approx(expected_deg_0)
        assert ea1[Axis.A] == pytest.approx(expected_deg_50)

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

        line_idx = ops.indices_of(CommandType.LINE_TO)[0]
        expected = (50.0 / (diameter * math.pi)) * 360.0
        ea = ops.extra_axes(line_idx)
        assert ea is not None
        assert ea[Axis.A] == pytest.approx(expected)

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

        line_idx = ops.indices_of(CommandType.LINE_TO)[0]
        circ = diameter * math.pi
        expected = (50.0 / circ) * 360.0 * gear_ratio
        ea = ops.extra_axes(line_idx)
        assert ea is not None
        assert ea[Axis.A] == pytest.approx(expected)

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

        line_idx = ops.indices_of(CommandType.LINE_TO)[0]
        circ = diameter * math.pi
        expected = -(50.0 / circ) * 360.0
        ea = ops.extra_axes(line_idx)
        assert ea is not None
        assert ea[Axis.A] == pytest.approx(expected)

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

        arc_idx = ops.indices_of(CommandType.ARC_TO)[0]
        src_offset_y = 0
        expected_deg = (src_offset_y / (diameter * math.pi)) * 360.0
        assert ops.arc_params(arc_idx)[1] == pytest.approx(expected_deg)

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

        bez_idx = ops.indices_of(CommandType.BEZIER_TO)[0]
        circ = diameter * math.pi
        c1, c2 = ops.bezier_params(bez_idx)
        assert c1[1] == pytest.approx((20.0 / circ) * 360.0)
        assert c2[1] == pytest.approx((40.0 / circ) * 360.0)

    def test_quadratic_bezier_control_converted(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.quadratic_bezier_to(control=(10, 30, 0), end=(20, 50, 0))

        mapping = KinematicMapping(
            rotary_axis=Axis.A,
            diameter=diameter,
        )
        mapping.apply(ops)

        qb_idx = ops.indices_of(CommandType.QUADRATIC_BEZIER_TO)[0]
        circ = diameter * math.pi
        ctrl = ops.quadratic_bezier_params(qb_idx)
        assert ctrl[1] == pytest.approx((30.0 / circ) * 360.0)

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
        assert ops.len() == 3

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

        move_idx = ops.indices_of(CommandType.MOVE_TO)[0]
        line_idx = ops.indices_of(CommandType.LINE_TO)[0]
        assert ops.endpoint(move_idx)[1] == pytest.approx(100.0)
        assert ops.endpoint(line_idx)[1] == pytest.approx(100.0)

        circ = diameter * math.pi
        expected_deg_0 = (100.0 / circ) * 360.0
        expected_deg_50 = (150.0 / circ) * 360.0
        ea0 = ops.extra_axes(move_idx)
        ea1 = ops.extra_axes(line_idx)
        assert ea0 is not None
        assert ea1 is not None
        assert ea0[Axis.A] == pytest.approx(expected_deg_0)
        assert ea1[Axis.A] == pytest.approx(expected_deg_50)

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

        bez_idx = ops.indices_of(CommandType.BEZIER_TO)[0]
        circ = diameter * math.pi
        c1, c2 = ops.bezier_params(bez_idx)
        assert c1[1] == pytest.approx((120.0 / circ) * 360.0)
        assert c2[1] == pytest.approx((140.0 / circ) * 360.0)
        assert ops.endpoint(bez_idx)[1] == pytest.approx(100.0)


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

        move_idx = ops.indices_of(CommandType.MOVE_TO)[0]
        line_idx = ops.indices_of(CommandType.LINE_TO)[0]
        mapping.apply(ops)

        expected_si_0 = 200.0 + 0.0 * cdir[1]
        expected_si_1 = 200.0 + 100.0 * cdir[1]
        assert ops.endpoint(move_idx)[1] == pytest.approx(expected_si_0)
        assert ops.endpoint(line_idx)[1] == pytest.approx(expected_si_1)

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
