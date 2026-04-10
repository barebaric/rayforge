import math

import pytest

from rayforge.core.ops.axis import Axis
from rayforge.core.ops import Ops
from rayforge.core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    MoveToCommand,
    MovingCommand,
)
from rayforge.pipeline.transformer.axis_mapper import AxisMapper


class TestAxisMapperIdentity:
    def test_no_rotary_config_is_noop(self):
        ops = Ops()
        ops.move_to(10, 20, 0)
        ops.line_to(30, 40, 0)
        mapper = AxisMapper(enabled=False)
        mapper.run(ops)
        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end == pytest.approx((10.0, 20.0, 0.0))
        assert cmds[1].end == pytest.approx((30.0, 40.0, 0.0))
        assert cmds[0].extra_axes == {}
        assert cmds[1].extra_axes == {}


class TestAxisMapperDegrees:
    def test_y_to_a_conversion(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 0, 0)
        ops.line_to(30, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[1] == pytest.approx(0.0)
        assert cmds[1].end[1] == pytest.approx(0.0)

        expected_deg_0 = (0.0 / (diameter * math.pi)) * 360.0
        expected_deg_50 = (50.0 / (diameter * math.pi)) * 360.0
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(expected_deg_0)
        assert cmds[1].extra_axes[Axis.A] == pytest.approx(expected_deg_50)

    def test_z_depth_affects_conversion(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 50, -2.0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        effective_d = diameter + 2.0 * (-2.0)
        circ = effective_d * math.pi
        expected = (50.0 / circ) * 360.0
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(expected)

    def test_zero_effective_diameter(self):
        ops = Ops()
        ops.move_to(10, 50, -25.0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(0.0)


class TestAxisMapperPark:
    def test_park_move_inserted(self):
        ops = Ops()
        ops.set_power(0.5)
        ops.move_to(10, 50, 0)
        ops.line_to(30, 100, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
            has_physical_source=True,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert len(cmds) == 3

        assert isinstance(cmds[0], MoveToCommand)
        assert cmds[0].end[1] == pytest.approx(0.0)
        assert cmds[0].extra_axes == {}

        assert cmds[1].end[1] == pytest.approx(0.0)
        assert Axis.A in cmds[1].extra_axes

    def test_no_park_without_physical_source(self):
        ops = Ops()
        ops.move_to(10, 50, 0)
        ops.line_to(30, 100, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
            has_physical_source=False,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert len(cmds) == 2


class TestAxisMapperArcs:
    def test_arc_center_offset_converted(self):
        ops = Ops()
        ops.move_to(10, 20, 0)
        ops.arc_to(30, 40, i=5.0, j=3.0, clockwise=True, z=0.0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
        )
        mapper.run(ops)

        arcs = [c for c in ops if isinstance(c, ArcToCommand)]
        assert len(arcs) == 1
        arc = arcs[0]

        circ = 25.0 * math.pi
        expected_j = (3.0 / circ) * 360.0
        assert arc.center_offset[1] == pytest.approx(expected_j)
        assert arc.center_offset[0] == pytest.approx(5.0)


class TestAxisMapperBezier:
    def test_bezier_control_points_converted(self):
        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.bezier_to(c1=(10, 20, 0), c2=(20, 30, 0), end=(30, 10, 0))

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
        )
        mapper.run(ops)

        beziers = [c for c in ops if isinstance(c, BezierToCommand)]
        assert len(beziers) == 1
        b = beziers[0]

        circ = 25.0 * math.pi
        assert b.control1[0] == pytest.approx(10.0)
        assert b.control1[1] == pytest.approx((20.0 / circ) * 360.0)
        assert b.control2[1] == pytest.approx((30.0 / circ) * 360.0)


class TestAxisMapperNonMovingPassThrough:
    def test_state_commands_untouched(self):
        ops = Ops()
        ops.set_power(0.5)
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
        )
        mapper.run(ops)

        state_cmds = [c for c in ops if c.is_state_command()]
        assert len(state_cmds) == 1


class TestAxisMapperDegreeFormulaMatchesGcode:
    def test_bit_identical_to_gcode_encoder_formula(self):
        diameter = 25.0
        y_val = 78.53981633974483
        z_val = 0.0

        expected = (y_val / (diameter * math.pi)) * 360.0

        ops = Ops()
        ops.move_to(10, y_val, z_val)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MoveToCommand)]
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(expected)
