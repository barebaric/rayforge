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
from rayforge.machine.models.rotary_module import RotaryMode, RotaryType


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


class TestAxisMapperReplacementRaw:
    def test_replacement_raw_y_unchanged(self):
        ops = Ops()
        ops.move_to(10, 50, 0)
        ops.line_to(30, 100, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=0.0,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert len(cmds) == 2
        assert cmds[0].end[1] == pytest.approx(50.0)
        assert cmds[1].end[1] == pytest.approx(100.0)
        assert cmds[0].extra_axes == {}
        assert cmds[1].extra_axes == {}

    def test_replacement_raw_no_park_move(self):
        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=0.0,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert len(cmds) == 1


class TestAxisMapperReplacementDegrees:
    def test_replacement_produces_degrees_in_extra_axes(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert len(cmds) == 1
        assert cmds[0].end[1] == pytest.approx(0.0)
        expected_deg = (50.0 / (diameter * math.pi)) * 360.0
        assert cmds[0].extra_axes[Axis.Y] == pytest.approx(expected_deg)

    def test_replacement_x_axis(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(50, 10, 0)

        mapper = AxisMapper(
            source_axis=Axis.X,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[0] == pytest.approx(0.0)
        expected_deg = (50.0 / (diameter * math.pi)) * 360.0
        assert cmds[0].extra_axes[Axis.X] == pytest.approx(expected_deg)

    def test_replacement_bezier_control_points_in_degrees(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.bezier_to(c1=(10, 20, 0), c2=(20, 30, 0), end=(30, 10, 0))

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        mapper.run(ops)

        beziers = [c for c in ops if isinstance(c, BezierToCommand)]
        assert len(beziers) == 1
        b = beziers[0]
        circ = diameter * math.pi
        assert b.control1[1] == pytest.approx((20.0 / circ) * 360.0)
        assert b.control2[1] == pytest.approx((30.0 / circ) * 360.0)
        assert b.end[1] == pytest.approx(0.0)

    def test_replacement_arc_center_offset_in_degrees(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(10, 20, 0)
        ops.arc_to(30, 40, i=5.0, j=3.0, clockwise=True, z=0.0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        mapper.run(ops)

        arcs = [c for c in ops if isinstance(c, ArcToCommand)]
        assert len(arcs) == 1
        circ = diameter * math.pi
        assert arcs[0].center_offset[1] == pytest.approx((3.0 / circ) * 360.0)
        assert arcs[0].center_offset[0] == pytest.approx(5.0)

    def test_replacement_no_park_move(self):
        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=25.0,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=100.0,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert len(cmds) == 1


class TestAxisMapperRollerDiameter:
    def test_roller_degrees_4th_axis(self):
        object_diameter = 70.0
        roller_diameter = 20.0

        ops = Ops()
        ops.move_to(0, math.pi * object_diameter, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=object_diameter,
            rotary_type=RotaryType.ROLLERS,
            roller_diameter=roller_diameter,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MoveToCommand)]
        gear_ratio = object_diameter / roller_diameter
        expected = 360.0 * gear_ratio
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(expected)

    def test_roller_replacement_degrees(self):
        object_diameter = 70.0
        roller_diameter = 20.0
        mm_per_rot = 100.0

        ops = Ops()
        source_val = math.pi * object_diameter
        ops.move_to(10, source_val, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=object_diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
            rotary_type=RotaryType.ROLLERS,
            roller_diameter=roller_diameter,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[1] == pytest.approx(0.0)
        gear_ratio = object_diameter / roller_diameter
        expected_deg = (
            (source_val / (object_diameter * math.pi)) * 360.0 * gear_ratio
        )
        assert cmds[0].extra_axes[Axis.Y] == pytest.approx(expected_deg)

    def test_jaws_ignores_roller_diameter(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper_jaws = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            rotary_type=RotaryType.JAWS,
            roller_diameter=20.0,
        )
        ops_jaws = Ops()
        ops_jaws.move_to(10, 50, 0)
        mapper_jaws.run(ops_jaws)

        mapper_baseline = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            rotary_type=RotaryType.JAWS,
            roller_diameter=0.0,
        )
        ops_baseline = Ops()
        ops_baseline.move_to(10, 50, 0)
        mapper_baseline.run(ops_baseline)

        cmds_jaws = [c for c in ops_jaws if isinstance(c, MovingCommand)]
        cmds_baseline = [
            c for c in ops_baseline if isinstance(c, MovingCommand)
        ]
        assert cmds_jaws[0].extra_axes[Axis.A] == pytest.approx(
            cmds_baseline[0].extra_axes[Axis.A]
        )


class TestAxisMapperReverseAxis:
    def test_reverse_4th_axis_negates_degrees(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            reverse_axis=True,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        expected = -((50.0 / (diameter * math.pi)) * 360.0)
        assert cmds[0].extra_axes[Axis.A] == pytest.approx(expected)

    def test_reverse_replacement_negates_degrees(self):
        diameter = 25.0
        mm_per_rot = 100.0
        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
            reverse_axis=True,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].end[1] == pytest.approx(0.0)
        expected = -((50.0 / (diameter * math.pi)) * 360.0)
        assert cmds[0].extra_axes[Axis.Y] == pytest.approx(expected)

    def test_no_reverse_produces_positive(self):
        diameter = 25.0
        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            reverse_axis=False,
        )
        mapper.run(ops)

        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        assert cmds[0].extra_axes[Axis.A] > 0


class TestDegreesToScaledMuPass:
    def _run_full_pipeline(self, ops, mapper, mm_per_rot):
        mapper.run(ops)
        cmds = [c for c in ops if isinstance(c, MovingCommand)]
        AxisMapper.degrees_to_scaled_mu_pass(
            cmds, mapper.source_axis, mm_per_rot
        )
        return cmds

    def test_scaled_y_after_downstream_pass(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        cmds = self._run_full_pipeline(ops, mapper, mm_per_rot)

        expected = 50.0 * mm_per_rot / (math.pi * diameter)
        assert cmds[0].end[1] == pytest.approx(expected)
        assert Axis.Y not in cmds[0].extra_axes

    def test_scaled_x_after_downstream_pass(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(50, 10, 0)

        mapper = AxisMapper(
            source_axis=Axis.X,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        cmds = self._run_full_pipeline(ops, mapper, mm_per_rot)

        expected = 50.0 * mm_per_rot / (math.pi * diameter)
        assert cmds[0].end[0] == pytest.approx(expected)

    def test_scaled_bezier_after_downstream_pass(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(0, 0, 0)
        ops.bezier_to(c1=(10, 20, 0), c2=(20, 30, 0), end=(30, 10, 0))

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        cmds = self._run_full_pipeline(ops, mapper, mm_per_rot)

        beziers = [c for c in cmds if isinstance(c, BezierToCommand)]
        scale = mm_per_rot / (math.pi * diameter)
        assert beziers[0].control1[1] == pytest.approx(20.0 * scale)
        assert beziers[0].control2[1] == pytest.approx(30.0 * scale)
        assert beziers[0].end[1] == pytest.approx(10.0 * scale)

    def test_scaled_arc_after_downstream_pass(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(10, 20, 0)
        ops.arc_to(30, 40, i=5.0, j=3.0, clockwise=True, z=0.0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
        )
        cmds = self._run_full_pipeline(ops, mapper, mm_per_rot)

        arcs = [c for c in cmds if isinstance(c, ArcToCommand)]
        scale = mm_per_rot / (math.pi * diameter)
        assert arcs[0].center_offset[1] == pytest.approx(3.0 * scale)
        assert arcs[0].center_offset[0] == pytest.approx(5.0)

    def test_roller_scaled_after_downstream_pass(self):
        object_diameter = 70.0
        roller_diameter = 20.0
        mm_per_rot = 100.0
        source_val = math.pi * object_diameter

        ops = Ops()
        ops.move_to(10, source_val, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=object_diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
            rotary_type=RotaryType.ROLLERS,
            roller_diameter=roller_diameter,
        )
        cmds = self._run_full_pipeline(ops, mapper, mm_per_rot)

        gear_ratio = object_diameter / roller_diameter
        expected = (
            source_val * mm_per_rot / (math.pi * object_diameter) * gear_ratio
        )
        assert cmds[0].end[1] == pytest.approx(expected)

    def test_reverse_scaled_after_downstream_pass(self):
        diameter = 25.0
        mm_per_rot = 100.0

        ops = Ops()
        ops.move_to(10, 50, 0)

        mapper = AxisMapper(
            source_axis=Axis.Y,
            rotary_axis=Axis.A,
            rotary_diameter=diameter,
            mode=RotaryMode.AXIS_REPLACEMENT,
            mu_per_rotation=mm_per_rot,
            reverse_axis=True,
        )
        cmds = self._run_full_pipeline(ops, mapper, mm_per_rot)

        expected = -(50.0 * mm_per_rot / (math.pi * diameter))
        assert cmds[0].end[1] == pytest.approx(expected)

    def test_downstream_pass_noop_when_no_degrees(self):
        mm_per_rot = 100.0
        commands = []
        AxisMapper.degrees_to_scaled_mu_pass(commands, Axis.Y, mm_per_rot)


class TestReplacementRoundtrip:
    @pytest.mark.parametrize(
        "mu,diameter,mm_per_rot,gear_ratio,reverse",
        [
            (50.0, 25.0, 100.0, 1.0, False),
            (0.0, 25.0, 100.0, 1.0, False),
            (100.0, 10.0, 50.0, 1.0, False),
            (50.0, 25.0, 200.0, 1.0, False),
            (78.5398, 25.0, 100.0, 1.0, False),
            (50.0, 25.0, 100.0, 3.5, False),
            (100.0, 70.0, 50.0, 3.5, False),
            (50.0, 25.0, 100.0, 1.0, True),
            (50.0, 25.0, 100.0, 3.5, True),
        ],
    )
    def test_two_step_equals_direct(
        self, mu, diameter, mm_per_rot, gear_ratio, reverse
    ):
        from rayforge.machine.kinematic_math import KinematicMath

        direct = KinematicMath.mu_to_scaled_mu(
            mu, diameter, mm_per_rot, gear_ratio=gear_ratio, reverse=reverse
        )

        degrees = KinematicMath.mu_to_degrees(
            mu, diameter, gear_ratio=gear_ratio, reverse=reverse
        )
        two_step = KinematicMath.degrees_to_scaled_mu(degrees, mm_per_rot)

        assert two_step == pytest.approx(direct)
