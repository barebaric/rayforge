import math

import pytest

from rayforge.machine.assembly import Assembly, JointType, Link
from rayforge.machine.driver.driver import Axis
from rayforge.simulator.machine_state import MachineState


def _make_state(x=0.0, y=0.0, z=0.0):
    state = MachineState()
    state.axes[Axis.X] = x
    state.axes[Axis.Y] = y
    state.axes[Axis.Z] = z
    return state


def _three_axis_assembly():
    return Assembly(
        [
            Link("base", parent=None, joint_type=JointType.FIXED),
            Link(
                "gantry_x",
                parent="base",
                joint_type=JointType.PRISMATIC,
                joint_axis=(1.0, 0.0, 0.0),
                driver_axis=Axis.X,
            ),
            Link(
                "gantry_y",
                parent="gantry_x",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 1.0, 0.0),
                driver_axis=Axis.Y,
            ),
            Link(
                "laser_head",
                parent="gantry_y",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 0.0, 1.0),
                driver_axis=Axis.Z,
            ),
        ]
    )


def _rotary_assembly(diameter=50.0):
    asm = Assembly(
        [
            Link("base", parent=None, joint_type=JointType.FIXED),
            Link(
                "gantry_x",
                parent="base",
                joint_type=JointType.PRISMATIC,
                joint_axis=(1.0, 0.0, 0.0),
                driver_axis=Axis.X,
            ),
            Link(
                "gantry_y",
                parent="gantry_x",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 1.0, 0.0),
                driver_axis=Axis.Y,
            ),
            Link(
                "laser_head",
                parent="gantry_y",
                joint_type=JointType.PRISMATIC,
                joint_axis=(0.0, 0.0, 1.0),
                driver_axis=Axis.Z,
            ),
            Link("rotary_base", parent="base", joint_type=JointType.FIXED),
            Link(
                "rotary_chuck",
                parent="rotary_base",
                joint_type=JointType.REVOLUTE,
                joint_axis=(1.0, 0.0, 0.0),
                driver_axis=Axis.Y,
            ),
        ]
    )
    asm.set_rotary_diameter(diameter)
    return asm


class TestThreeAxis:
    def test_head_position_origin(self):
        asm = _three_axis_assembly()
        state = _make_state()
        assert asm.head_position(state) == (0.0, 0.0, 0.0)

    def test_head_position_translated(self):
        asm = _three_axis_assembly()
        state = _make_state(x=10.0, y=20.0, z=5.0)
        assert asm.head_position(state) == (10.0, 20.0, 5.0)

    def test_no_rotary(self):
        asm = _three_axis_assembly()
        assert not asm.has_rotary
        assert asm.rotary_diameter is None

    def test_cylinder_angle_zero(self):
        asm = _three_axis_assembly()
        state = _make_state(y=100.0)
        assert asm.cylinder_angle(state) == 0.0

    def test_forward_kinematics_returns_all_links(self):
        asm = _three_axis_assembly()
        state = _make_state(x=1.0, y=2.0, z=3.0)
        poses = asm.forward_kinematics(state)
        assert set(poses.keys()) == {
            "base",
            "gantry_x",
            "gantry_y",
            "laser_head",
        }
        pos, orient = poses["laser_head"]
        assert pos == (1.0, 2.0, 3.0)
        assert orient.shape == (3, 3)


class TestRotary:
    def test_has_rotary(self):
        asm = _rotary_assembly()
        assert asm.has_rotary
        assert asm.rotary_diameter == 50.0

    def test_cylinder_angle_quarter_turn(self):
        diameter = 50.0
        circumference = diameter * math.pi
        asm = _rotary_assembly(diameter)
        state = _make_state(y=circumference / 4)
        angle = asm.cylinder_angle(state)
        assert abs(angle - math.pi / 2) < 1e-9

    def test_head_position_same_as_cartesian(self):
        asm = _rotary_assembly()
        state = _make_state(x=5.0, y=10.0, z=3.0)
        assert asm.head_position(state) == (5.0, 10.0, 3.0)

    def test_cylinder_angle_zero_at_origin(self):
        asm = _rotary_assembly()
        state = _make_state()
        assert asm.cylinder_angle(state) == 0.0


class TestValidation:
    def test_empty_links_raises(self):
        with pytest.raises(ValueError, match="at least one link"):
            Assembly([])

    def test_duplicate_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate link name"):
            Assembly(
                [
                    Link("base", parent=None, joint_type=JointType.FIXED),
                    Link("base", parent="base", joint_type=JointType.FIXED),
                ]
            )

    def test_missing_parent_raises(self):
        with pytest.raises(ValueError, match="unknown parent"):
            Assembly(
                [
                    Link("a", parent="missing", joint_type=JointType.FIXED),
                ]
            )

    def test_no_root_raises(self):
        with pytest.raises(ValueError, match="exactly one root"):
            Assembly(
                [
                    Link("a", parent="b", joint_type=JointType.FIXED),
                    Link("b", parent="a", joint_type=JointType.FIXED),
                ]
            )

    def test_cycle_raises(self):
        with pytest.raises((ValueError, RecursionError)):
            Assembly(
                [
                    Link("base", parent=None, joint_type=JointType.FIXED),
                    Link(
                        "a",
                        parent="base",
                        joint_type=JointType.FIXED,
                    ),
                    Link(
                        "b",
                        parent="c",
                        joint_type=JointType.FIXED,
                    ),
                    Link(
                        "c",
                        parent="b",
                        joint_type=JointType.FIXED,
                    ),
                ]
            )

    def test_multiple_roots_raises(self):
        with pytest.raises(ValueError, match="exactly one root link"):
            Assembly(
                [
                    Link("root1", parent=None, joint_type=JointType.FIXED),
                    Link("root2", parent=None, joint_type=JointType.FIXED),
                ]
            )

    def test_prismatic_without_driver_raises(self):
        with pytest.raises(ValueError, match="requires a driver_axis"):
            Link(
                "a",
                parent=None,
                joint_type=JointType.PRISMATIC,
                joint_axis=(1, 0, 0),
            )

    def test_fixed_with_driver_raises(self):
        with pytest.raises(ValueError, match="cannot have a driver_axis"):
            Link(
                "a",
                parent=None,
                joint_type=JointType.FIXED,
                driver_axis=Axis.X,
            )

    def test_head_position_no_laser_head_raises(self):
        asm = Assembly(
            [
                Link("base", parent=None, joint_type=JointType.FIXED),
            ]
        )
        with pytest.raises(ValueError, match="no 'laser_head' link"):
            asm.head_position(MachineState())


class TestFiveAxis:
    def test_tilt_head_rotation(self):
        asm = Assembly(
            [
                Link("base", parent=None, joint_type=JointType.FIXED),
                Link(
                    "gantry_x",
                    parent="base",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(1, 0, 0),
                    driver_axis=Axis.X,
                ),
                Link(
                    "gantry_y",
                    parent="gantry_x",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(0, 1, 0),
                    driver_axis=Axis.Y,
                ),
                Link(
                    "gantry_z",
                    parent="gantry_y",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(0, 0, 1),
                    driver_axis=Axis.Z,
                ),
                Link(
                    "tilt_head",
                    parent="gantry_z",
                    joint_type=JointType.REVOLUTE,
                    joint_axis=(0, 0, 1),
                    driver_axis=Axis.B,
                ),
                Link(
                    "laser_head",
                    parent="tilt_head",
                    joint_type=JointType.FIXED,
                ),
            ]
        )
        state = MachineState()
        state.axes[Axis.B] = 45.0
        poses = asm.forward_kinematics(state)
        assert "tilt_head" in poses
        assert "laser_head" in poses
