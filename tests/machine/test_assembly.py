import math

import pytest

from rayforge.machine.assembly import Assembly, JointType, Link, LinkRole
from rayforge.machine.driver.driver import Axis
from rayforge.simulator.machine_state import MachineState


def _make_state(x=0.0, y=0.0, z=0.0, a=0.0, b=0.0):
    state = MachineState()
    state.axes[Axis.X] = x
    state.axes[Axis.Y] = y
    state.axes[Axis.Z] = z
    state.axes[Axis.A] = a
    state.axes[Axis.B] = b
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
                role=LinkRole.HEAD,
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
                role=LinkRole.HEAD,
            ),
            Link("rotary_base", parent="base", joint_type=JointType.FIXED),
            Link(
                "rotary_chuck",
                parent="rotary_base",
                joint_type=JointType.REVOLUTE,
                joint_axis=(1.0, 0.0, 0.0),
                driver_axis=Axis.Y,
                role=LinkRole.CHUCK,
            ),
        ]
    )
    asm.set_rotary_diameter(diameter)
    return asm


class TestThreeAxis:
    def test_head_positions_origin(self):
        asm = _three_axis_assembly()
        state = _make_state()
        heads = asm.head_positions(state)
        assert heads == {"laser_head": (0.0, 0.0, 0.0)}

    def test_head_positions_translated(self):
        asm = _three_axis_assembly()
        state = _make_state(x=10.0, y=20.0, z=5.0)
        heads = asm.head_positions(state)
        assert heads == {"laser_head": (10.0, 20.0, 5.0)}

    def test_no_rotary(self):
        asm = _three_axis_assembly()
        assert not asm.has_rotary
        assert asm.rotary_diameter is None

    def test_chuck_angles_empty(self):
        asm = _three_axis_assembly()
        angles = asm.chuck_angles(_make_state(y=100.0))
        assert angles == {}

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

    def test_chuck_angles_quarter_turn(self):
        diameter = 50.0
        circumference = diameter * math.pi
        asm = _rotary_assembly(diameter)
        state = _make_state(y=circumference / 4)
        angles = asm.chuck_angles(state)
        assert abs(angles["rotary_chuck"] - math.pi / 2) < 1e-9

    def test_head_positions_same_as_cartesian(self):
        asm = _rotary_assembly()
        state = _make_state(x=5.0, y=10.0, z=3.0)
        heads = asm.head_positions(state)
        assert heads == {"laser_head": (5.0, 10.0, 3.0)}

    def test_chuck_angles_zero_at_origin(self):
        asm = _rotary_assembly()
        state = _make_state()
        angles = asm.chuck_angles(state)
        assert angles["rotary_chuck"] == 0.0


class TestMultipleHeads:
    def test_two_heads(self):
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
                    "head_a",
                    parent="gantry_x",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(0.0, 1.0, 0.0),
                    driver_axis=Axis.Y,
                    role=LinkRole.HEAD,
                ),
                Link(
                    "head_b",
                    parent="gantry_x",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(0.0, 0.0, 1.0),
                    driver_axis=Axis.Z,
                    role=LinkRole.HEAD,
                ),
            ]
        )
        state = _make_state(x=5.0, y=10.0, z=3.0)
        heads = asm.head_positions(state)
        assert "head_a" in heads
        assert "head_b" in heads
        assert heads["head_a"] == (5.0, 10.0, 0.0)
        assert heads["head_b"] == (5.0, 0.0, 3.0)

    def test_get_links_by_role(self):
        asm = Assembly(
            [
                Link("base", parent=None, joint_type=JointType.FIXED),
                Link(
                    "head_a",
                    parent="base",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(1.0, 0.0, 0.0),
                    driver_axis=Axis.X,
                    role=LinkRole.HEAD,
                ),
                Link(
                    "head_b",
                    parent="base",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(0.0, 1.0, 0.0),
                    driver_axis=Axis.Y,
                    role=LinkRole.HEAD,
                ),
            ]
        )
        head_links = asm.get_links_by_role(LinkRole.HEAD)
        assert len(head_links) == 2
        names = {link.name for link in head_links}
        assert names == {"head_a", "head_b"}
        chuck_links = asm.get_links_by_role(LinkRole.CHUCK)
        assert chuck_links == []


class TestMultipleChucks:
    def test_two_chucks(self):
        asm = Assembly(
            [
                Link("base", parent=None, joint_type=JointType.FIXED),
                Link(
                    "head",
                    parent="base",
                    joint_type=JointType.PRISMATIC,
                    joint_axis=(1.0, 0.0, 0.0),
                    driver_axis=Axis.X,
                    role=LinkRole.HEAD,
                ),
                Link(
                    "chuck_a",
                    parent="base",
                    joint_type=JointType.REVOLUTE,
                    joint_axis=(1.0, 0.0, 0.0),
                    driver_axis=Axis.Y,
                    role=LinkRole.CHUCK,
                ),
                Link(
                    "chuck_b",
                    parent="base",
                    joint_type=JointType.REVOLUTE,
                    joint_axis=(0.0, 1.0, 0.0),
                    driver_axis=Axis.A,
                    role=LinkRole.CHUCK,
                ),
            ]
        )
        asm.set_rotary_diameter(25.0)
        state = _make_state(y=25.0 * math.pi, a=25.0 * math.pi / 2)
        angles = asm.chuck_angles(state)
        assert abs(angles["chuck_a"] - 2 * math.pi) < 1e-9
        assert abs(angles["chuck_b"] - math.pi) < 1e-9

    def test_get_links_by_role_chuck(self):
        asm = _rotary_assembly()
        chucks = asm.get_links_by_role(LinkRole.CHUCK)
        assert len(chucks) == 1
        assert chucks[0].name == "rotary_chuck"

    def test_get_link_existing(self):
        asm = _three_axis_assembly()
        link = asm.get_link("gantry_y")
        assert link is not None
        assert link.name == "gantry_y"

    def test_get_link_missing_returns_none(self):
        asm = _three_axis_assembly()
        assert asm.get_link("nonexistent") is None


class TestNoRole:
    def test_head_positions_raises_when_no_head_role(self):
        asm = Assembly(
            [
                Link("base", parent=None, joint_type=JointType.FIXED),
            ]
        )
        with pytest.raises(ValueError, match="no links with role HEAD"):
            asm.head_positions(MachineState())


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
        with pytest.raises(ValueError, match="exactly one root"):
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
                    "tilt",
                    parent="gantry_z",
                    joint_type=JointType.REVOLUTE,
                    joint_axis=(0, 0, 1),
                    driver_axis=Axis.B,
                ),
                Link(
                    "laser_head",
                    parent="tilt",
                    joint_type=JointType.FIXED,
                    role=LinkRole.HEAD,
                ),
            ]
        )
        state = MachineState()
        state.axes[Axis.B] = 45.0
        poses = asm.forward_kinematics(state)
        assert "tilt" in poses
        assert "laser_head" in poses
        heads = asm.head_positions(state)
        assert "laser_head" in heads
