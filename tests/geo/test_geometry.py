from copy import deepcopy
import pytest
import math
import numpy as np
from typing import cast

from rayforge.core.geo import (
    Geometry,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
)
from rayforge.core.geo.query import get_total_distance


def _create_translate_matrix(x, y, z):
    """Creates a NumPy translation matrix."""
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_scale_matrix(sx, sy, sz):
    """Creates a NumPy scaling matrix."""
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_z_rotate_matrix(angle_rad):
    """Creates a NumPy Z-axis rotation matrix."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


@pytest.fixture
def empty_geometry():
    return Geometry()


@pytest.fixture
def sample_geometry():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.arc_to(20, 0, i=5, j=-10)
    return geo


def test_initialization(empty_geometry):
    assert len(empty_geometry.commands) == 0
    assert empty_geometry.last_move_to == (0.0, 0.0, 0.0)


def test_add_commands(empty_geometry):
    empty_geometry.move_to(5, 5)
    assert len(empty_geometry.commands) == 1
    assert isinstance(empty_geometry.commands[0], MoveToCommand)

    empty_geometry.line_to(10, 10)
    assert isinstance(empty_geometry.commands[1], LineToCommand)


def test_clear_commands(sample_geometry):
    sample_geometry.clear()
    assert len(sample_geometry.commands) == 0


def test_move_to(sample_geometry):
    sample_geometry.move_to(15, 15)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, MoveToCommand)
    assert last_cmd.end == (15.0, 15.0, 0.0)


def test_line_to(sample_geometry):
    sample_geometry.line_to(20, 20)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, 0.0)


def test_close_path(sample_geometry):
    sample_geometry.move_to(5, 5, -1.0)
    sample_geometry.close_path()
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == sample_geometry.last_move_to
    assert last_cmd.end == (5.0, 5.0, -1.0)


def test_arc_to(sample_geometry):
    sample_geometry.arc_to(5, 5, 2, 3, clockwise=False)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, ArcToCommand)
    assert last_cmd.end == (5.0, 5.0, 0.0)
    assert last_cmd.clockwise is False


def test_serialization_deserialization(sample_geometry):
    geo_dict = sample_geometry.to_dict()
    new_geo = Geometry.from_dict(geo_dict)

    assert len(new_geo.commands) == len(sample_geometry.commands)
    assert new_geo.last_move_to == sample_geometry.last_move_to
    for cmd1, cmd2 in zip(new_geo.commands, sample_geometry.commands):
        assert type(cmd1) is type(cmd2)
        assert cmd1.end == cmd2.end


def test_from_dict_ignores_state_commands():
    geo_dict = {
        "commands": [
            {"type": "MoveToCommand", "end": [0, 0, 0]},
            {"type": "SetPowerCommand", "power": 500},
            {"type": "LineToCommand", "end": [10, 10, 0]},
        ],
        "last_move_to": [0, 0, 0],
    }
    geo = Geometry.from_dict(geo_dict)
    assert len(geo.commands) == 2
    assert isinstance(geo.commands[0], MoveToCommand)
    assert isinstance(geo.commands[1], LineToCommand)


def test_transform_identity():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)
    original_geo = deepcopy(geo)

    identity_matrix = np.identity(4, dtype=float)
    geo.transform(identity_matrix)

    arc_cmd = cast(ArcToCommand, geo.commands[1])
    orig_arc_cmd = cast(ArcToCommand, original_geo.commands[1])

    assert geo.commands[0].end == pytest.approx(original_geo.commands[0].end)
    assert arc_cmd.end == pytest.approx(orig_arc_cmd.end)
    assert arc_cmd.center_offset == pytest.approx(orig_arc_cmd.center_offset)
    assert geo.last_move_to == pytest.approx(original_geo.last_move_to)


def test_transform_translate():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)

    translate_matrix = _create_translate_matrix(10, -5, 15)
    geo.transform(translate_matrix)
    arc_cmd = cast(ArcToCommand, geo.commands[1])

    assert geo.commands[0].end == pytest.approx((20, 15, 45))
    assert arc_cmd.end == pytest.approx((60, 55, 55))
    assert arc_cmd.center_offset == pytest.approx((5, 7))
    assert geo.last_move_to == pytest.approx((20, 15, 45))


def test_transform_scale_non_uniform_linearizes_arc():
    geo = Geometry()
    geo.move_to(10, 20, 5)
    geo.arc_to(22, 22, i=5, j=7, z=-10)
    scale_matrix = _create_scale_matrix(2, 3, 4)
    geo.transform(scale_matrix)

    assert geo.commands[0].end == pytest.approx((20, 60, 20))
    # Arcs are linearized on non-uniform scale
    assert isinstance(geo.commands[1], LineToCommand)
    final_cmd = geo.commands[-1]
    assert final_cmd.end is not None
    final_point = final_cmd.end
    expected_final_point = (22 * 2, 22 * 3, -10 * 4)
    assert final_point == pytest.approx(expected_final_point)


def test_transform_rotate_preserves_z():
    geo = Geometry()
    geo.move_to(10, 10, -5)
    rotate_matrix = _create_z_rotate_matrix(math.radians(90))
    geo.transform(rotate_matrix)
    assert geo.commands[0].end is not None
    x, y, z = geo.commands[0].end
    assert z == -5
    assert x == pytest.approx(-10)
    assert y == pytest.approx(10)


def test_copy_method(sample_geometry):
    """Tests the deep copy functionality of the Geometry class."""
    original_geo = sample_geometry
    copied_geo = original_geo.copy()

    # Check for initial equality and deep copy semantics
    assert copied_geo is not original_geo
    assert copied_geo.commands is not original_geo.commands
    assert len(copied_geo.commands) == len(original_geo.commands)
    assert copied_geo.last_move_to == original_geo.last_move_to
    # Check a specific command's value to ensure it was copied
    original_line_to = cast(LineToCommand, original_geo.commands[1])
    copied_line_to = cast(LineToCommand, copied_geo.commands[1])
    assert copied_line_to.end == original_line_to.end

    # Modify the original and check that the copy is unaffected
    original_geo.line_to(100, 100)  # Adds a 4th command
    cast(MoveToCommand, original_geo.commands[0]).end = (99, 99, 99)

    # The copy should still have the original number of commands
    assert len(copied_geo.commands) == 3
    # The copy's first command should have its original value
    copied_move_to = cast(MoveToCommand, copied_geo.commands[0])
    assert copied_move_to.end == (0.0, 0.0, 0.0)


def test_distance(sample_geometry):
    """Tests the distance calculation for a Geometry object."""
    # move(0,0) -> line(10,10) -> arc(20,0)
    # Approximating arc as a line for distance calc
    dist1 = math.hypot(10 - 0, 10 - 0)
    dist2 = math.hypot(20 - 10, 0 - 10)
    expected_dist = dist1 + dist2

    assert sample_geometry.distance() == pytest.approx(expected_dist)
    # Also test the query function directly
    assert get_total_distance(sample_geometry.commands) == pytest.approx(
        expected_dist
    )


def test_geo_command_distance():
    last_point = (0.0, 0.0, 0.0)
    line_cmd = LineToCommand((3.0, 4.0, 0.0))
    assert line_cmd.distance(last_point) == pytest.approx(5.0)

    move_cmd = MoveToCommand((-3.0, -4.0, 0.0))
    assert move_cmd.distance(last_point) == pytest.approx(5.0)

    # Arc distance is approximated as a straight line
    arc_cmd = ArcToCommand(
        end=(3.0, 4.0, 0.0), center_offset=(0, 0), clockwise=False
    )
    assert arc_cmd.distance(last_point) == pytest.approx(5.0)


def test_split_into_components_empty_geometry():
    geo = Geometry()
    components = geo.split_into_components()
    assert len(components) == 0


def test_split_into_components_single_contour():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    components = geo.split_into_components()
    assert len(components) == 1
    assert len(components[0].commands) == 3


def test_split_into_components_two_separate_shapes():
    geo = Geometry()
    # Shape 1
    geo.move_to(0, 0)
    geo.line_to(10, 0)
    geo.line_to(10, 10)
    geo.line_to(0, 10)
    geo.close_path()
    # Shape 2
    geo.move_to(20, 20)
    geo.line_to(30, 20)
    geo.line_to(30, 30)
    geo.line_to(20, 30)
    geo.close_path()

    components = geo.split_into_components()
    assert len(components) == 2
    assert len(components[0].commands) == 5
    assert len(components[1].commands) == 5


def test_split_into_components_containment_letter_o():
    geo = Geometry()
    # Outer circle (r=10, center=0,0)
    geo.move_to(10, 0)
    geo.arc_to(-10, 0, i=-10, j=0, clockwise=False)
    geo.arc_to(10, 0, i=10, j=0, clockwise=False)
    # Inner circle (r=5, center=0,0)
    geo.move_to(5, 0)
    geo.arc_to(-5, 0, i=-5, j=0, clockwise=False)
    geo.arc_to(5, 0, i=5, j=0, clockwise=False)

    components = geo.split_into_components()
    assert len(components) == 1
    assert len(components[0].commands) == 6  # 2 moves, 4 arcs


def test_from_points():
    """Tests the Geometry.from_points classmethod."""
    # Test case 1: Empty list
    geo_empty = Geometry.from_points([])
    assert geo_empty.is_empty()

    # Test case 2: Single point
    geo_single = Geometry.from_points([(10, 20)])
    assert len(geo_single.commands) == 1
    assert isinstance(geo_single.commands[0], MoveToCommand)
    assert geo_single.commands[0].end == (10, 20, 0)
    assert geo_single.last_move_to == (10, 20, 0)
    # A single point doesn't get closed
    assert not any(
        isinstance(cmd, LineToCommand) for cmd in geo_single.commands
    )

    # Test case 3: Triangle (closed by default)
    points = [(0, 0), (10, 0), (5, 10)]
    geo_triangle = Geometry.from_points(points)
    assert len(geo_triangle.commands) == 4
    assert isinstance(geo_triangle.commands[0], MoveToCommand)
    assert geo_triangle.commands[0].end == (0, 0, 0)
    assert isinstance(geo_triangle.commands[3], LineToCommand)
    assert geo_triangle.commands[3].end == (0, 0, 0)  # from close_path

    # Test case 4: Triangle (open)
    geo_triangle_open = Geometry.from_points(points, close=False)
    assert len(geo_triangle_open.commands) == 3
    assert isinstance(geo_triangle_open.commands[0], MoveToCommand)
    assert geo_triangle_open.commands[0].end == (0, 0, 0)
    assert isinstance(geo_triangle_open.commands[2], LineToCommand)
    assert geo_triangle_open.commands[2].end == (5, 10, 0)  # last point
    # Final check: end point is not the start point
    assert (
        geo_triangle_open.commands[-1].end
        != geo_triangle_open.commands[0].end
    )

    # Test case 5: Points with Z coordinates (closed)
    points_3d = [(0, 0, 1), (10, 0, 2), (5, 10, 3)]
    geo_3d = Geometry.from_points(points_3d)
    assert len(geo_3d.commands) == 4
    assert geo_3d.commands[0].end == (0, 0, 1)
    assert geo_3d.commands[1].end == (10, 0, 2)
    assert geo_3d.commands[2].end == (5, 10, 3)
    assert geo_3d.commands[3].end == (0, 0, 1)  # from close_path
    assert geo_3d.last_move_to == (0, 0, 1)

    # Test case 6: Points with Z coordinates (open)
    geo_3d_open = Geometry.from_points(points_3d, close=False)
    assert len(geo_3d_open.commands) == 3
    assert geo_3d_open.commands[0].end == (0, 0, 1)
    assert geo_3d_open.commands[1].end == (10, 0, 2)
    assert geo_3d_open.commands[2].end == (5, 10, 3)
    assert geo_3d_open.last_move_to == (0, 0, 1)
