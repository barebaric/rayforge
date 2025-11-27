import pytest
import math
import numpy as np
from typing import cast
from unittest.mock import patch

from rayforge.core.geo import (
    Geometry,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
)
from rayforge.core.geo.query import get_total_distance


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
    """
    Tests that the distance calculation correctly computes true arc length.
    """
    # The test must now expect the true distance, not an approximation.
    dist_line = math.hypot(10 - 0, 10 - 0)

    # Arc parameters for manual length calculation:
    # Center is (10,10) + (5,-10) = (15,0). Radius = dist from center to start.
    radius = math.hypot(10 - 15, 10 - 0)  # sqrt((-5)^2 + 10^2) = sqrt(125)
    start_angle = math.atan2(10 - 0, 10 - 15)  # atan2(10, -5)
    end_angle = math.atan2(0 - 0, 20 - 15)  # atan2(0, 5) -> 0

    # The default for arc_to is clockwise=True.
    # For a clockwise arc from a larger angle (start_angle in Q2) to a smaller
    # one (end_angle on the axis), the span is just the difference.
    angle_span = start_angle - end_angle
    dist_arc = radius * angle_span

    expected_dist = dist_line + dist_arc

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

    # We'll test a 90-degree arc from (10,0) to (0,10) centered at the origin.
    start_point_for_arc = (10.0, 0.0, 0.0)
    arc_cmd = ArcToCommand(
        end=(0.0, 10.0, 0.0), center_offset=(-10.0, 0.0), clockwise=False
    )
    # Radius is 10, angle span is PI/2.
    expected_arc_length = 0.5 * math.pi * 10
    assert arc_cmd.distance(start_point_for_arc) == pytest.approx(
        expected_arc_length
    )


def test_area():
    # Test case 1: Empty and open paths
    assert Geometry().area() == 0.0
    assert Geometry.from_points([(0, 0), (10, 10)], close=False).area() == 0.0

    # Test case 2: Simple 10x10 CCW square
    square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    assert square.area() == pytest.approx(100.0)

    # Test case 3: Simple 10x10 CW square (should have same positive area)
    square_cw = Geometry.from_points([(0, 0), (0, 10), (10, 10), (10, 0)])
    assert square_cw.area() == pytest.approx(100.0)

    # Test case 4: Shape with a hole
    # Outer CCW square (0,0) -> (10,10)
    geo_with_hole = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])
    # Inner CW square (hole) (2,2) -> (8,8)
    hole = Geometry.from_points([(2, 2), (2, 8), (8, 8), (8, 2)])
    geo_with_hole.commands.extend(hole.commands)
    # Expected area = 100 - (6*6) = 64
    assert geo_with_hole.area() == pytest.approx(64.0)

    # Test case 5: Two separate shapes
    geo_two_shapes = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    second_shape = Geometry.from_points(
        [(10, 10), (15, 10), (15, 15), (10, 15)]
    )
    geo_two_shapes.commands.extend(second_shape.commands)
    # Expected area = 25 + 25 = 50
    assert geo_two_shapes.area() == pytest.approx(50.0)


def test_segments():
    """Tests the segments() method for extracting point lists."""
    # Test case 1: Empty geometry
    geo_empty = Geometry()
    assert geo_empty.segments() == []

    # Test case 2: Single open path
    geo_open = Geometry()
    geo_open.move_to(0, 0, 1)
    geo_open.line_to(10, 0, 2)
    geo_open.arc_to(10, 10, i=0, j=5, z=3)
    expected_open = [[(0, 0, 1), (10, 0, 2), (10, 10, 3)]]
    assert geo_open.segments() == expected_open

    # Test case 3: Single closed path
    geo_closed = Geometry.from_points([(0, 0), (10, 0), (0, 10)])
    expected_closed = [[(0, 0, 0), (10, 0, 0), (0, 10, 0), (0, 0, 0)]]
    assert geo_closed.segments() == expected_closed

    # Test case 4: Multiple disjoint segments
    geo_multi = Geometry()
    # Segment 1
    geo_multi.move_to(0, 0)
    geo_multi.line_to(1, 1)
    # Segment 2
    geo_multi.move_to(10, 10)
    geo_multi.line_to(11, 11)
    geo_multi.line_to(12, 12)
    expected_multi = [
        [(0, 0, 0), (1, 1, 0)],
        [(10, 10, 0), (11, 11, 0), (12, 12, 0)],
    ]
    assert geo_multi.segments() == expected_multi

    # Test case 5: Path starting with a LineTo (implicit start at 0,0,0)
    geo_implicit_start = Geometry()
    geo_implicit_start.line_to(5, 5)
    geo_implicit_start.line_to(10, 0)
    expected_implicit = [[(0, 0, 0), (5, 5, 0), (10, 0, 0)]]
    assert geo_implicit_start.segments() == expected_implicit


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
        geo_triangle_open.commands[-1].end != geo_triangle_open.commands[0].end
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


def test_dump_and_load(sample_geometry):
    """
    Tests the dump() and load() methods for space-efficient serialization.
    """
    # Test with a non-empty geometry
    dumped_data = sample_geometry.dump()
    loaded_geo = Geometry.load(dumped_data)

    assert dumped_data["last_move_to"] == list(sample_geometry.last_move_to)
    assert len(dumped_data["commands"]) == 3
    # M 0 0 0
    assert dumped_data["commands"][0] == ["M", 0.0, 0.0, 0.0]
    # L 10 10 0
    assert dumped_data["commands"][1] == ["L", 10.0, 10.0, 0.0]
    # A 20 0 0 5 -10 1 (default clockwise is True)
    assert dumped_data["commands"][2] == ["A", 20.0, 0.0, 0.0, 5.0, -10.0, 1]

    assert loaded_geo.last_move_to == sample_geometry.last_move_to
    assert len(loaded_geo.commands) == len(sample_geometry.commands)
    for original_cmd, loaded_cmd in zip(
        sample_geometry.commands, loaded_geo.commands
    ):
        assert type(original_cmd) is type(loaded_cmd)
        # Easy way to check all attributes are the same
        assert original_cmd.to_dict() == loaded_cmd.to_dict()

    # Test with an empty geometry
    empty_geo = Geometry()
    dumped_empty = empty_geo.dump()
    loaded_empty = Geometry.load(dumped_empty)

    assert dumped_empty["last_move_to"] == [0.0, 0.0, 0.0]
    assert dumped_empty["commands"] == []
    assert loaded_empty.is_empty()
    assert loaded_empty.last_move_to == (0.0, 0.0, 0.0)


# --- Wrapper Method Tests ---
# These tests verify that the Geometry methods correctly wrap and call the
# underlying stateless functions from other modules.


@patch("rayforge.core.geo.contours.close_geometry_gaps")
def test_close_gaps_wrapper(mock_close_gaps, sample_geometry):
    """Tests the Geometry.close_gaps() wrapper method."""
    # The wrapper modifies the object in-place, so we need to simulate the
    # return value of the underlying function.
    mock_return_geo = Geometry()
    mock_return_geo.line_to(1, 1)  # Give it some unique content
    mock_close_gaps.return_value = mock_return_geo

    # The method should return `self`
    result = sample_geometry.close_gaps(tolerance=1e-5)
    assert result is sample_geometry

    # Check that the underlying function was called correctly
    mock_close_gaps.assert_called_once_with(sample_geometry, tolerance=1e-5)

    # Check that the geometry's commands were updated from the result
    assert sample_geometry.commands == mock_return_geo.commands


@patch("rayforge.core.geo.contours.split_inner_and_outer_contours")
@patch("rayforge.core.geo.split.split_into_contours")
def test_split_inner_and_outer_contours_wrapper(
    mock_split_contours, mock_split_inner_outer, sample_geometry
):
    """
    Tests the Geometry.split_into_inner_and_outer_contours() wrapper method.
    """
    # 1. Setup mock return values
    mock_contour_list = [Geometry(), Geometry()]
    mock_split_contours.return_value = mock_contour_list

    mock_result_tuple = ([mock_contour_list[0]], [mock_contour_list[1]])
    mock_split_inner_outer.return_value = mock_result_tuple

    # 2. Call the wrapper method
    result = sample_geometry.split_inner_and_outer_contours()

    # 3. Assertions
    # Check that the first function was called correctly
    mock_split_contours.assert_called_once_with(sample_geometry)

    # Check that the second function was called with the result of the first
    mock_split_inner_outer.assert_called_once_with(mock_contour_list)

    # Check that the final result is what the second function returned
    assert result is mock_result_tuple


@patch("rayforge.core.geo.geometry.is_closed")
def test_is_closed_wrapper(mock_is_closed, sample_geometry):
    """Tests the Geometry.is_closed() wrapper method."""
    mock_is_closed.return_value = True
    result = sample_geometry.is_closed(tolerance=1e-5)
    assert result is True
    mock_is_closed.assert_called_once_with(
        sample_geometry.commands, tolerance=1e-5
    )


@patch("rayforge.core.geo.contours.remove_inner_edges")
def test_remove_inner_edges_wrapper(mock_remove, sample_geometry):
    """Tests the Geometry.remove_inner_edges() wrapper method."""
    mock_remove.return_value = "Success"
    result = sample_geometry.remove_inner_edges()
    assert result == "Success"
    mock_remove.assert_called_once_with(sample_geometry)


@patch("rayforge.core.geo.split.split_into_contours")
def test_split_into_contours_wrapper(mock_split, sample_geometry):
    """Tests the Geometry.split_into_contours() wrapper method."""
    sample_geometry.split_into_contours()
    mock_split.assert_called_once_with(sample_geometry)


@patch("rayforge.core.geo.split.split_into_components")
def test_split_into_components_wrapper(mock_split, sample_geometry):
    """Tests the Geometry.split_into_components() wrapper method."""
    sample_geometry.split_into_components()
    mock_split.assert_called_once_with(sample_geometry)


@patch("rayforge.core.geo.analysis.encloses")
def test_encloses_wrapper(mock_encloses, sample_geometry):
    """Tests the Geometry.encloses() wrapper method."""
    other_geo = Geometry()
    sample_geometry.encloses(other_geo)
    mock_encloses.assert_called_once_with(sample_geometry, other_geo)


@patch("rayforge.core.geo.intersect.check_self_intersection")
def test_has_self_intersections_wrapper(mock_check, sample_geometry):
    """Tests the Geometry.has_self_intersections() wrapper method."""
    sample_geometry.has_self_intersections(fail_on_t_junction=True)
    mock_check.assert_called_once_with(
        sample_geometry.commands, fail_on_t_junction=True
    )


@patch("rayforge.core.geo.intersect.check_intersection")
def test_intersects_with_wrapper(mock_check, sample_geometry):
    """Tests the Geometry.intersects_with() wrapper method."""
    other_geo = Geometry()
    sample_geometry.intersects_with(other_geo)
    mock_check.assert_called_once_with(
        sample_geometry.commands, other_geo.commands
    )


@patch("rayforge.core.geo.transform.grow_geometry")
def test_grow_wrapper(mock_grow, sample_geometry):
    """Tests the Geometry.grow() wrapper method."""
    sample_geometry.grow(amount=5.0)
    mock_grow.assert_called_once_with(sample_geometry, offset=5.0)


@patch("rayforge.core.geo.transform.apply_affine_transform")
def test_transform_wrapper(mock_transform, sample_geometry):
    """
    Tests that Geometry.transform() delegates correctly to
    apply_affine_transform.
    """
    matrix = np.identity(4)
    # Mock the return to ensure the wrapper updates self.commands
    mock_cmds = [LineToCommand((99, 99, 99))]
    mock_transform.return_value = mock_cmds

    # We must capture the original commands list before calling transform,
    # because transform() updates self.commands to point to the result
    # (mock_cmds).
    # The mock records the argument passed (original commands), but
    # self.commands will be different when we assert.
    original_commands = sample_geometry.commands

    sample_geometry.transform(matrix)

    mock_transform.assert_called_once_with(original_commands, matrix)
    assert sample_geometry.commands == mock_cmds
