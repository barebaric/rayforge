import pytest
import math
import numpy as np
from unittest.mock import patch, ANY

from rayforge.core.geo import (
    Geometry,
)
from rayforge.core.geo.query import get_total_distance_from_array

# Constants needed for mocking numpy data structure
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
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
    assert len(empty_geometry) == 0
    assert empty_geometry.last_move_to == (0.0, 0.0, 0.0)


def test_add_commands(empty_geometry):
    empty_geometry.move_to(5, 5)
    assert len(empty_geometry) == 1
    empty_geometry.line_to(10, 10)
    assert len(empty_geometry) == 2
    empty_geometry._sync_to_numpy()
    assert empty_geometry.data is not None
    assert empty_geometry.data[0, COL_TYPE] == CMD_TYPE_MOVE
    assert empty_geometry.data[1, COL_TYPE] == CMD_TYPE_LINE


def test_simplify_wrapper(sample_geometry):
    """Tests that simplify calls the underlying module."""
    # Ensure numpy data is populated before we patch
    sample_geometry._sync_to_numpy()
    original_data = sample_geometry.data

    with patch(
        "rayforge.core.geo.geometry.simplify_geometry_from_array"
    ) as mock_simplify:
        # Mock returns a simplified numpy array
        mock_return_data = np.array([[CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0, 0, 0]])
        mock_simplify.return_value = mock_return_data

        result = sample_geometry.simplify(tolerance=0.5)

        assert result is sample_geometry
        # Check that the geometry's internal data was replaced by the result
        np.testing.assert_array_equal(sample_geometry.data, mock_return_data)
        # Check that the function was called with the ORIGINAL data array
        assert original_data is not None
        mock_simplify.assert_called_once()
        np.testing.assert_array_equal(
            mock_simplify.call_args[0][0], original_data
        )
        assert mock_simplify.call_args[0][1] == 0.5


def test_clear_commands(sample_geometry):
    sample_geometry.clear()
    assert len(sample_geometry) == 0
    assert sample_geometry.is_empty()


def test_move_to(sample_geometry):
    sample_geometry.move_to(15, 15)
    sample_geometry._sync_to_numpy()
    assert sample_geometry.data is not None
    last_row = sample_geometry.data[-1]
    assert last_row[COL_TYPE] == CMD_TYPE_MOVE
    assert (last_row[COL_X : COL_Z + 1] == (15.0, 15.0, 0.0)).all()


def test_line_to(sample_geometry):
    sample_geometry.line_to(20, 20)
    sample_geometry._sync_to_numpy()
    assert sample_geometry.data is not None
    last_row = sample_geometry.data[-1]
    assert last_row[COL_TYPE] == CMD_TYPE_LINE
    assert (last_row[COL_X : COL_Z + 1] == (20.0, 20.0, 0.0)).all()


def test_close_path(sample_geometry):
    sample_geometry.move_to(5, 5, -1.0)
    sample_geometry.close_path()
    sample_geometry._sync_to_numpy()
    assert sample_geometry.data is not None
    last_row = sample_geometry.data[-1]
    assert last_row[COL_TYPE] == CMD_TYPE_LINE
    assert (last_row[COL_X : COL_Z + 1] == sample_geometry.last_move_to).all()
    assert (last_row[COL_X : COL_Z + 1] == (5.0, 5.0, -1.0)).all()


def test_arc_to(sample_geometry):
    sample_geometry.arc_to(5, 5, 2, 3, clockwise=False)
    sample_geometry._sync_to_numpy()
    assert sample_geometry.data is not None
    last_row = sample_geometry.data[-1]
    assert last_row[COL_TYPE] == CMD_TYPE_ARC
    assert (last_row[COL_X : COL_Z + 1] == (5.0, 5.0, 0.0)).all()
    assert last_row[-1] == 0.0  # Clockwise is False


def test_serialization_deserialization(sample_geometry):
    geo_dict = sample_geometry.dump()
    new_geo = Geometry.from_dict(geo_dict)
    assert new_geo == sample_geometry


def test_copy_method(sample_geometry):
    """Tests the deep copy functionality of the Geometry class."""
    original_geo = sample_geometry
    copied_geo = original_geo.copy()

    # Check for initial equality and deep copy semantics
    assert copied_geo is not original_geo
    assert copied_geo == original_geo
    assert copied_geo.last_move_to == original_geo.last_move_to

    # Modify the original and check that the copy is unaffected
    original_geo.line_to(100, 100)
    assert copied_geo != original_geo
    assert len(copied_geo) == 3
    assert len(original_geo) == 4


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
    assert get_total_distance_from_array(
        sample_geometry.data
    ) == pytest.approx(expected_dist)


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
    geo_with_hole.extend(hole)  # Use extend to merge numpy data
    # Expected area = 100 - (6*6) = 64
    assert geo_with_hole.area() == pytest.approx(64.0)

    # Test case 5: Two separate shapes
    geo_two_shapes = Geometry.from_points([(0, 0), (5, 0), (5, 5), (0, 5)])
    second_shape = Geometry.from_points(
        [(10, 10), (15, 10), (15, 15), (10, 15)]
    )
    geo_two_shapes.extend(second_shape)  # Use extend to merge numpy data
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
    assert len(geo_single) == 1
    geo_single._sync_to_numpy()
    assert geo_single.data is not None
    assert geo_single.data[0, COL_TYPE] == CMD_TYPE_MOVE
    assert (geo_single.data[0, 1:4] == (10, 20, 0)).all()
    assert geo_single.last_move_to == (10, 20, 0)

    # Test case 3: Triangle (closed by default)
    points = [(0, 0), (10, 0), (5, 10)]
    geo_triangle = Geometry.from_points(points)
    assert len(geo_triangle) == 4
    geo_triangle._sync_to_numpy()
    assert geo_triangle.data is not None
    assert geo_triangle.data[0, COL_TYPE] == CMD_TYPE_MOVE
    assert (geo_triangle.data[0, 1:4] == (0, 0, 0)).all()
    assert geo_triangle.data[3, COL_TYPE] == CMD_TYPE_LINE
    assert (geo_triangle.data[3, 1:4] == (0, 0, 0)).all()

    # Test case 4: Triangle (open)
    geo_triangle_open = Geometry.from_points(points, close=False)
    assert len(geo_triangle_open) == 3
    geo_triangle_open._sync_to_numpy()
    assert geo_triangle_open.data is not None
    assert (
        geo_triangle_open.data[-1, 1:4] != geo_triangle_open.data[0, 1:4]
    ).any()

    # Test case 5: Points with Z coordinates (closed)
    points_3d = [(0, 0, 1), (10, 0, 2), (5, 10, 3)]
    geo_3d = Geometry.from_points(points_3d)
    assert len(geo_3d) == 4
    geo_3d._sync_to_numpy()
    assert geo_3d.data is not None
    assert (geo_3d.data[0, 1:4] == (0, 0, 1)).all()
    assert (geo_3d.data[3, 1:4] == (0, 0, 1)).all()
    assert geo_3d.last_move_to == (0, 0, 1)


def test_dump_and_load(sample_geometry):
    """
    Tests the dump() and load() methods for space-efficient serialization.
    """
    # Test with a non-empty geometry
    dumped_data = sample_geometry.dump()
    loaded_geo = Geometry.load(dumped_data)

    assert dumped_data["last_move_to"] == list(sample_geometry.last_move_to)
    assert len(dumped_data["commands"]) == 3
    assert dumped_data["commands"][0] == ["M", 0.0, 0.0, 0.0]
    assert dumped_data["commands"][1] == ["L", 10.0, 10.0, 0.0]
    assert dumped_data["commands"][2] == ["A", 20.0, 0.0, 0.0, 5.0, -10.0, 1]

    assert loaded_geo == sample_geometry

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
    mock_return_geo = Geometry()
    mock_return_geo.line_to(1, 1)
    mock_close_gaps.return_value = mock_return_geo

    result = sample_geometry.close_gaps(tolerance=1e-5)
    assert result is sample_geometry

    # The function is called with a COPY of the original geometry.
    # We use ANY to check that it was called with a Geometry object.
    mock_close_gaps.assert_called_once_with(ANY, tolerance=1e-5)

    # Check that the geometry's data was updated from the result
    assert sample_geometry == mock_return_geo


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
    assert sample_geometry.data is not None
    mock_is_closed.assert_called_once()
    # Check that the call was made with the numpy array
    np.testing.assert_array_equal(
        mock_is_closed.call_args[0][0], sample_geometry.data
    )
    assert mock_is_closed.call_args[1]["tolerance"] == 1e-5


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


@patch("rayforge.core.geo.intersect.check_self_intersection_from_array")
def test_has_self_intersections_wrapper(mock_check, sample_geometry):
    """Tests the Geometry.has_self_intersections() wrapper method."""
    sample_geometry.has_self_intersections(fail_on_t_junction=True)
    assert sample_geometry.data is not None
    mock_check.assert_called_once()
    np.testing.assert_array_equal(
        mock_check.call_args[0][0], sample_geometry.data
    )
    assert mock_check.call_args[1]["fail_on_t_junction"] is True


@patch("rayforge.core.geo.intersect.check_intersection_from_array")
def test_intersects_with_wrapper(mock_check, sample_geometry):
    """Tests the Geometry.intersects_with() wrapper method."""
    other_geo = Geometry()
    other_geo.line_to(1, 1)  # Make other_geo non-empty
    sample_geometry.intersects_with(other_geo)
    assert sample_geometry.data is not None
    assert other_geo.data is not None
    mock_check.assert_called_once()
    np.testing.assert_array_equal(
        mock_check.call_args[0][0], sample_geometry.data
    )
    np.testing.assert_array_equal(mock_check.call_args[0][1], other_geo.data)


@patch("rayforge.core.geo.transform.grow_geometry")
def test_grow_wrapper(mock_grow, sample_geometry):
    """Tests the Geometry.grow() wrapper method."""
    sample_geometry.grow(amount=5.0)
    mock_grow.assert_called_once_with(sample_geometry, offset=5.0)


@patch("rayforge.core.geo.transform.apply_affine_transform_to_array")
def test_transform_wrapper(mock_transform_array, sample_geometry):
    """
    Tests that Geometry.transform() delegates correctly to
    apply_affine_transform_to_array when shadow data is available.
    """
    matrix = np.identity(4)

    # Ensure sample_geometry has its internal numpy buffer populated
    sample_geometry._sync_to_numpy()
    original_data = sample_geometry.data

    # Mock the return: a single MoveTo command at (99, 99, 99)
    mock_data = np.zeros((1, 7))
    mock_data[0, COL_TYPE] = CMD_TYPE_MOVE
    mock_data[0, COL_X] = 99.0
    mock_data[0, COL_Y] = 99.0
    mock_data[0, COL_Z] = 99.0

    mock_transform_array.return_value = mock_data

    sample_geometry.transform(matrix)

    # Verify delegation
    assert mock_transform_array.called
    args, _ = mock_transform_array.call_args
    # First arg is the numpy array, second is the matrix
    assert original_data is not None
    np.testing.assert_array_equal(args[0], original_data)
    assert args[1] is matrix

    # Verify that the geometry's data has been updated
    np.testing.assert_array_equal(sample_geometry.data, mock_data)


def test_to_cairo():
    """Tests the to_cairo() method for drawing geometry to a Cairo context."""
    from unittest.mock import Mock
    import cairo

    geo = Geometry()
    geo.move_to(5, 5)
    geo.line_to(10, 10)
    geo.arc_to(15, 5, i=0, j=-5, clockwise=False)

    ctx = Mock(spec=cairo.Context)
    geo.to_cairo(ctx)

    ctx.move_to.assert_called_once_with(5, 5)
    ctx.line_to.assert_called_once_with(10, 10)
    ctx.arc.assert_called_once()


def test_to_cairo_empty_geometry():
    """Tests that to_cairo() handles empty geometry gracefully."""
    from unittest.mock import Mock
    import cairo

    geo = Geometry()
    ctx = Mock(spec=cairo.Context)
    geo.to_cairo(ctx)

    ctx.move_to.assert_not_called()
    ctx.line_to.assert_not_called()
    ctx.arc.assert_not_called()
    ctx.arc_negative.assert_not_called()


def test_to_cairo_clockwise_arc():
    """Tests that to_cairo() correctly draws clockwise arcs."""
    from unittest.mock import Mock
    import cairo

    geo = Geometry()
    geo.move_to(10, 10)
    geo.arc_to(15, 10, i=0, j=-5, clockwise=True)

    ctx = Mock(spec=cairo.Context)
    geo.to_cairo(ctx)

    ctx.move_to.assert_called_once_with(10, 10)
    ctx.arc_negative.assert_called_once()
    ctx.arc.assert_not_called()


def test_get_command_at_valid_index():
    """Tests get_command_at() with valid indices."""
    geo = Geometry()
    geo.move_to(0, 0, 1)
    geo.line_to(10, 10, 2)
    geo.arc_to(20, 0, i=5, j=-10, clockwise=False, z=3)

    cmd0 = geo.get_command_at(0)
    assert cmd0 == (CMD_TYPE_MOVE, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)

    cmd1 = geo.get_command_at(1)
    assert cmd1 == (CMD_TYPE_LINE, 10.0, 10.0, 2.0, 0.0, 0.0, 0.0)

    cmd2 = geo.get_command_at(2)
    assert cmd2 == (CMD_TYPE_ARC, 20.0, 0.0, 3.0, 5.0, -10.0, 0.0)


def test_get_command_at_negative_index():
    """Tests get_command_at() with negative index."""
    geo = Geometry()
    geo.move_to(0, 0)
    assert geo.get_command_at(-1) is None


def test_get_command_at_out_of_bounds():
    """Tests get_command_at() with index out of bounds."""
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    assert geo.get_command_at(2) is None
    assert geo.get_command_at(100) is None


def test_get_command_at_empty_geometry():
    """Tests get_command_at() on empty geometry."""
    geo = Geometry()
    assert geo.get_command_at(0) is None


def test_iter_commands():
    """Tests iter_commands() yields all commands correctly."""
    geo = Geometry()
    geo.move_to(0, 0, 1)
    geo.line_to(10, 10, 2)
    geo.arc_to(20, 0, i=5, j=-10, clockwise=False, z=3)

    commands = list(geo.iter_commands())

    assert len(commands) == 3
    assert commands[0] == (CMD_TYPE_MOVE, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert commands[1] == (CMD_TYPE_LINE, 10.0, 10.0, 2.0, 0.0, 0.0, 0.0)
    assert commands[2] == (CMD_TYPE_ARC, 20.0, 0.0, 3.0, 5.0, -10.0, 0.0)


def test_iter_commands_empty_geometry():
    """Tests iter_commands() on empty geometry."""
    geo = Geometry()
    commands = list(geo.iter_commands())
    assert commands == []


def test_iter_commands_with_pending_data():
    """Tests iter_commands() syncs pending data before iteration."""
    geo = Geometry()
    geo.move_to(5, 5, 1)
    geo.line_to(15, 15, 2)

    commands = list(geo.iter_commands())

    assert len(commands) == 2
    assert commands[0] == (CMD_TYPE_MOVE, 5.0, 5.0, 1.0, 0.0, 0.0, 0.0)
    assert commands[1] == (CMD_TYPE_LINE, 15.0, 15.0, 2.0, 0.0, 0.0, 0.0)


def test_iter_commands_clockwise_arc():
    """Tests iter_commands() with clockwise arc."""
    geo = Geometry()
    geo.move_to(10, 10)
    geo.arc_to(15, 10, i=0, j=-5, clockwise=True)

    commands = list(geo.iter_commands())

    assert len(commands) == 2
    assert commands[0] == (CMD_TYPE_MOVE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0)
    assert commands[1] == (CMD_TYPE_ARC, 15.0, 10.0, 0.0, 0.0, -5.0, 1.0)
