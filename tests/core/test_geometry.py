from copy import deepcopy
import pytest
import math
import numpy as np
from typing import cast
from rayforge.core.geometry import (
    Geometry,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
)


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


def test_move_to_3d(sample_geometry):
    sample_geometry.move_to(15, 15, -5.0)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, MoveToCommand)
    assert last_cmd.end == (15.0, 15.0, -5.0)


def test_line_to(sample_geometry):
    sample_geometry.line_to(20, 20)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, 0.0)


def test_line_to_3d(sample_geometry):
    sample_geometry.line_to(20, 20, -2.5)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, LineToCommand)
    assert last_cmd.end == (20.0, 20.0, -2.5)


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


def test_arc_to_3d(sample_geometry):
    sample_geometry.arc_to(5, 5, 2, 3, clockwise=False, z=-10.0)
    last_cmd = sample_geometry.commands[-1]
    assert isinstance(last_cmd, ArcToCommand)
    assert last_cmd.end == (5.0, 5.0, -10.0)


def test_rect_calculation(sample_geometry):
    # Points are (0,0), (10,10), and (20,0) from the arc.
    # Bbox should be (0,0) to (20,10)
    min_x, min_y, max_x, max_y = sample_geometry.rect()
    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(20.0)
    assert max_y == pytest.approx(10.0)


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


def test_linearize_arc_method(sample_geometry):
    """Tests the internal _linearize_arc method."""
    # The second command is a line_to(10,10), which is the start of the arc
    start_point = cast(LineToCommand, sample_geometry.commands[1]).end
    # The third command is the arc
    arc_cmd = cast(ArcToCommand, sample_geometry.commands[2])

    segments = sample_geometry._linearize_arc(arc_cmd, start_point)

    # Check that linearization produces a reasonable number of segments
    assert len(segments) >= 2

    # Check that the start and end points of the chain of segments match
    # the original arc's start and end points.
    first_segment_start, _ = segments[0]
    _, last_segment_end = segments[-1]

    assert first_segment_start == pytest.approx(start_point)
    assert last_segment_end == pytest.approx(arc_cmd.end)


def test_find_closest_point_empty_geometry():
    geo = Geometry()
    assert geo.find_closest_point(10, 10) is None


def test_find_closest_point_single_line():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)

    # Point on the line
    result = geo.find_closest_point(5, 0)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 0))

    # Point directly above the line
    result = geo.find_closest_point(5, 5)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 0))

    # Point past the end
    result = geo.find_closest_point(15, 0)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(1.0)
    assert point == pytest.approx((10, 0))

    # Point before the start
    result = geo.find_closest_point(-5, 5)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.0)
    assert point == pytest.approx((0, 0))


def test_find_closest_point_diagonal_line():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)

    # Closest point is the midpoint of the segment
    result = geo.find_closest_point(0, 10)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 5))


def test_find_closest_point_square():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(10, 0)  # cmd 1
    geo.line_to(10, 10)  # cmd 2
    geo.line_to(0, 10)  # cmd 3
    geo.close_path()  # cmd 4 (line to 0,0 from 0,10)

    # Closest to bottom edge
    result = geo.find_closest_point(5, -2)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 0))

    # Closest to top-right corner (end of segment 2)
    result = geo.find_closest_point(11, 11)
    assert result is not None
    idx, t, point = result
    assert idx == 2
    assert t == pytest.approx(1.0)
    assert point == pytest.approx((10, 10))

    # Closest to left edge (part of segment 4)
    result = geo.find_closest_point(-3, 5)
    assert result is not None
    idx, t, point = result
    assert idx == 4
    # Segment 4 is from (0, 10) to (0, 0). (0, 5) is halfway.
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((0, 5))


def test_find_closest_point_arc():
    geo = Geometry()
    geo.move_to(10, 0)
    # 90 deg counter-clockwise arc, center (0,0), radius 10
    geo.arc_to(0, 10, i=-10, j=0, clockwise=False)

    # Point at 45 degrees on the arc
    p_on_arc_x = 10 * math.cos(math.radians(45))
    p_on_arc_y = 10 * math.sin(math.radians(45))
    result = geo.find_closest_point(p_on_arc_x, p_on_arc_y)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    # Tolerance needed due to arc linearization
    assert t == pytest.approx(0.5, abs=1e-2)
    assert point == pytest.approx((p_on_arc_x, p_on_arc_y), abs=1e-2)

    # Point "inside" the arc, should snap to 45 degree point
    result = geo.find_closest_point(5, 5)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5, abs=1e-2)
    assert point == pytest.approx((p_on_arc_x, p_on_arc_y), abs=1e-2)

    # Point past the end of the arc
    result = geo.find_closest_point(-5, 15)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(1.0)
    assert point == pytest.approx((0, 10))


def test_find_closest_point_with_gap():
    geo = Geometry()
    geo.move_to(0, 0)  # cmd 0
    geo.line_to(10, 0)  # cmd 1
    geo.move_to(0, 10)  # cmd 2
    geo.line_to(10, 10)  # cmd 3

    # Point closer to bottom line
    result = geo.find_closest_point(5, 4)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 0))

    # Point closer to top line
    result = geo.find_closest_point(5, 6)
    assert result is not None
    idx, t, point = result
    assert idx == 3
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 10))
