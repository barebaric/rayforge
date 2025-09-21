import pytest
import math
from rayforge.core.geo import Geometry
from rayforge.core.ops import Ops
from rayforge.core.ops.commands import (
    MoveToCommand,
    LineToCommand,
    ScanLinePowerCommand,
)
from rayforge.core.geo.query import (
    get_bounding_rect,
    get_total_distance,
    find_closest_point_on_path,
)


@pytest.fixture
def sample_geometry():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.arc_to(20, 0, i=5, j=-10)
    return geo


def test_get_bounding_rect(sample_geometry):
    # Points are (0,0), (10,10), and (20,0) from the arc.
    # Bbox should be (0,0) to (20,10)
    min_x, min_y, max_x, max_y = get_bounding_rect(sample_geometry.commands)
    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(20.0)
    assert max_y == pytest.approx(10.0)


def test_get_bounding_rect_with_ops_commands():
    """Tests bounding box calculation with a list of ops commands."""
    ops_list = [
        MoveToCommand((0, 0, 0)),
        LineToCommand((10, 0, 0)),
        LineToCommand((10, 10, 0)),
    ]
    min_x, min_y, max_x, max_y = get_bounding_rect(ops_list)
    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(10.0)
    assert max_y == pytest.approx(10.0)


def test_get_bounding_rect_ignores_travel():
    """Tests that travel-only moves do not affect the bounding box."""
    ops_list = [
        MoveToCommand((0, 0, 0)),
        LineToCommand((10, 10, 0)),
        MoveToCommand((100, 100, 0)),  # Should be ignored
    ]
    min_x, min_y, max_x, max_y = get_bounding_rect(ops_list)
    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(10.0)
    assert max_y == pytest.approx(10.0)


def test_get_total_distance_with_geo_commands(sample_geometry):
    # Geometry fixture (all moves are "drawing")
    dist_geo = get_total_distance(sample_geometry.commands)
    expected_geo = math.hypot(10, 10) + math.hypot(10, -10)
    assert dist_geo == pytest.approx(expected_geo)


def test_get_total_distance_with_ops_commands():
    """
    Tests distance calculation with a list of ops commands,
    including scanline.
    """
    # Ops fixture with travel and cutting moves
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(3, 4)  # Cutting move, length 5
    ops.move_to(10, 10)  # Travel move from (3,4), length sqrt(7^2+6^2)
    ops.add(
        ScanLinePowerCommand(
            end=(20, 10, 0),
            power_values=bytearray(),
        )
    )  # dist 10

    dist = get_total_distance(ops.commands)
    expected = (
        math.hypot(0, 0)
        + 5.0
        + math.hypot(10 - 3, 10 - 4)
        + math.hypot(20 - 10, 10 - 10)
    )
    assert dist == pytest.approx(expected)


def test_find_closest_point_on_path_empty_geometry():
    geo = Geometry()
    assert find_closest_point_on_path(geo.commands, 10, 10) is None


def test_find_closest_point_on_path_single_line():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)

    # Point on the line
    result = find_closest_point_on_path(geo.commands, 5, 0)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 0))

    # Point directly above the line
    result = find_closest_point_on_path(geo.commands, 5, 5)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5)
    assert point == pytest.approx((5, 0))


def test_find_closest_point_on_path_arc():
    geo = Geometry()
    geo.move_to(10, 0)
    # 90 deg counter-clockwise arc, center (0,0), radius 10
    geo.arc_to(0, 10, i=-10, j=0, clockwise=False)

    # Point at 45 degrees on the arc
    p_on_arc_x = 10 * math.cos(math.radians(45))
    p_on_arc_y = 10 * math.sin(math.radians(45))
    result = find_closest_point_on_path(geo.commands, p_on_arc_x, p_on_arc_y)
    assert result is not None
    idx, t, point = result
    assert idx == 1
    assert t == pytest.approx(0.5, abs=1e-2)
    assert point == pytest.approx((p_on_arc_x, p_on_arc_y), abs=1e-2)
