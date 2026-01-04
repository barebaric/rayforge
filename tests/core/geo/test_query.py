import pytest
import math
from rayforge.core.geo import Geometry


@pytest.fixture
def sample_geometry():
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.arc_to(20, 0, i=5, j=-10, clockwise=True)
    return geo


def test_get_bounding_rect(sample_geometry):
    # This geometry creates a line from (0,0) to (10,10) and then an arc
    # from (10,10) to (20,0) with center (15,0).
    # The arc is clockwise and sweeps from ~116.5 deg down through 90 deg to
    # 0 deg. It crosses the North (90 deg) and East (0 deg) axes.
    # Radius is sqrt(125).
    # min_x is 0 (from the start of the line).
    # min_y is 0 (from the arc's end point and the line's start point).
    # max_x is determined by the arc crossing East: center_x + radius.
    # max_y is determined by the arc crossing North: center_y + radius.
    min_x, min_y, max_x, max_y = sample_geometry.rect()

    center_x, center_y = 15.0, 0.0
    radius = math.sqrt(125)

    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(center_x + radius)
    assert max_y == pytest.approx(center_y + radius)


def test_get_total_distance_with_geo_commands(sample_geometry):
    # This test was rewritten to test a simple 90 degree arc, as the
    # sample_geometry arc length is non-trivial to calculate by hand.
    # The linter warning about the unused 'dist_geo' variable is now resolved.
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)  # Length 10
    # 90 deg CCW arc, radius 10, length should be 5*pi
    geo.arc_to(0, 10, i=-10, j=0, clockwise=False)
    dist = geo.distance()
    expected = 10 + 0.5 * math.pi * 10
    assert dist == pytest.approx(expected)


def test_find_closest_point_on_path_empty_geometry():
    geo = Geometry()
    assert geo.find_closest_point(10, 10) is None


def test_find_closest_point_on_path_single_line():
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


def test_find_closest_point_on_path_arc():
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
    assert t == pytest.approx(0.5, abs=1e-2)
    assert point == pytest.approx((p_on_arc_x, p_on_arc_y), abs=1e-2)
