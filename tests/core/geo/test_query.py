import pytest
import math
from rayforge.core.geo import Geometry
from rayforge.core.geo.query import bboxes_intersect


@pytest.fixture
def sample_geometry():
    """
    Creates a geometry with a line and a circular arc.
    The arc is designed to bulge outside the bounding box of its endpoints.

    Path:
    1. Move to (0, 0)
    2. Line to (10, 10)
    3. Arc to (20, 10).
       Start of arc: (10, 10).
       End of arc: (20, 10).
       Center: (15, 10). (Midpoint is (15,10), radius 5).
       Center Offset from Start (10,10) is (5, 0).

       Vector Center->Start: (-5, 0) -> 180 degrees.
       Vector Center->End:   (5, 0)  -> 0 degrees.

       Clockwise (CW): 180 -> 90 -> 0.
       This arc bulges upwards (North).
       Apex at (15, 15).
    """
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    # Center (15, 10). Start (10, 10). Offset = (5, 0).
    geo.arc_to(20, 10, i=5, j=0, clockwise=True)
    return geo


def test_get_bounding_rect(sample_geometry):
    min_x, min_y, max_x, max_y = sample_geometry.rect()

    # Min X: 0 (start of line)
    # Max X: 20 (end of arc)
    # Min Y: 0 (start of line)
    # Max Y: 15 (Apex of the arc at 90 degrees relative to center)

    assert min_x == pytest.approx(0.0)
    assert min_y == pytest.approx(0.0)
    assert max_x == pytest.approx(20.0)
    assert max_y == pytest.approx(15.0)


def test_get_bounding_rect_with_beziers():
    """
    Tests that the bounding box is calculated correctly even when the
    geometry is forced to use Beziers (approximating the arc).
    """
    # Same geometry logic as sample_geometry, but with force_beziers=True
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 10)
    geo.arc_to(20, 10, i=5, j=0, clockwise=True)

    min_x, min_y, max_x, max_y = geo.rect()

    # The bezier approximation should be very close to the true bounding box.
    # Allow a small tolerance for the approximation error.
    assert min_x == pytest.approx(0.0, abs=1e-3)
    assert min_y == pytest.approx(0.0, abs=1e-3)
    assert max_x == pytest.approx(20.0, abs=1e-3)
    assert max_y == pytest.approx(15.0, abs=1e-3)


def test_get_total_distance_with_geo_commands():
    # 90 deg CCW arc, radius 10, length should be 5*pi
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(10, 0)  # Length 10
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


class TestBboxesIntersect:
    def test_overlapping(self):
        bbox1 = (0, 0, 10, 10)
        bbox2 = (5, 5, 15, 15)
        assert bboxes_intersect(bbox1, bbox2) is True

    def test_non_overlapping(self):
        bbox1 = (0, 0, 10, 10)
        bbox2 = (20, 20, 30, 30)
        assert bboxes_intersect(bbox1, bbox2) is False

    def test_touching_edge(self):
        bbox1 = (0, 0, 10, 10)
        bbox2 = (10, 0, 20, 10)
        assert bboxes_intersect(bbox1, bbox2) is True

    def test_touching_corner(self):
        bbox1 = (0, 0, 10, 10)
        bbox2 = (10, 10, 20, 20)
        assert bboxes_intersect(bbox1, bbox2) is True

    def test_one_inside_other(self):
        bbox1 = (0, 0, 20, 20)
        bbox2 = (5, 5, 15, 15)
        assert bboxes_intersect(bbox1, bbox2) is True
        assert bboxes_intersect(bbox2, bbox1) is True

    def test_identical(self):
        bbox = (0, 0, 10, 10)
        assert bboxes_intersect(bbox, bbox) is True

    def test_negative_coordinates(self):
        bbox1 = (-10, -10, 0, 0)
        bbox2 = (-5, -5, 5, 5)
        assert bboxes_intersect(bbox1, bbox2) is True

    def test_separated_horizontally(self):
        bbox1 = (0, 0, 10, 10)
        bbox2 = (15, 0, 25, 10)
        assert bboxes_intersect(bbox1, bbox2) is False

    def test_separated_vertically(self):
        bbox1 = (0, 0, 10, 10)
        bbox2 = (0, 15, 10, 25)
        assert bboxes_intersect(bbox1, bbox2) is False
