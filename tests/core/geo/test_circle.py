import pytest
import math

from rayforge.core.geo.circle import (
    circle_circle_intersection,
    circle_intersects_rect,
    circle_is_contained_by_rect,
    line_segment_intersects_circle,
    project_point_onto_circle,
)


@pytest.fixture
def selection_rect():
    return (10.0, 10.0, 50.0, 50.0)


def test_circle_circle_intersection():
    # Two intersection points
    c1, r1 = (0, 0), 5
    c2, r2 = (8, 0), 5
    intersections = circle_circle_intersection(c1, r1, c2, r2)
    assert len(intersections) == 2

    # Sort both actual and expected results for a stable, order-independent
    # comparison
    sorted_intersections = sorted(intersections)
    expected_intersections = sorted([(4.0, 3.0), (4.0, -3.0)])
    assert sorted_intersections == pytest.approx(expected_intersections)

    # One intersection point (tangent)
    c1, r1 = (0, 0), 5
    c2, r2 = (10, 0), 5
    intersections = circle_circle_intersection(c1, r1, c2, r2)
    assert (
        len(intersections) == 2
    )  # Due to float precision, may return two very close points
    assert intersections[0] == pytest.approx((5, 0))
    assert intersections[1] == pytest.approx((5, 0))

    # No intersection (separate)
    c1, r1 = (0, 0), 5
    c2, r2 = (11, 0), 5
    assert circle_circle_intersection(c1, r1, c2, r2) == []

    # No intersection (one inside other)
    c1, r1 = (0, 0), 10
    c2, r2 = (1, 0), 1
    assert circle_circle_intersection(c1, r1, c2, r2) == []

    # Coincident circles
    c1, r1 = (0, 0), 5
    c2, r2 = (0, 0), 5
    assert circle_circle_intersection(c1, r1, c2, r2) == []


def test_circle_is_contained_by_rect(selection_rect):
    # Fully contained
    assert circle_is_contained_by_rect((30, 30), 10, selection_rect)
    # Touching edge
    assert circle_is_contained_by_rect((20, 30), 10, selection_rect)
    # Intersecting
    assert not circle_is_contained_by_rect((5, 30), 10, selection_rect)
    # Outside
    assert not circle_is_contained_by_rect((100, 100), 5, selection_rect)


def test_circle_intersects_rect(selection_rect):
    # Intersects
    assert circle_intersects_rect((5, 30), 10, selection_rect)
    # Fully contained (should not intersect boundary)
    assert not circle_intersects_rect((30, 30), 5, selection_rect)
    # Rect is fully contained in circle (should not intersect boundary)
    assert not circle_intersects_rect((30, 30), 100, selection_rect)
    # Touching
    assert circle_intersects_rect((0, 30), 10, selection_rect)
    # Separate
    assert not circle_intersects_rect((100, 100), 5, selection_rect)


def test_project_point_onto_circle_basic():
    """Test projecting a point onto circle circumference."""
    center = (0, 0)
    radius = 10.0
    point = (20, 0)
    result = project_point_onto_circle(point, center, radius)
    assert result is not None
    assert result == pytest.approx((10.0, 0.0))


def test_project_point_onto_circle_quadrants():
    """Test projection in all four quadrants."""
    center = (0, 0)
    radius = 5.0

    # Quadrant I
    result = project_point_onto_circle((10, 10), center, radius)
    assert result is not None
    assert result[0] > 0 and result[1] > 0

    # Quadrant II
    result = project_point_onto_circle((-10, 10), center, radius)
    assert result is not None
    assert result[0] < 0 and result[1] > 0

    # Quadrant III
    result = project_point_onto_circle((-10, -10), center, radius)
    assert result is not None
    assert result[0] < 0 and result[1] < 0

    # Quadrant IV
    result = project_point_onto_circle((10, -10), center, radius)
    assert result is not None
    assert result[0] > 0 and result[1] < 0


def test_project_point_onto_circle_at_center():
    """Test projecting from center returns None."""
    center = (0, 0)
    radius = 10.0
    result = project_point_onto_circle((0, 0), center, radius)
    assert result is None


def test_project_point_onto_circle_near_center():
    """Test projecting from near center returns None."""
    center = (0, 0)
    radius = 10.0
    result = project_point_onto_circle((1e-10, 1e-10), center, radius)
    assert result is None


def test_project_point_onto_circle_on_circumference():
    """Test projecting a point already on the circle."""
    center = (0, 0)
    radius = 10.0
    point = (10, 0)
    result = project_point_onto_circle(point, center, radius)
    assert result == pytest.approx(point)


def test_project_point_onto_circle_offset_center():
    """Test projection with offset center."""
    center = (100, 200)
    radius = 50.0
    point = (150, 200)
    result = project_point_onto_circle(point, center, radius)
    assert result is not None
    assert result == pytest.approx((150.0, 200.0))


def test_project_point_onto_circle_diagonal():
    """Test projection from diagonal direction."""
    center = (0, 0)
    radius = math.sqrt(2)
    point = (10, 10)
    result = project_point_onto_circle(point, center, radius)
    assert result is not None
    # Projected point should be on the 45-degree line
    assert abs(result[0] - result[1]) < 1e-9
    # Distance from center should equal radius
    dist = math.hypot(result[0], result[1])
    assert dist == pytest.approx(radius)


class TestLineSegmentIntersectsCircle:
    def test_segment_crosses_circle(self):
        assert line_segment_intersects_circle((0, 0), (10, 0), (5, 0), 2)

    def test_segment_tangent_to_circle(self):
        assert line_segment_intersects_circle((0, 0), (10, 0), (5, 2), 2)

    def test_segment_outside_circle(self):
        assert not line_segment_intersects_circle((0, 5), (10, 5), (5, 0), 2)

    def test_segment_entirely_inside_circle(self):
        assert line_segment_intersects_circle((4, 0), (6, 0), (5, 0), 10)

    def test_one_endpoint_inside(self):
        assert line_segment_intersects_circle((4, 0), (20, 0), (5, 0), 3)

    def test_zero_length_inside(self):
        assert line_segment_intersects_circle((5, 0), (5, 0), (5, 0), 1)

    def test_zero_length_outside(self):
        assert not line_segment_intersects_circle((10, 0), (10, 0), (5, 0), 1)
