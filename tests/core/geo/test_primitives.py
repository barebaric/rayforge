import pytest

from rayforge.core.geo.primitives import (
    find_closest_point_on_line,
    find_closest_point_on_line_segment,
    get_segment_region_intersections,
    is_point_in_polygon,
    is_point_in_rect,
    is_point_on_segment,
    line_intersection,
    line_segment_intersection,
    line_segment_intersects_rect,
    midpoint,
    rect_a_contains_rect_b,
)


@pytest.fixture
def square_polygon():
    return [(0, 0), (10, 0), (10, 10), (0, 10)]


def test_is_point_in_polygon(square_polygon):
    # Points inside
    assert is_point_in_polygon((5, 5), square_polygon) is True
    assert is_point_in_polygon((0.1, 0.1), square_polygon) is True

    # Points outside
    assert is_point_in_polygon((15, 5), square_polygon) is False
    assert is_point_in_polygon((-5, 5), square_polygon) is False
    assert is_point_in_polygon((5, 15), square_polygon) is False
    assert is_point_in_polygon((5, -5), square_polygon) is False

    # Points on edge should be considered inside
    assert is_point_in_polygon((5, 0), square_polygon) is True  # Bottom edge
    assert is_point_in_polygon((10, 5), square_polygon) is True  # Right edge
    assert is_point_in_polygon((5, 10), square_polygon) is True  # Top edge
    assert is_point_in_polygon((0, 5), square_polygon) is True  # Left edge
    assert is_point_in_polygon((0, 0), square_polygon) is True  # Corner
    assert is_point_in_polygon((10, 10), square_polygon) is True  # Corner


def test_line_segment_intersection():
    # Crossing lines
    p1, p2 = (0, 0), (10, 10)
    p3, p4 = (0, 10), (10, 0)
    assert line_segment_intersection(p1, p2, p3, p4) == pytest.approx((5, 5))

    # T-junction (endpoint on segment)
    p1, p2 = (0, 0), (10, 0)
    p3, p4 = (5, -5), (5, 5)
    assert line_segment_intersection(p1, p2, p3, p4) == pytest.approx((5, 0))

    # No intersection (parallel)
    p1, p2 = (0, 0), (10, 0)
    p3, p4 = (0, 5), (10, 5)
    assert line_segment_intersection(p1, p2, p3, p4) is None

    # No intersection (not parallel, but segments don't meet)
    p1, p2 = (0, 0), (1, 1)
    p3, p4 = (0, 10), (1, 9)
    assert line_segment_intersection(p1, p2, p3, p4) is None

    # Collinear, overlapping
    p1, p2 = (0, 0), (5, 0)
    p3, p4 = (3, 0), (8, 0)
    # Our simple implementation returns None for collinear cases.
    assert line_segment_intersection(p1, p2, p3, p4) is None


def test_get_segment_region_intersections():
    p1 = (0.0, 50.0)
    p2 = (100.0, 50.0)
    region = [(40.0, 45.0), (60.0, 45.0), (60.0, 55.0), (40.0, 55.0)]

    # Test a simple crossing
    intersections = get_segment_region_intersections(p1, p2, [region])
    # Should find intersections at 40% and 60% of the line
    assert intersections == pytest.approx([0.0, 0.4, 0.6, 1.0])

    # Test a line fully outside
    p_out1 = (-20, 0)
    p_out2 = (-10, 0)
    intersections = get_segment_region_intersections(p_out1, p_out2, [region])
    # Should only return the start and end points
    assert intersections == pytest.approx([0.0, 1.0])


def test_line_intersection():
    # Intersection
    p1, p2 = (0, 0), (10, 10)
    p3, p4 = (0, 10), (10, 0)
    assert line_intersection(p1, p2, p3, p4) == pytest.approx((5, 5))

    # Parallel lines
    p1, p2 = (0, 0), (10, 0)
    p3, p4 = (0, 1), (10, 1)
    assert line_intersection(p1, p2, p3, p4) is None

    # Intersection outside segments (infinite lines)
    p1, p2 = (0, 0), (1, 0)
    p3, p4 = (0, 1), (0, 2)
    # x-axis and y-axis intersect at 0,0
    assert line_intersection(p1, p2, p3, p4) == pytest.approx((0, 0))


def test_is_point_on_segment():
    p1, p2 = (0, 0), (10, 10)
    # Point in the middle
    assert is_point_on_segment((5, 5), p1, p2) is True
    # Endpoints
    assert is_point_on_segment((0, 0), p1, p2) is True
    assert is_point_on_segment((10, 10), p1, p2) is True
    # Point on the line, but outside segment
    assert is_point_on_segment((11, 11), p1, p2) is False
    assert is_point_on_segment((-1, -1), p1, p2) is False


def test_find_closest_point_on_line():
    # Case 1: Simple horizontal line
    p1, p2 = (0, 0), (10, 0)
    x, y = 5, 5
    assert find_closest_point_on_line(p1, p2, x, y) == pytest.approx((5, 0))

    # Case 2: Simple vertical line
    p1, p2 = (0, 0), (0, 10)
    x, y = 5, 5
    assert find_closest_point_on_line(p1, p2, x, y) == pytest.approx((0, 5))

    # Case 3: Diagonal line y=x
    p1, p2 = (0, 0), (10, 10)
    x, y = 0, 10
    assert find_closest_point_on_line(p1, p2, x, y) == pytest.approx((5, 5))

    # Case 4: Point is already on the line
    p1, p2 = (0, 0), (10, 10)
    x, y = 3, 3
    assert find_closest_point_on_line(p1, p2, x, y) == pytest.approx((3, 3))

    # Case 5: Point is outside the segment p1-p2
    # The function is for an *infinite* line, so it should still work
    p1, p2 = (0, 0), (10, 0)
    x, y = 20, 5
    assert find_closest_point_on_line(p1, p2, x, y) == pytest.approx((20, 0))

    # Case 6: Edge case - p1 and p2 are the same point
    p1, p2 = (5, 5), (5, 5)
    x, y = 10, 10
    # The function should return p1 in this case
    assert find_closest_point_on_line(p1, p2, x, y) == pytest.approx((5, 5))


def test_find_closest_point_on_line_segment():
    p1, p2 = (0, 0), (10, 0)

    # Closest point is projection
    t, pt, dist_sq = find_closest_point_on_line_segment(p1, p2, 5, 5)
    assert t == pytest.approx(0.5)
    assert pt == pytest.approx((5, 0))
    assert dist_sq == pytest.approx(25)

    # Closest point is p1
    t, pt, dist_sq = find_closest_point_on_line_segment(p1, p2, -5, 5)
    assert t == pytest.approx(0.0)
    assert pt == pytest.approx((0, 0))
    assert dist_sq == pytest.approx(50)

    # Closest point is p2
    t, pt, dist_sq = find_closest_point_on_line_segment(p1, p2, 15, 5)
    assert t == pytest.approx(1.0)
    assert pt == pytest.approx((10, 0))
    assert dist_sq == pytest.approx(50)

    # Point is on the segment
    t, pt, dist_sq = find_closest_point_on_line_segment(p1, p2, 7, 0)
    assert t == pytest.approx(0.7)
    assert pt == pytest.approx((7, 0))
    assert dist_sq == pytest.approx(0)


@pytest.fixture
def selection_rect():
    return (10.0, 10.0, 50.0, 50.0)


def test_is_point_in_rect(selection_rect):
    # Inside
    assert is_point_in_rect((25, 25), selection_rect)
    # On edge
    assert is_point_in_rect((10, 25), selection_rect)
    assert is_point_in_rect((25, 50), selection_rect)
    # Outside
    assert not is_point_in_rect((5, 25), selection_rect)
    assert not is_point_in_rect((60, 25), selection_rect)


def test_rect_a_contains_rect_b(selection_rect):
    # Fully contained
    contained_rect = (20, 20, 40, 40)
    assert rect_a_contains_rect_b(selection_rect, contained_rect)
    # Touching edge
    touching_rect = (10, 20, 40, 40)
    assert rect_a_contains_rect_b(selection_rect, touching_rect)
    # Intersecting
    intersecting_rect = (40, 40, 60, 60)
    assert not rect_a_contains_rect_b(selection_rect, intersecting_rect)
    # Outside
    outside_rect = (100, 100, 120, 120)
    assert not rect_a_contains_rect_b(selection_rect, outside_rect)


def test_line_segment_intersects_rect(selection_rect):
    # Fully contained
    assert line_segment_intersects_rect((20, 20), (40, 40), selection_rect)
    # One point in, one out
    assert line_segment_intersects_rect((25, 25), (60, 60), selection_rect)
    # Crossing through
    assert line_segment_intersects_rect((0, 25), (60, 25), selection_rect)
    # Touching edge
    assert line_segment_intersects_rect((0, 10), (20, 10), selection_rect)
    # Fully outside
    assert not line_segment_intersects_rect((0, 0), (5, 5), selection_rect)
    # Bbox intersects, and segment does too (diagonal case)
    assert line_segment_intersects_rect((0, 60), (60, 0), selection_rect)


def test_midpoint():
    a = (1.0, 2.0, 3.0)
    b = (5.0, 6.0, 7.0)
    assert midpoint(a, b) == (3.0, 4.0, 5.0)


def test_midpoint_negative():
    assert midpoint((-2.0, 0.0, 4.0), (2.0, 0.0, -4.0)) == (
        0.0,
        0.0,
        0.0,
    )
