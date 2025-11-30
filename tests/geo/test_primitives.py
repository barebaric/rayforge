import pytest
import math
from collections import namedtuple

from rayforge.core.geo.primitives import (
    is_point_in_polygon,
    line_segment_intersection,
    get_segment_region_intersections,
    line_intersection,
    is_angle_between,
    get_arc_bounding_box,
    normalize_angle,
    get_arc_angles,
    get_arc_midpoint,
    circle_circle_intersection,
    is_point_on_segment,
    find_closest_point_on_line_segment,
    find_closest_point_on_arc,
    find_closest_point_on_line,
    is_point_in_rect,
    rect_a_contains_rect_b,
    line_segment_intersects_rect,
    arc_intersects_rect,
    circle_is_contained_by_rect,
    circle_intersects_rect,
)


@pytest.mark.parametrize(
    "target, start, end, clockwise, expected",
    [
        # Counter-Clockwise (CCW), No Wrap
        (math.pi / 2, math.pi / 4, 3 * math.pi / 4, False, True),  # Inside
        (0, math.pi / 4, 3 * math.pi / 4, False, False),  # Outside (before)
        (
            math.pi,
            math.pi / 4,
            3 * math.pi / 4,
            False,
            False,
        ),  # Outside (after)
        (
            math.pi / 4,
            math.pi / 4,
            3 * math.pi / 4,
            False,
            True,
        ),  # On start boundary
        (
            3 * math.pi / 4,
            math.pi / 4,
            3 * math.pi / 4,
            False,
            True,
        ),  # On end boundary
        # CCW, With Wrap (e.g., 315 deg to 45 deg)
        (0, 7 * math.pi / 4, math.pi / 4, False, True),  # Inside (on axis)
        (
            2 * math.pi - 0.1,
            7 * math.pi / 4,
            math.pi / 4,
            False,
            True,
        ),  # Inside
        (math.pi, 7 * math.pi / 4, math.pi / 4, False, False),  # Outside
        (
            7 * math.pi / 4,
            7 * math.pi / 4,
            math.pi / 4,
            False,
            True,
        ),  # On start
        (math.pi / 4, 7 * math.pi / 4, math.pi / 4, False, True),  # On end
        # Clockwise (CW), No Wrap
        (math.pi / 2, 3 * math.pi / 4, math.pi / 4, True, True),  # Inside
        (
            math.pi,
            3 * math.pi / 4,
            math.pi / 4,
            True,
            False,
        ),  # Outside (before)
        (0, 3 * math.pi / 4, math.pi / 4, True, False),  # Outside (after)
        (
            3 * math.pi / 4,
            3 * math.pi / 4,
            math.pi / 4,
            True,
            True,
        ),  # On start
        (math.pi / 4, 3 * math.pi / 4, math.pi / 4, True, True),  # On end
        # CW, With Wrap (e.g., 45 deg to 315 deg)
        (0, math.pi / 4, 7 * math.pi / 4, True, True),  # Inside (on axis)
        (math.pi, math.pi / 4, 7 * math.pi / 4, True, False),  # Outside
        (math.pi / 4, math.pi / 4, 7 * math.pi / 4, True, True),  # On start
        (7 * math.pi / 4, math.pi / 4, 7 * math.pi / 4, True, True),  # On end
        # Angle Normalization Tests (should behave identically to above)
        (math.pi / 2, math.pi / 4 + 2 * math.pi, 3 * math.pi / 4, False, True),
        (math.pi / 2, math.pi / 4, 3 * math.pi / 4 - 4 * math.pi, False, True),
        (math.pi / 2 + 6 * math.pi, math.pi / 4, 3 * math.pi / 4, False, True),
        # Zero-length arc
        (math.pi / 4, math.pi / 4, math.pi / 4, False, True),  # On point
        (math.pi / 2, math.pi / 4, math.pi / 4, False, False),  # Off point
        (math.pi / 4, math.pi / 4, math.pi / 4, True, True),  # On point
        (math.pi / 2, math.pi / 4, math.pi / 4, True, False),  # Off point
    ],
)
def test_is_angle_between(target, start, end, clockwise, expected):
    assert is_angle_between(target, start, end, clockwise) is expected


def get_arc_params(center, radius, start_angle, end_angle):
    """Helper to generate arc parameters for testing."""
    start_pos = (
        center[0] + radius * math.cos(start_angle),
        center[1] + radius * math.sin(start_angle),
    )
    end_pos = (
        center[0] + radius * math.cos(end_angle),
        center[1] + radius * math.sin(end_angle),
    )
    center_offset = (center[0] - start_pos[0], center[1] - start_pos[1])
    return start_pos, end_pos, center_offset


def test_get_arc_bounding_box_within_quadrant():
    """Test arc that does not cross any cardinal axes."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi / 6, math.pi / 3  # Both in Q1
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    # CCW
    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox = (
        min(start_pos[0], end_pos[0]),
        min(start_pos[1], end_pos[1]),
        max(start_pos[0], end_pos[0]),
        max(start_pos[1], end_pos[1]),
    )
    assert bbox == pytest.approx(expected_bbox)

    # CW (swap start and end)
    bbox_cw = get_arc_bounding_box(end_pos, start_pos, center_offset, True)
    assert bbox_cw == pytest.approx(expected_bbox)


def test_get_arc_bounding_box_crosses_east():
    """Test arc crossing the 0 radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = -math.pi / 4, math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_max_x = center[0] + radius
    assert bbox[2] == pytest.approx(expected_max_x)
    assert bbox[0] == pytest.approx(min(start_pos[0], end_pos[0]))
    assert bbox[1] == pytest.approx(min(start_pos[1], end_pos[1]))
    assert bbox[3] == pytest.approx(max(start_pos[1], end_pos[1]))


def test_get_arc_bounding_box_crosses_north_y_up():
    """Test arc crossing the PI/2 radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi / 4, 3 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_max_y = center[1] + radius
    assert bbox[3] == pytest.approx(expected_max_y)
    assert bbox[0] == pytest.approx(min(start_pos[0], end_pos[0]))
    assert bbox[1] == pytest.approx(min(start_pos[1], end_pos[1]))
    assert bbox[2] == pytest.approx(max(start_pos[0], end_pos[0]))


def test_get_arc_bounding_box_crosses_west():
    """Test arc crossing the PI radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = 3 * math.pi / 4, 5 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_min_x = center[0] - radius
    assert bbox[0] == pytest.approx(expected_min_x)
    assert bbox[1] == pytest.approx(min(start_pos[1], end_pos[1]))
    assert bbox[2] == pytest.approx(max(start_pos[0], end_pos[0]))
    assert bbox[3] == pytest.approx(max(start_pos[1], end_pos[1]))


def test_get_arc_bounding_box_crosses_south_y_up():
    """Test arc crossing the 3*PI/2 radian axis."""
    center, radius = (100, 200), 50
    start_angle, end_angle = 5 * math.pi / 4, 7 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_min_y = center[1] - radius
    assert bbox[1] == pytest.approx(expected_min_y)
    assert bbox[0] == pytest.approx(min(start_pos[0], end_pos[0]))
    assert bbox[2] == pytest.approx(max(start_pos[0], end_pos[0]))
    assert bbox[3] == pytest.approx(max(start_pos[1], end_pos[1]))


def test_get_arc_bounding_box_semicircle():
    """Test a 180-degree arc."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi, 0
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    # A CCW arc from pi to 0 traces the BOTTOM semicircle, crossing 3*pi/2.
    bbox_ccw = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox_bottom = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1],
    )
    assert bbox_ccw == pytest.approx(expected_bbox_bottom)

    # A CW arc from pi to 0 traces the TOP semicircle, crossing pi/2.
    bbox_cw = get_arc_bounding_box(start_pos, end_pos, center_offset, True)
    expected_bbox_top = (
        center[0] - radius,
        center[1],
        center[0] + radius,
        center[1] + radius,
    )
    assert bbox_cw == pytest.approx(expected_bbox_top)


def test_get_arc_bounding_box_large_arc_crossing_three_axes():
    """Test a large arc that spans multiple quadrants."""
    center, radius = (100, 200), 50
    # CCW from 45 degrees to 315 degrees
    start_angle, end_angle = math.pi / 4, 7 * math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    # Crosses North (PI/2), West (PI), and South (3*PI/2)
    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox = (
        center[0] - radius,  # From crossing West
        center[1] - radius,  # From crossing South
        start_pos[0],  # Max x is from start/end points
        center[1] + radius,  # From crossing North
    )
    assert bbox == pytest.approx(expected_bbox)


def test_get_arc_bounding_box_zero_length_arc():
    """Test a zero-length arc."""
    center, radius = (100, 200), 50
    start_angle, end_angle = math.pi / 4, math.pi / 4
    start_pos, end_pos, center_offset = get_arc_params(
        center, radius, start_angle, end_angle
    )

    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, False)
    expected_bbox = (start_pos[0], start_pos[1], end_pos[0], end_pos[1])
    assert bbox == pytest.approx(expected_bbox)


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


# --- NEW TESTS START HERE ---


@pytest.mark.parametrize(
    "angle, expected",
    [
        (math.pi / 2, math.pi / 2),
        (-math.pi / 2, 3 * math.pi / 2),
        (3 * math.pi, math.pi),
        (2 * math.pi, 0),
        (0, 0),
    ],
)
def test_normalize_angle(angle, expected):
    assert normalize_angle(angle) == pytest.approx(expected)


def test_get_arc_angles():
    center = (0, 0)
    # 90 degree CCW arc
    start_pos, end_pos = (1, 0), (0, 1)
    start_a, end_a, sweep = get_arc_angles(
        start_pos, end_pos, center, clockwise=False
    )
    assert start_a == pytest.approx(0)
    assert end_a == pytest.approx(math.pi / 2)
    assert sweep == pytest.approx(math.pi / 2)

    # 90 degree CW arc
    start_a_cw, end_a_cw, sweep_cw = get_arc_angles(
        start_pos, end_pos, center, clockwise=True
    )
    assert start_a_cw == pytest.approx(0)
    assert end_a_cw == pytest.approx(math.pi / 2)
    assert sweep_cw == pytest.approx(-3 * math.pi / 2)

    # CCW wrap-around
    start_pos, end_pos = (-1, -1), (1, -1)
    start_a, end_a, sweep = get_arc_angles(
        start_pos, end_pos, center, clockwise=False
    )
    # atan2 returns angles in [-pi, pi]. The test must match this range.
    assert start_a == pytest.approx(-3 * math.pi / 4)
    assert end_a == pytest.approx(-math.pi / 4)
    assert sweep == pytest.approx(math.pi / 2)


def test_get_arc_midpoint():
    center = (10, 20)
    radius = 5
    # 90 degree CCW arc from East to North
    start_pos = (center[0] + radius, center[1])
    end_pos = (center[0], center[1] + radius)
    midpoint = get_arc_midpoint(start_pos, end_pos, center, clockwise=False)
    expected_mid_angle = math.pi / 4
    expected_midpoint = (
        center[0] + radius * math.cos(expected_mid_angle),
        center[1] + radius * math.sin(expected_mid_angle),
    )
    assert midpoint == pytest.approx(expected_midpoint)

    # 180 degree CW arc from West to East (top semi-circle)
    start_pos = (center[0] - radius, center[1])
    end_pos = (center[0] + radius, center[1])
    midpoint_cw = get_arc_midpoint(start_pos, end_pos, center, clockwise=True)
    # Midpoint should be at the North point
    expected_midpoint_cw = (center[0], center[1] + radius)
    assert midpoint_cw == pytest.approx(expected_midpoint_cw)


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

    # Case 5: Line where projection is outside the segment p1-p2
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


# Mock object for arc commands used in find_closest_point_on_arc
MockArc = namedtuple("MockArc", ["end", "center_offset", "clockwise"])


def test_find_closest_point_on_arc():
    start_pos = (10, 0, 0)  # (x, y, z)
    center = (0, 0)

    # Test 1: Closest point is projection onto arc
    # 180 degree CCW arc from (10,0) to (-10,0)
    end_pos = (-10, 0, 0)
    center_offset = (center[0] - start_pos[0], center[1] - start_pos[1])
    arc_cmd = MockArc(
        end=end_pos, center_offset=center_offset, clockwise=False
    )
    x, y = 0, 20  # Point to check
    result = find_closest_point_on_arc(arc_cmd, start_pos, x, y)
    assert result is not None
    t, pt, dist_sq = result
    assert t == pytest.approx(0.5)
    assert pt == pytest.approx((0, 10))  # Closest point is top of the arc
    assert dist_sq == pytest.approx(100)  # (20-10)^2

    # Test 2: Closest point is start of arc
    x, y = 20, 0
    result = find_closest_point_on_arc(arc_cmd, start_pos, x, y)
    assert result is not None
    t, pt, dist_sq = result
    assert t == pytest.approx(0.0)
    assert pt == pytest.approx((10, 0))
    assert dist_sq == pytest.approx(100)  # (20-10)^2

    # Test 3: Closest point is end of arc
    x, y = -20, 0
    result = find_closest_point_on_arc(arc_cmd, start_pos, x, y)
    assert result is not None
    t, pt, dist_sq = result
    assert t == pytest.approx(1.0)
    assert pt == pytest.approx((-10, 0))
    assert dist_sq == pytest.approx(100)  # (-20 - -10)^2

    # Test 4: Point is the center (should default to start point)
    x, y = 0, 0
    result = find_closest_point_on_arc(arc_cmd, start_pos, x, y)
    assert result is not None
    t, pt, dist_sq = result
    assert pt == pytest.approx((10, 0))
    assert dist_sq == pytest.approx(100)

    # Test 5: Spiral (different start/end radius), should trigger linearization
    # This is an approximation, so we can't be too strict on the assertions
    # Make a spiral from radius 10 to radius 5
    end_pos_spiral = (-5, 0, 0)
    center_offset = (center[0] - start_pos[0], center[1] - start_pos[1])
    arc_cmd_spiral = MockArc(
        end=end_pos_spiral, center_offset=center_offset, clockwise=False
    )
    x, y = 0, 20  # Point to check
    result = find_closest_point_on_arc(arc_cmd_spiral, start_pos, x, y)
    assert result is not None  # Just check it returns a result
    t, pt, dist_sq = result
    assert 0.0 <= t <= 1.0
    # The closest point should be near the top of the arc, around (0, 7.5)
    assert pt[0] == pytest.approx(0, abs=2)
    assert pt[1] == pytest.approx(7.5, abs=2)


# --- NEW TESTS for refactored entity logic ---


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


def test_arc_intersects_rect(selection_rect):
    # Center (30,30), radius 10. Arc from East to North. Fully inside.
    assert arc_intersects_rect(
        (40, 30), (30, 40), (30, 30), False, selection_rect
    )
    # Arc from outside to inside
    assert arc_intersects_rect(
        (0, 30), (30, 40), (30, 30), False, selection_rect
    )
    # Arc passing through (top semi-circle)
    assert arc_intersects_rect(
        (0, 30), (60, 30), (30, 0), True, selection_rect
    )
    # Arc fully outside
    assert not arc_intersects_rect(
        (100, 100), (110, 110), (100, 110), False, selection_rect
    )
    # Arc bbox intersects, but arc does not (C-shape arc around a corner)
    assert not arc_intersects_rect(
        (0, 30), (30, 0), (0, 0), False, selection_rect
    )


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
