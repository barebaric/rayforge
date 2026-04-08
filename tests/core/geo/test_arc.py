import pytest
import math
from collections import namedtuple

from rayforge.core.geo.arc import (
    arc_intersects_circle,
    arc_intersects_rect,
    determine_arc_direction,
    find_closest_point_on_arc,
    get_arc_angles,
    get_arc_bounding_box,
    get_arc_midpoint,
    is_angle_between,
    is_arc_fully_inside_regions,
    normalize_angle,
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


@pytest.fixture
def selection_rect():
    return (10.0, 10.0, 50.0, 50.0)


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


def test_determine_arc_direction_clockwise():
    """Test clockwise arc direction determination."""
    center = (0, 0)
    start = (10, 0)

    # Mouse below the start point (Y-down screen coords) = clockwise
    mouse = (5, -5)
    result = determine_arc_direction(center, start, mouse)
    assert result is True


def test_determine_arc_direction_counter_clockwise():
    """Test counter-clockwise arc direction determination."""
    center = (0, 0)
    start = (10, 0)

    # Mouse above the start point = counter-clockwise
    mouse = (5, 5)
    result = determine_arc_direction(center, start, mouse)
    assert result is False


def test_determine_arc_direction_quadrants():
    """Test arc direction in all quadrants."""
    center = (0, 0)

    # Start in Quadrant I
    start = (10, 10)
    mouse_cw = (15, 5)
    mouse_ccw = (5, 15)
    assert determine_arc_direction(center, start, mouse_cw) is True
    assert determine_arc_direction(center, start, mouse_ccw) is False

    # Start in Quadrant II
    start = (-10, 10)
    mouse_cw = (-5, 15)
    mouse_ccw = (-15, 5)
    assert determine_arc_direction(center, start, mouse_cw) is True
    assert determine_arc_direction(center, start, mouse_ccw) is False

    # Start in Quadrant III
    start = (-10, -10)
    mouse_cw = (-15, -5)
    mouse_ccw = (-5, -15)
    assert determine_arc_direction(center, start, mouse_cw) is True
    assert determine_arc_direction(center, start, mouse_ccw) is False

    # Start in Quadrant IV
    start = (10, -10)
    mouse_cw = (5, -15)
    mouse_ccw = (15, -5)
    assert determine_arc_direction(center, start, mouse_cw) is True
    assert determine_arc_direction(center, start, mouse_ccw) is False


def test_determine_arc_direction_cardinal_directions():
    """Test arc direction from cardinal start points."""
    center = (0, 0)

    # Start at East (0 radians)
    start = (10, 0)
    assert determine_arc_direction(center, start, (5, -5)) is True
    assert determine_arc_direction(center, start, (5, 5)) is False

    # Start at North (pi/2 radians)
    start = (0, 10)
    assert determine_arc_direction(center, start, (5, 5)) is True
    assert determine_arc_direction(center, start, (-5, 5)) is False

    # Start at West (pi radians)
    start = (-10, 0)
    assert determine_arc_direction(center, start, (-5, 5)) is True
    assert determine_arc_direction(center, start, (-5, -5)) is False

    # Start at South (3*pi/2 radians)
    start = (0, -10)
    assert determine_arc_direction(center, start, (-5, -5)) is True
    assert determine_arc_direction(center, start, (5, -5)) is False


def test_determine_arc_direction_offset_center():
    """Test arc direction with offset center."""
    center = (100, 200)
    start = (110, 200)

    # Mouse below start (relative to center)
    mouse = (105, 195)
    result = determine_arc_direction(center, start, mouse)
    assert result is True

    # Mouse above start (relative to center)
    mouse = (105, 205)
    result = determine_arc_direction(center, start, mouse)
    assert result is False


def test_determine_arc_direction_colinear():
    """Test arc direction when points are colinear."""
    center = (0, 0)
    start = (10, 0)

    # Mouse colinear with center and start
    mouse = (20, 0)
    result = determine_arc_direction(center, start, mouse)
    # Cross product is 0, so result should be False
    assert result is False


class TestArcIntersectsCircle:
    def test_arc_start_inside_circle(self):
        start = (5, 5)
        end = (5, -5)
        center = (5, 0)
        assert arc_intersects_circle(start, end, center, True, (5, 5), 3)

    def test_arc_end_inside_circle(self):
        start = (10, 0)
        end = (0, 10)
        center = (5, 5)
        assert arc_intersects_circle(start, end, center, False, (0, 10), 3)

    def test_arc_entirely_inside_circle(self):
        start = (1, 0)
        end = (0, 1)
        center = (0, 0)
        assert arc_intersects_circle(start, end, center, False, (0, 0), 5)

    def test_arc_crosses_circle(self):
        start = (10, 0)
        end = (0, 10)
        center = (0, 0)
        assert arc_intersects_circle(start, end, center, False, (7, 7), 3)

    def test_arc_outside_circle(self):
        start = (10, 0)
        end = (0, 10)
        center = (0, 0)
        assert not arc_intersects_circle(
            start, end, center, False, (-10, -10), 1
        )

    def test_degenerate_arc_inside(self):
        start = (3, 0)
        end = (3, 0)
        center = (3, 0)
        assert arc_intersects_circle(start, end, center, True, (3, 0), 5)

    def test_degenerate_arc_outside(self):
        start = (10, 0)
        end = (10, 0)
        center = (10, 0)
        assert not arc_intersects_circle(start, end, center, True, (0, 0), 1)

    def test_arc_midpoint_inside_circle(self):
        start = (6, 0)
        end = (0, 6)
        center = (0, 0)
        mid = get_arc_midpoint(start, end, center, False)
        cc = (mid[0], mid[1])
        assert arc_intersects_circle(start, end, center, False, cc, 1)


class TestIsArcFullyInsideRegions:
    def _square(self, x, y, w, h):
        return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]

    def test_small_arc_inside_large_square(self):
        start = (4, 5)
        end = (6, 5)
        co = (1, 0)
        regions = [self._square(0, 0, 10, 10)]
        assert is_arc_fully_inside_regions(start, end, co, True, regions)

    def test_arc_extending_outside(self):
        start = (1, 5)
        end = (9, 5)
        co = (4, 0)
        regions = [self._square(3, 0, 4, 10)]
        assert not is_arc_fully_inside_regions(start, end, co, True, regions)

    def test_arc_fully_outside(self):
        start = (50, 50)
        end = (52, 50)
        co = (1, 0)
        regions = [self._square(0, 0, 10, 10)]
        assert not is_arc_fully_inside_regions(start, end, co, True, regions)

    def test_empty_regions(self):
        start = (4, 5)
        end = (6, 5)
        co = (1, 0)
        assert not is_arc_fully_inside_regions(start, end, co, True, [])

    def test_arc_center_inside_but_bbox_corner_outside(self):
        start = (9, 5)
        end = (11, 5)
        co = (1, 0)
        regions = [self._square(0, 0, 10, 10)]
        assert not is_arc_fully_inside_regions(start, end, co, True, regions)

    def test_multiple_regions_containing_arc(self):
        start = (4, 5)
        end = (6, 5)
        co = (1, 0)
        regions = [
            self._square(0, 0, 5, 10),
            self._square(5, 0, 5, 10),
        ]
        assert is_arc_fully_inside_regions(start, end, co, True, regions)
