import math
from typing import List, Tuple, Optional, Any
import numpy as np
from .constants import (
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
    GEO_ARRAY_COLS,
)
from .primitives import (
    is_point_in_polygon,
    line_segment_intersects_rect,
    find_closest_point_on_line_segment,
)
from .types import Point, Point3D, Polygon, Rect


def normalize_angle(angle: float) -> float:
    """Normalizes an angle to the [0, 2*pi) range."""
    return (angle + 2 * math.pi) % (2 * math.pi)


def get_arc_angles(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
) -> Point3D:
    """
    Returns (start_angle, end_angle, sweep_angle) for an arc.
    Handles the clockwise/counter-clockwise logic and wrapping.
    """
    start_angle = math.atan2(
        start_pos[1] - center[1], start_pos[0] - center[0]
    )
    end_angle = math.atan2(end_pos[1] - center[1], end_pos[0] - center[0])

    sweep = end_angle - start_angle
    if clockwise:
        if sweep > 0:
            sweep -= 2 * math.pi
    else:
        if sweep < 0:
            sweep += 2 * math.pi

    return start_angle, end_angle, sweep


def get_arc_midpoint(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
) -> Point:
    """Calculates the midpoint coordinates along the arc's circumference."""
    start_a, _, sweep = get_arc_angles(start_pos, end_pos, center, clockwise)
    mid_angle = start_a + sweep / 2.0
    radius = math.hypot(start_pos[0] - center[0], start_pos[1] - center[1])
    return (
        center[0] + radius * math.cos(mid_angle),
        center[1] + radius * math.sin(mid_angle),
    )


def is_angle_between(
    target: float, start: float, end: float, clockwise: bool
) -> bool:
    """
    Checks if a target angle is within the sweep of an arc defined by start
    and end angles. Handles wrapping around 2*PI.
    """
    target = normalize_angle(target)
    start = normalize_angle(start)
    end = normalize_angle(end)

    if clockwise:
        if start < end:
            return target <= start or target >= end
        return end <= target <= start
    else:
        if start > end:
            return target >= start or target <= end
        return start <= target <= end


def get_arc_bounding_box(
    start_pos: Point,
    end_pos: Point,
    center_offset: Point,
    clockwise: bool,
) -> Rect:
    """
    Calculates the tight bounding box (min_x, min_y, max_x, max_y) for an arc.
    """
    center_x = start_pos[0] + center_offset[0]
    center_y = start_pos[1] + center_offset[1]
    radius = math.hypot(center_offset[0], center_offset[1])

    min_x = min(start_pos[0], end_pos[0])
    min_y = min(start_pos[1], end_pos[1])
    max_x = max(start_pos[0], end_pos[0])
    max_y = max(start_pos[1], end_pos[1])

    start_angle = math.atan2(start_pos[1] - center_y, start_pos[0] - center_x)
    end_angle = math.atan2(end_pos[1] - center_y, end_pos[0] - center_x)

    # Check if the arc sweep crosses the cardinal axes (0, 90, 180, 270 deg)
    # 0 radians (East)
    if is_angle_between(0, start_angle, end_angle, clockwise):
        max_x = max(max_x, center_x + radius)
    # PI/2 radians (South, in a Y-down coord system) or (North, Y-up)
    if is_angle_between(math.pi / 2, start_angle, end_angle, clockwise):
        max_y = max(max_y, center_y + radius)
    # PI radians (West)
    if is_angle_between(math.pi, start_angle, end_angle, clockwise):
        min_x = min(min_x, center_x - radius)
    # 3*PI/2 radians (North, in a Y-down coord system) or (South, Y-up)
    if is_angle_between(3 * math.pi / 2, start_angle, end_angle, clockwise):
        min_y = min(min_y, center_y - radius)

    return min_x, min_y, max_x, max_y


def determine_arc_direction(center: Point, start: Point, mouse: Point) -> bool:
    """
    Determines if an arc should be clockwise based on mouse position.

    Uses the cross product of vectors (Center->Start) and (Center->Mouse).
    In screen coordinates (Y-down), a positive cross product indicates
    clockwise direction.

    Args:
        center: The center point of the arc.
        start: The start point of the arc.
        mouse: The current mouse position.

    Returns:
        True if the arc should be clockwise, False for counter-clockwise.
    """
    vec_s_x = start[0] - center[0]
    vec_s_y = start[1] - center[1]
    vec_m_x = mouse[0] - center[0]
    vec_m_y = mouse[1] - center[1]

    det = vec_s_x * vec_m_y - vec_s_y * vec_m_x
    return bool(det < 0)


def _find_closest_on_linearized_arc(
    arc_row: np.ndarray,
    start_pos: Point3D,
    x: float,
    y: float,
) -> Optional[Tuple[float, Point, float]]:
    """Helper to find the closest point on a linearized arc."""
    from .linearize import linearize_arc

    arc_segments = linearize_arc(arc_row, start_pos)
    if not arc_segments:
        return None

    min_dist_sq_sub = float("inf")
    best_sub_result = None

    for j, (p1_3d, p2_3d) in enumerate(arc_segments):
        t_sub, pt_sub, dist_sq_sub = find_closest_point_on_line_segment(
            p1_3d[:2], p2_3d[:2], x, y
        )
        if dist_sq_sub < min_dist_sq_sub:
            min_dist_sq_sub = dist_sq_sub
            best_sub_result = (j, t_sub, pt_sub, dist_sq_sub)

    if not best_sub_result:
        return None

    j_best, t_sub_best, pt_best, dist_sq_best = best_sub_result
    t_arc = (j_best + t_sub_best) / len(arc_segments)
    return t_arc, pt_best, dist_sq_best


def _find_closest_point_on_arc_from_array(
    arc_row: np.ndarray,
    start_pos: Point3D,
    x: float,
    y: float,
) -> Optional[Tuple[float, Point, float]]:
    """Internal NumPy-native implementation."""
    p0 = start_pos[:2]
    p1 = (arc_row[COL_X], arc_row[COL_Y])
    center_offset = (arc_row[COL_I], arc_row[COL_J])
    clockwise = bool(arc_row[COL_CW])
    center = (
        p0[0] + center_offset[0],
        p0[1] + center_offset[1],
    )
    radius_start = math.dist(p0, center)
    radius_end = math.dist(p1, center)

    if not math.isclose(radius_start, radius_end):
        return _find_closest_on_linearized_arc(arc_row, start_pos, x, y)

    radius = radius_start
    if radius < 1e-9:
        dist_sq = (x - p0[0]) ** 2 + (y - p0[1]) ** 2
        return 0.0, p0, dist_sq

    vec_to_point = (x - center[0], y - center[1])
    dist_to_center = math.hypot(vec_to_point[0], vec_to_point[1])
    if dist_to_center < 1e-9:
        closest_on_circle = p0
    else:
        closest_on_circle = (
            center[0] + vec_to_point[0] / dist_to_center * radius,
            center[1] + vec_to_point[1] / dist_to_center * radius,
        )

    start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
    end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
    point_angle = math.atan2(
        closest_on_circle[1] - center[1], closest_on_circle[0] - center[0]
    )

    angle_range = end_angle - start_angle
    angle_to_check = point_angle - start_angle

    if clockwise:
        if angle_range > 1e-9:
            angle_range -= 2 * math.pi
        if angle_to_check > 1e-9:
            angle_to_check -= 2 * math.pi
    else:
        if angle_range < -1e-9:
            angle_range += 2 * math.pi
        if angle_to_check < -1e-9:
            angle_to_check += 2 * math.pi

    is_on_arc = False
    if clockwise:
        if angle_to_check >= angle_range - 1e-9 and angle_to_check <= 1e-9:
            is_on_arc = True
    else:
        if angle_to_check <= angle_range + 1e-9 and angle_to_check >= -1e-9:
            is_on_arc = True

    if is_on_arc:
        closest_point = closest_on_circle
        t = angle_to_check / angle_range if abs(angle_range) > 1e-9 else 0.0
    else:
        dist_sq_p0 = (x - p0[0]) ** 2 + (y - p0[1]) ** 2
        dist_sq_p1 = (x - p1[0]) ** 2 + (y - p1[1]) ** 2
        if dist_sq_p0 <= dist_sq_p1:
            closest_point, t = p0, 0.0
        else:
            closest_point, t = p1, 1.0

    dist_sq = (x - closest_point[0]) ** 2 + (y - closest_point[1]) ** 2
    t = max(0.0, min(1.0, t))
    return t, closest_point, dist_sq


def find_closest_point_on_arc(
    arc_input: Any, start_pos: Point3D, x: float, y: float
) -> Optional[Tuple[float, Point, float]]:
    """
    Finds the closest point on an arc, using an analytical method for
    circular arcs and falling back to linearization for spirals.
    This function is backward-compatible and accepts either a NumPy array row
    or an object with .end, .center_offset, and .clockwise attributes.
    """
    if isinstance(arc_input, np.ndarray):
        return _find_closest_point_on_arc_from_array(
            arc_input, start_pos, x, y
        )
    else:
        temp_row = np.zeros(GEO_ARRAY_COLS, dtype=np.float64)
        temp_row[COL_TYPE] = CMD_TYPE_ARC
        if hasattr(arc_input, "end") and arc_input.end is not None:
            temp_row[COL_X] = arc_input.end[0]
            temp_row[COL_Y] = arc_input.end[1]
            temp_row[COL_Z] = arc_input.end[2]
        if hasattr(arc_input, "center_offset"):
            temp_row[COL_I] = arc_input.center_offset[0]
            temp_row[COL_J] = arc_input.center_offset[1]
        if hasattr(arc_input, "clockwise"):
            temp_row[COL_CW] = 1.0 if arc_input.clockwise else 0.0
        return _find_closest_point_on_arc_from_array(temp_row, start_pos, x, y)


def arc_intersects_rect(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
    rect: Rect,
) -> bool:
    """Checks if an arc intersects with a rectangle."""

    # Broad phase: Check if arc's AABB intersects rect's AABB
    arc_box = get_arc_bounding_box(
        start_pos,
        end_pos,
        (center[0] - start_pos[0], center[1] - start_pos[1]),
        clockwise,
    )
    if (
        arc_box[2] < rect[0]
        or arc_box[0] > rect[2]
        or arc_box[3] < rect[1]
        or arc_box[1] > rect[3]
    ):
        return False

    # A mock command object for linearize_arc
    class MockArcCmd:
        def __init__(self, end, center_offset, is_clockwise):
            self.end = end
            self.center_offset = center_offset
            self.clockwise = is_clockwise

    # Detailed phase: linearize the arc and check each segment.
    mock_cmd = MockArcCmd(
        end=(end_pos[0], end_pos[1], 0.0),
        center_offset=(center[0] - start_pos[0], center[1] - start_pos[1]),
        is_clockwise=clockwise,
    )
    start_3d = (start_pos[0], start_pos[1], 0.0)
    radius = math.hypot(start_pos[0] - center[0], start_pos[1] - center[1])
    # Use a sensible resolution for selection hit-testing
    from .linearize import linearize_arc

    segments = linearize_arc(mock_cmd, start_3d, resolution=radius * 0.1)

    for p1_3d, p2_3d in segments:
        if line_segment_intersects_rect(p1_3d[:2], p2_3d[:2], rect):
            return True

    return False


def arc_intersects_circle(
    start_pos: Point,
    end_pos: Point,
    center: Point,
    clockwise: bool,
    circle_center: Point,
    circle_radius: float,
) -> bool:
    """
    Checks if an arc enters a circle (intersects boundary or is
    entirely inside).  Uses purely analytical geometry -- no linearization.
    """
    from .circle import circle_circle_intersection

    radius = math.hypot(start_pos[0] - center[0], start_pos[1] - center[1])
    if radius < 1e-9:
        return (
            math.hypot(
                start_pos[0] - circle_center[0],
                start_pos[1] - circle_center[1],
            )
            <= circle_radius
        )

    if (
        math.hypot(
            start_pos[0] - circle_center[0],
            start_pos[1] - circle_center[1],
        )
        <= circle_radius
    ):
        return True
    if (
        math.hypot(
            end_pos[0] - circle_center[0],
            end_pos[1] - circle_center[1],
        )
        <= circle_radius
    ):
        return True

    intersections = circle_circle_intersection(
        center, radius, circle_center, circle_radius
    )
    if intersections:
        start_angle = math.atan2(
            start_pos[1] - center[1], start_pos[0] - center[0]
        )
        end_angle = math.atan2(end_pos[1] - center[1], end_pos[0] - center[0])
        for pt in intersections:
            angle = math.atan2(pt[1] - center[1], pt[0] - center[0])
            if is_angle_between(angle, start_angle, end_angle, clockwise):
                return True

    mid = get_arc_midpoint(start_pos, end_pos, center, clockwise)
    if (
        math.hypot(mid[0] - circle_center[0], mid[1] - circle_center[1])
        <= circle_radius
    ):
        return True

    return False


def is_arc_fully_inside_regions(
    start_pos: Point,
    end_pos: Point,
    center_offset: Point,
    clockwise: bool,
    regions: List[Polygon],
) -> bool:
    center = (
        start_pos[0] + center_offset[0],
        start_pos[1] + center_offset[1],
    )
    bbox = get_arc_bounding_box(start_pos, end_pos, center_offset, clockwise)
    mid = get_arc_midpoint(start_pos, end_pos, center, clockwise)
    sample_points = [
        (bbox[0], bbox[1]),
        (bbox[2], bbox[1]),
        (bbox[2], bbox[3]),
        (bbox[0], bbox[3]),
        start_pos,
        end_pos,
        mid,
    ]
    for p in sample_points:
        if not any(is_point_in_polygon(p, region) for region in regions):
            return False
    return True
