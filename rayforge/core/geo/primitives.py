import math
from typing import List, Tuple, Optional, Any
from .linearize import linearize_arc


def normalize_angle(angle: float) -> float:
    """Normalizes an angle to the [0, 2*pi) range."""
    return (angle + 2 * math.pi) % (2 * math.pi)


def get_arc_angles(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    center: Tuple[float, float],
    clockwise: bool,
) -> Tuple[float, float, float]:
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
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    center: Tuple[float, float],
    clockwise: bool,
) -> Tuple[float, float]:
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
    # Normalize all angles to be in the range [0, 2*PI)
    target = normalize_angle(target)
    start = normalize_angle(start)
    end = normalize_angle(end)

    if clockwise:
        # For a clockwise arc, the sweep is from start down to end.
        # This can wrap around 0.
        if start < end:  # Wraps around 2pi (e.g., from 30 deg down to 330 deg)
            return target <= start or target >= end
        # Does not wrap (e.g., from 180 deg down to 90 deg)
        return end <= target <= start
    else:
        # For a counter-clockwise arc, the sweep is from start up to end.
        # This can wrap around 0.
        if start > end:  # Wraps around 2pi (e.g., from 330 deg up to 30 deg)
            return target >= start or target <= end
        # Does not wrap (e.g., from 90 deg up to 180 deg)
        return start <= target <= end


def circle_circle_intersection(
    c1: Tuple[float, float], r1: float, c2: Tuple[float, float], r2: float
) -> List[Tuple[float, float]]:
    """
    Calculates the intersection points of two circles.
    Returns a list of 0, 1, or 2 points.
    """
    dx, dy = c2[0] - c1[0], c2[1] - c1[1]
    d_sq = dx**2 + dy**2
    d = math.sqrt(d_sq)

    # Check for no intersection or concentric circles/containment
    if d < 1e-9 or d > r1 + r2 or d < abs(r1 - r2):
        return []

    a = (r1**2 - r2**2 + d_sq) / (2 * d)
    h_sq = max(0, r1**2 - a**2)
    h = math.sqrt(h_sq)

    x2 = c1[0] + a * dx / d
    y2 = c1[1] + a * dy / d

    x3_1 = x2 + h * dy / d
    y3_1 = y2 - h * dx / d
    x3_2 = x2 - h * dy / d
    y3_2 = y2 + h * dx / d

    return [(x3_1, y3_1), (x3_2, y3_2)]


def is_point_on_segment(
    pt: Tuple[float, float], p1: Tuple[float, float], p2: Tuple[float, float]
) -> bool:
    """
    Checks if a point is strictly on a line segment defined by two endpoints.
    Assumes the point is collinear with the segment.
    """
    # Vector P1->Pt dot P1->P2 >= 0
    dot1 = (pt[0] - p1[0]) * (p2[0] - p1[0]) + (pt[1] - p1[1]) * (
        p2[1] - p1[1]
    )
    if dot1 < 0:
        return False
    # Vector P2->Pt dot P2->P1 >= 0
    dot2 = (pt[0] - p2[0]) * (p1[0] - p2[0]) + (pt[1] - p2[1]) * (
        p1[1] - p2[1]
    )
    if dot2 < 0:
        return False
    return True


def get_arc_bounding_box(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    center_offset: Tuple[float, float],
    clockwise: bool,
) -> Tuple[float, float, float, float]:
    """
    Calculates the tight bounding box (min_x, min_y, max_x, max_y) for an arc.
    """
    center_x = start_pos[0] + center_offset[0]
    center_y = start_pos[1] + center_offset[1]
    radius = math.hypot(center_offset[0], center_offset[1])

    # Initialize bounds with the start and end points of the arc.
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


def is_point_in_polygon(
    point: Tuple[float, float], polygon: List[Tuple[float, float]]
) -> bool:
    """
    Checks if a point is inside or on the boundary of a polygon using a
    robust, multi-stage process (AABB -> Boundary -> Ray-Casting).
    """
    x, y = point
    n = len(polygon)
    if n < 3:
        return False

    # --- Stage 0: AABB Optimization ---
    # Fast fail if the point is outside the bounding box of the polygon.
    # We compute min/max manually to avoid creating intermediate lists.
    min_x = max_x = polygon[0][0]
    min_y = max_y = polygon[0][1]

    for px, py in polygon:
        if px < min_x:
            min_x = px
        elif px > max_x:
            max_x = px
        if py < min_y:
            min_y = py
        elif py > max_y:
            max_y = py

    if x < min_x or x > max_x or y < min_y or y > max_y:
        return False

    # --- Stage 1: Boundary Check ---
    # First, check if the point lies exactly on any of the polygon's edges.
    for i in range(n):
        p1 = polygon[i]
        p2 = polygon[(i + 1) % n]
        p1x, p1y = p1
        p2x, p2y = p2

        # Check for collinearity via cross-product (with a tolerance
        # for float errors)
        cross_product = (y - p1y) * (p2x - p1x) - (x - p1x) * (p2y - p1y)
        if abs(cross_product) < 1e-9:
            # If collinear, check if the point is within the segment's
            # bounding box
            if min(p1x, p2x) <= x <= max(p1x, p2x) and min(
                p1y, p2y
            ) <= y <= max(p1y, p2y):
                return True  # Point is on an edge

    # --- Stage 2: Ray-Casting for Interior Check ---
    # If not on the boundary, use ray-casting to check if it's in the interior.
    inside = False
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if p1y == p2y:
            # Skip horizontal edges in the ray-casting part
            p1x, p1y = p2x, p2y
            continue

        if min(p1y, p2y) < y <= max(p1y, p2y):
            # Calculate the x-intersection of the line segment and the ray.
            x_intersect = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if x_intersect > x:
                inside = not inside
        p1x, p1y = p2x, p2y

    return inside


def line_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Finds the intersection point of two infinite lines defined by pairs of
    points (p1, p2) and (p3, p4).
    Returns the intersection point or None if lines are parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
    if denom == 0:
        return None

    ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
    return (x1 + ua * (x2 - x1), y1 + ua * (y2 - y1))


def line_segment_intersection(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
) -> Optional[Tuple[float, float]]:
    """
    Finds the intersection point of two 2D line segments (p1,p2) and (p3,p4).
    Returns the intersection point or None.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None  # Parallel or collinear

    t_num = (x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)
    u_num = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3))

    t = t_num / den
    u = u_num / den

    if 0 <= t <= 1 and 0 <= u <= 1:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None


def find_closest_point_on_line(
    p1: Tuple[float, float], p2: Tuple[float, float], x: float, y: float
) -> Tuple[float, float]:
    """
    Finds the closest point on an infinite 2D line defined by p1 and p2
    to the point (x, y).

    Args:
        p1: First point defining the line (x, y).
        p2: Second point defining the line (x, y).
        x: The x-coordinate of the point to project.
        y: The y-coordinate of the point to project.

    Returns:
        The (x, y) coordinates of the closest point on the line.
    """
    # Vector from p1 to p2
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    # Vector from p1 to point (x,y)
    px, py = x - p1[0], y - p1[1]

    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        # p1 and p2 are practically the same point
        return p1

    # Project vector p onto vector d
    t = (px * dx + py * dy) / len_sq
    return p1[0] + t * dx, p1[1] + t * dy


def find_closest_point_on_line_segment(
    p1: Tuple[float, float], p2: Tuple[float, float], x: float, y: float
) -> Tuple[float, Tuple[float, float], float]:
    """
    Finds the closest point on a 2D line segment.

    Returns:
        A tuple containing:
        - The parameter `t` (from 0.0 to 1.0) along the segment.
        - A tuple of the (x, y) coordinates of the closest point.
        - The squared distance from the input point to the closest point.
    """
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:  # Treat as a single point
        t = 0.0
    else:
        # Project (x,y) onto the line defined by p1 and p2
        t = ((x - p1[0]) * dx + (y - p1[1]) * dy) / len_sq
        t = max(0.0, min(1.0, t))  # Clamp to the segment

    closest_x = p1[0] + t * dx
    closest_y = p1[1] + t * dy
    dist_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
    return t, (closest_x, closest_y), dist_sq


def _find_closest_on_linearized_arc(
    arc_cmd: Any, start_pos: Tuple[float, float, float], x: float, y: float
) -> Optional[Tuple[float, Tuple[float, float], float]]:
    """Helper to find the closest point on a linearized arc."""
    arc_segments = linearize_arc(arc_cmd, start_pos)
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


def find_closest_point_on_arc(
    arc_cmd: Any, start_pos: Tuple[float, float, float], x: float, y: float
) -> Optional[Tuple[float, Tuple[float, float], float]]:
    """
    Finds the closest point on an arc, using an analytical method for
    circular arcs and falling back to linearization for spirals.
    """
    p0 = start_pos[:2]
    p1 = arc_cmd.end[:2]
    center = (
        p0[0] + arc_cmd.center_offset[0],
        p0[1] + arc_cmd.center_offset[1],
    )
    radius_start = math.dist(p0, center)
    radius_end = math.dist(p1, center)

    if not math.isclose(radius_start, radius_end):
        return _find_closest_on_linearized_arc(arc_cmd, start_pos, x, y)

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

    if arc_cmd.clockwise:
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
    if arc_cmd.clockwise:
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


def get_segment_region_intersections(
    p1_2d: Tuple[float, float],
    p2_2d: Tuple[float, float],
    regions: List[List[Tuple[float, float]]],
) -> List[float]:
    """
    Calculates intersection points of a line segment with polygon boundaries.
    """
    cut_points_t = {0.0, 1.0}
    for region in regions:
        for i in range(len(region)):
            p3 = region[i]
            p4 = region[(i + 1) % len(region)]
            intersection = line_segment_intersection(p1_2d, p2_2d, p3, p4)

            if intersection:
                ix, iy = intersection
                seg_dx, seg_dy = p2_2d[0] - p1_2d[0], p2_2d[1] - p1_2d[1]

                if abs(seg_dx) > abs(seg_dy):
                    t = (ix - p1_2d[0]) / seg_dx if seg_dx != 0 else 0.0
                else:
                    t = (iy - p1_2d[1]) / seg_dy if seg_dy != 0 else 0.0
                cut_points_t.add(max(0.0, min(1.0, t)))

    return sorted(list(cut_points_t))


def is_point_in_rect(
    point: Tuple[float, float], rect: Tuple[float, float, float, float]
) -> bool:
    """Checks if a 2D point is inside a rectangle."""
    x, y = point
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2


def rect_a_contains_rect_b(
    rect_a: Tuple[float, float, float, float],
    rect_b: Tuple[float, float, float, float],
) -> bool:
    """Checks if rect_a fully contains rect_b."""
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    return bx1 >= ax1 and by1 >= ay1 and bx2 <= ax2 and by2 <= ay2


def line_segment_intersects_rect(
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    rect: Tuple[float, float, float, float],
) -> bool:
    """Checks if a line segment intersects a rectangle."""
    from . import clipping

    # Use the robust Cohen-Sutherland clipping algorithm.
    # If the algorithm returns a clipped segment, it means there was an
    # intersection.
    # The algorithm expects 3D points, so we add a dummy Z coordinate.
    start_3d = (p1[0], p1[1], 0.0)
    end_3d = (p2[0], p2[1], 0.0)
    return clipping.clip_line_segment(start_3d, end_3d, rect) is not None


def arc_intersects_rect(
    start_pos: Tuple[float, float],
    end_pos: Tuple[float, float],
    center: Tuple[float, float],
    clockwise: bool,
    rect: Tuple[float, float, float, float],
) -> bool:
    """Checks if an arc intersects with a rectangle."""

    # A mock command object for linearize_arc
    class MockArcCmd:
        def __init__(self, end, center_offset, clockwise):
            self.end = (end[0], end[1], 0.0)
            self.center_offset = center_offset
            self.clockwise = clockwise

    # Broad phase: Check if arc's AABB intersects rect's AABB
    arc_box = get_arc_bounding_box(
        start_pos,
        end_pos,
        (center[0] - start_pos[0], center[1] - start_pos[1]),
        clockwise,
    )
    if (
        arc_box[2] < rect[0]  # arc_xmax < rect_xmin
        or arc_box[0] > rect[2]  # arc_xmin > rect_xmax
        or arc_box[3] < rect[1]  # arc_ymax < rect_ymin
        or arc_box[1] > rect[3]  # arc_ymin > rect_ymax
    ):
        return False

    # Detailed phase: linearize the arc and check each segment.
    mock_cmd = MockArcCmd(
        end_pos,
        (center[0] - start_pos[0], center[1] - start_pos[1]),
        clockwise,
    )
    start_3d = (start_pos[0], start_pos[1], 0.0)
    radius = math.hypot(start_pos[0] - center[0], start_pos[1] - center[1])
    # Use a sensible resolution for selection hit-testing
    segments = linearize_arc(mock_cmd, start_3d, resolution=radius * 0.1)

    for p1_3d, p2_3d in segments:
        if line_segment_intersects_rect(p1_3d[:2], p2_3d[:2], rect):
            return True

    return False


def circle_is_contained_by_rect(
    center: Tuple[float, float],
    radius: float,
    rect: Tuple[float, float, float, float],
) -> bool:
    """Checks if a circle is fully contained within a rectangle."""
    cx, cy = center
    rx1, ry1, rx2, ry2 = rect
    return (
        (cx - radius) >= rx1
        and (cy - radius) >= ry1
        and (cx + radius) <= rx2
        and (cy + radius) <= ry2
    )


def circle_intersects_rect(
    center: Tuple[float, float],
    radius: float,
    rect: Tuple[float, float, float, float],
) -> bool:
    """Checks if a circle's boundary intersects with a rectangle."""
    cx, cy = center
    rx1, ry1, rx2, ry2 = rect

    # Quick rejection: if circle is fully contained, it doesn't intersect
    # the boundary.
    if circle_is_contained_by_rect(center, radius, rect):
        return False

    # 1. Check for overlap (closest point on rect to center is within radius)
    closest_x = max(rx1, min(cx, rx2))
    closest_y = max(ry1, min(cy, ry2))
    dist_sq_closest = (closest_x - cx) ** 2 + (closest_y - cy) ** 2
    if dist_sq_closest > radius * radius:
        return False  # No overlap at all

    # 2. If overlapping, check that the rect is not fully contained
    # within the circle, which would mean it doesn't touch the boundary.
    dx_far = max(abs(rx1 - cx), abs(rx2 - cx))
    dy_far = max(abs(ry1 - cy), abs(ry2 - cy))
    dist_sq_farthest = dx_far**2 + dy_far**2
    if dist_sq_farthest < radius * radius:
        return False  # Rect is entirely inside circle, not touching boundary

    # If it overlaps but is not fully contained by either shape, it must
    # intersect the boundary.
    return True
