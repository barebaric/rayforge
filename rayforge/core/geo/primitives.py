import math
from typing import List, Tuple, Optional, Any
from .linearize import linearize_arc


def is_angle_between(
    target: float, start: float, end: float, clockwise: bool
) -> bool:
    """
    Checks if a target angle is within the sweep of an arc defined by start
    and end angles. Handles wrapping around 2*PI.
    """
    # Normalize all angles to be in the range [0, 2*PI)
    target = (target + 2 * math.pi) % (2 * math.pi)
    start = (start + 2 * math.pi) % (2 * math.pi)
    end = (end + 2 * math.pi) % (2 * math.pi)

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


def find_closest_point_on_line_segment(
    p1: Tuple[float, float], p2: Tuple[float, float], x: float, y: float
) -> Tuple[float, Tuple[float, float], float]:
    """Finds the closest point on a 2D line segment.

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
