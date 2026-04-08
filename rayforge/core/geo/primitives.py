import numpy as np
from typing import List, Tuple, Optional
from .linearize import linearize_bezier_from_array
from .types import Point, Point3D, Polygon, Rect


def midpoint(a: Point3D, b: Point3D) -> Point3D:
    """Returns the midpoint between two 3D points."""
    return (
        (a[0] + b[0]) / 2,
        (a[1] + b[1]) / 2,
        (a[2] + b[2]) / 2,
    )


def is_point_on_segment(pt: Point, p1: Point, p2: Point) -> bool:
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


def is_point_in_polygon(point: Point, polygon: Polygon) -> bool:
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
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Optional[Point]:
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
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Optional[Point]:
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
    p1: Point, p2: Point, x: float, y: float
) -> Point:
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
    p1: Point, p2: Point, x: float, y: float
) -> Tuple[float, Point, float]:
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


def find_closest_point_on_bezier(
    bezier_row: np.ndarray,
    start_pos: Point3D,
    x: float,
    y: float,
) -> Optional[Tuple[float, Point, float]]:
    """
    Finds the closest point on a Bézier curve by linearizing it.
    Uses a fine resolution (0.005) for accurate closest-point detection.
    """
    bezier_segments = linearize_bezier_from_array(bezier_row, start_pos, 0.005)
    if not bezier_segments:
        return None

    min_dist_sq_sub = float("inf")
    best_sub_result = None

    for j, (p1_3d, p2_3d) in enumerate(bezier_segments):
        t_sub, pt_sub, dist_sq_sub = find_closest_point_on_line_segment(
            p1_3d[:2], p2_3d[:2], x, y
        )
        if dist_sq_sub < min_dist_sq_sub:
            min_dist_sq_sub = dist_sq_sub
            best_sub_result = (j, t_sub, pt_sub, dist_sq_sub)

    if not best_sub_result:
        return None

    j_best, t_sub_best, pt_best, dist_sq_best = best_sub_result
    # Approximate t for the whole curve from the linearized segment
    t_bezier = (j_best + t_sub_best) / len(bezier_segments)
    return t_bezier, pt_best, dist_sq_best


def get_segment_region_intersections(
    p1_2d: Point,
    p2_2d: Point,
    regions: List[Polygon],
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


def is_point_in_rect(point: Point, rect: Rect) -> bool:
    """Checks if a 2D point is inside a rectangle."""
    x, y = point
    rx1, ry1, rx2, ry2 = rect
    return rx1 <= x <= rx2 and ry1 <= y <= ry2


def rect_a_contains_rect_b(
    rect_a: Rect,
    rect_b: Rect,
) -> bool:
    """Checks if rect_a fully contains rect_b."""
    ax1, ay1, ax2, ay2 = rect_a
    bx1, by1, bx2, by2 = rect_b
    return bx1 >= ax1 and by1 >= ay1 and bx2 <= ax2 and by2 <= ay2


def line_segment_intersects_rect(
    p1: Point,
    p2: Point,
    rect: Rect,
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
