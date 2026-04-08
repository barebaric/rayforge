import math
from typing import List, Optional

from .types import Point, Rect
from .primitives import find_closest_point_on_line_segment


def circle_circle_intersection(
    c1: Point, r1: float, c2: Point, r2: float
) -> List[Point]:
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


def project_point_onto_circle(
    point: Point, center: Point, radius: float
) -> Optional[Point]:
    """
    Projects a point onto the circumference of a circle.

    Args:
        point: The point to project (x, y).
        center: The center of the circle (x, y).
        radius: The radius of the circle.

    Returns:
        The (x, y) coordinates of the projected point on the circle's
        circumference, or None if the point is at the center.
    """
    dx = point[0] - center[0]
    dy = point[1] - center[1]
    dist = math.hypot(dx, dy)

    if dist < 1e-9:
        return None

    scale = radius / dist
    return (center[0] + dx * scale, center[1] + dy * scale)


def circle_is_contained_by_rect(
    center: Point,
    radius: float,
    rect: Rect,
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
    center: Point,
    radius: float,
    rect: Rect,
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


def line_segment_intersects_circle(
    p1: Point, p2: Point, center: Point, radius: float
) -> bool:
    """
    Checks if a line segment enters a circle (intersects boundary or is
    entirely inside).
    """
    _, _, dist_sq = find_closest_point_on_line_segment(
        p1, p2, center[0], center[1]
    )
    return dist_sq <= radius * radius
