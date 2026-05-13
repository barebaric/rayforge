"""Line and line-segment shape queries — intersections, closest points,
and distance computations."""

from typing import List, Optional, Tuple

from rayforge.core.geo import Point

_Rect = Tuple[float, float, float, float]


def get_line_line_intersection(
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Optional[Point]:
    """Find the intersection of two infinite lines.

    Each line is defined by two points.

    Args:
        p1: First point on line A ``(x, y)``.
        p2: Second point on line A ``(x, y)``.
        p3: First point on line B ``(x, y)``.
        p4: Second point on line B ``(x, y)``.

    Returns:
        Intersection ``(x, y)`` or ``None`` if the lines are parallel.
    """
    ...


def get_line_segment_intersection(
    p1: Point,
    p2: Point,
    p3: Point,
    p4: Point,
) -> Optional[Point]:
    """Find the intersection of two finite line segments.

    Args:
        p1: Start of segment A ``(x, y)``.
        p2: End of segment A ``(x, y)``.
        p3: Start of segment B ``(x, y)``.
        p4: End of segment B ``(x, y)``.

    Returns:
        Intersection ``(x, y)`` or ``None`` if the segments do not
        intersect.
    """
    ...


def get_line_closest_point(
    line_p1: Point,
    line_p2: Point,
    x: float,
    y: float,
) -> Point:
    """Find the closest point on an infinite line to a query point.

    Args:
        line_p1: First point on the line ``(x, y)``.
        line_p2: Second point on the line ``(x, y)``.
        x: Query x coordinate.
        y: Query y coordinate.

    Returns:
        Closest point ``(x, y)`` on the line.
    """
    ...


def get_line_segment_closest_point(
    seg_p1: Point,
    seg_p2: Point,
    x: float,
    y: float,
) -> Tuple[float, Point, float]:
    """Find the closest point on a line segment to a query point.

    Args:
        seg_p1: Segment start ``(x, y)``.
        seg_p2: Segment end ``(x, y)``.
        x: Query x coordinate.
        y: Query y coordinate.

    Returns:
        ``(distance, (px, py), t)`` — distance to the closest point,
        the point itself, and the parametric value ``t`` in ``[0, 1]``.
    """
    ...


def get_point_line_distance(
    point: Point,
    line_p1: Point,
    line_p2: Point,
) -> float:
    """Compute the perpendicular distance from a point to an infinite
    line.

    Args:
        point: Query point ``(x, y)``.
        line_p1: First point on the line ``(x, y)``.
        line_p2: Second point on the line ``(x, y)``.

    Returns:
        Non-negative distance.
    """
    ...


def is_point_on_line_segment(
    point: Point,
    seg_p1: Point,
    seg_p2: Point,
) -> bool:
    """Check whether a point lies on a line segment.

    Args:
        point: Query point ``(x, y)``.
        seg_p1: Segment start ``(x, y)``.
        seg_p2: Segment end ``(x, y)``.

    Returns:
        ``True`` if the point is on (or very near) the segment.
    """
    ...


def does_line_segment_intersect_rect(
    p1: Point,
    p2: Point,
    rect: _Rect,
) -> bool:
    """Check whether a line segment intersects a rectangle.

    Args:
        p1: Segment start ``(x, y)``.
        p2: Segment end ``(x, y)``.
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``True`` if the segment and rectangle share at least one point.
    """
    ...


def does_line_segment_intersect_circle(
    p1: Point,
    p2: Point,
    circle_center: Point,
    circle_radius: float,
) -> bool:
    """Check whether a line segment intersects a circle.

    Args:
        p1: Segment start ``(x, y)``.
        p2: Segment end ``(x, y)``.
        circle_center: Circle center ``(x, y)``.
        circle_radius: Circle radius.

    Returns:
        ``True`` if the segment and circle intersect.
    """
    ...


def get_line_segment_polygon_intersections(
    p1: Point,
    p2: Point,
    polygon: List[List[Point]],
) -> List[float]:
    """Find parametric intersection values of a segment with polygon
    edges.

    Args:
        p1: Segment start ``(x, y)``.
        p2: Segment end ``(x, y)``.
        polygon: List of polygons (each a list of ``(x, y)`` vertices).

    Returns:
        List of ``t`` values in ``[0, 1]`` at intersection points.
    """
    ...
