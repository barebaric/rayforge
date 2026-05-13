"""Circle shape queries — intersections, containment, and projection."""

from typing import List, Optional

from rayforge.core.geo import Point

_Rect = tuple  # (float, float, float, float)


def get_circle_circle_intersections(
    c1: Point,
    r1: float,
    c2: Point,
    r2: float,
) -> List[Point]:
    """Compute intersection points of two circles.

    Args:
        c1: Center of the first circle ``(x, y)``.
        r1: Radius of the first circle.
        c2: Center of the second circle ``(x, y)``.
        r2: Radius of the second circle.

    Returns:
        A list of ``(x, y)`` intersection points (0, 1, or 2 points).
    """
    ...


def is_circle_inside_rect(
    center: Point,
    radius: float,
    rect: tuple,
) -> bool:
    """Check whether a circle is entirely inside a rectangle.

    Args:
        center: Circle center ``(x, y)``.
        radius: Circle radius.
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``True`` if the circle is fully contained.
    """
    ...


def does_circle_intersect_rect(
    center: Point,
    radius: float,
    rect: tuple,
) -> bool:
    """Check whether a circle intersects or touches a rectangle.

    Args:
        center: Circle center ``(x, y)``.
        radius: Circle radius.
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``True`` if the circle and rectangle share at least one point.
    """
    ...


def line_segment_intersects_circle(
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


def project_point_onto_circle(
    point: Point,
    center: Point,
    radius: float,
) -> Optional[Point]:
    """Project a point onto the circumference of a circle.

    Args:
        point: Point to project ``(x, y)``.
        center: Circle center ``(x, y)``.
        radius: Circle radius.

    Returns:
        The closest point ``(x, y)`` on the circle, or ``None`` if
        *point* coincides with *center*.
    """
    ...
