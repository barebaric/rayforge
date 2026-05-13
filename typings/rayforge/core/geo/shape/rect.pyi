"""Rectangle shape queries — point containment and overlap tests."""

from typing import Tuple

from rayforge.core.geo import Point

_Rect = Tuple[float, float, float, float]


def is_point_inside_rect(
    point: Point,
    rect: _Rect,
) -> bool:
    """Check whether a point is inside an axis-aligned rectangle.

    Args:
        point: Query point ``(x, y)``.
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``True`` if the point is inside (or on the boundary of) the
        rectangle.
    """
    ...


def does_rect_contain_rect(
    outer: _Rect,
    inner: _Rect,
) -> bool:
    """Check whether *outer* fully contains *inner*.

    Args:
        outer: Outer rectangle ``(x_min, y_min, x_max, y_max)``.
        inner: Inner rectangle ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``True`` if every edge of *inner* is inside *outer*.
    """
    ...


def does_rect_intersect_rect(
    r1: _Rect,
    r2: _Rect,
) -> bool:
    """Check whether two axis-aligned rectangles overlap.

    Touching at an edge or corner counts as an intersection.

    Args:
        r1: First rectangle ``(x_min, y_min, x_max, y_max)``.
        r2: Second rectangle ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``True`` if the rectangles share at least one point.
    """
    ...
