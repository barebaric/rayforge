"""Arc shape queries — bounds, angles, linearisation, and containment."""

from typing import List, Optional, Tuple, TypeAlias, Union

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo import Point, Point3D

_Segment3D = Tuple[Point3D, Point3D]
_PolygonsInput: TypeAlias = Union[
    List[List[Tuple[float, float]]],
    List[NDArray[np.float64]],
]


def get_arc_bounds(
    start: Point,
    end: Point,
    center: Point,
    clockwise: bool,
) -> Tuple[float, float, float, float]:
    """Compute the 2D bounding box of a circular arc.

    Args:
        start: Arc start point ``(x, y)``.
        end: Arc end point ``(x, y)``.
        center: Arc center ``(x, y)``.
        clockwise: ``True`` for clockwise winding.

    Returns:
        ``(x_min, y_min, x_max, y_max)`` bounding box.
    """
    ...


def get_arc_direction(
    center: Point,
    start: Point,
    mouse: Point,
) -> bool:
    """Determine the arc direction from a center, start, and mouse point.

    Useful for interactive arc drawing tools.

    Args:
        center: Arc center ``(x, y)``.
        start: Arc start ``(x, y)``.
        mouse: Current mouse/cursor position ``(x, y)``.

    Returns:
        ``True`` for clockwise, ``False`` for counter-clockwise.
    """
    ...


def get_arc_closest_point(
    arc_cmd: Union[List[float], object],
    start_pos: Point3D,
    x: float,
    y: float,
) -> Optional[Tuple[float, Point, float]]:
    """Find the closest point on an arc to a query position.

    Args:
        arc_cmd: An arc command row (8 floats) or an object with
            ``end``, ``center_offset``, and ``clockwise`` attributes.
        start_pos: Arc start ``(x, y, z)``.
        x: Query x.
        y: Query y.

    Returns:
        ``(distance, (px, py), t)`` or ``None``.

    Raises:
        TypeError: If *arc_cmd* is neither a list nor a suitable object.
    """
    ...


def get_arc_midpoint(
    start: Point,
    end: Point,
    center: Point,
    clockwise: bool,
) -> Point:
    """Compute the midpoint of a circular arc.

    Args:
        start: Arc start ``(x, y)``.
        end: Arc end ``(x, y)``.
        center: Arc center ``(x, y)``.
        clockwise: ``True`` for clockwise.

    Returns:
        The midpoint ``(x, y)`` on the arc.
    """
    ...


def get_arc_angles(
    start: Point,
    end: Point,
    center: Point,
    clockwise: bool,
) -> Tuple[float, float, float]:
    """Compute start angle, end angle, and sweep of an arc.

    Args:
        start: Arc start ``(x, y)``.
        end: Arc end ``(x, y)``.
        center: Arc center ``(x, y)``.
        clockwise: ``True`` for clockwise.

    Returns:
        ``(start_angle, end_angle, sweep)`` in radians.
    """
    ...


def does_arc_intersect_rect(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    rect: Tuple[float, float, float, float],
) -> bool:
    """Check whether an arc intersects an axis-aligned rectangle.

    Args:
        arc_start: Arc start ``(x, y)``.
        arc_end: Arc end ``(x, y)``.
        arc_center: Arc center ``(x, y)``.
        clockwise: ``True`` for clockwise.
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        ``True`` if the arc intersects or is contained by the rectangle.
    """
    ...


def does_arc_intersect_circle(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    circle_center: Point,
    circle_radius: float,
) -> bool:
    """Check whether an arc intersects a circle.

    Args:
        arc_start: Arc start ``(x, y)``.
        arc_end: Arc end ``(x, y)``.
        arc_center: Arc center ``(x, y)``.
        clockwise: ``True`` for clockwise.
        circle_center: Circle center ``(x, y)``.
        circle_radius: Circle radius.

    Returns:
        ``True`` if the arc and circle intersect.
    """
    ...


def is_arc_clockwise(
    points: List[Tuple[float, ...]],
    center: Tuple[float, ...],
) -> bool:
    """Determine the winding direction of an arc from sample points.

    Args:
        points: Sample points on the arc (2D or 3D; z ignored).
        center: Arc center (2D or 3D; z ignored).

    Returns:
        ``True`` if the arc winds clockwise.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def is_arc_inside_polygons(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    polygons: _PolygonsInput,
) -> bool:
    """Check whether an arc lies entirely inside a set of polygons.

    Args:
        arc_start: Arc start ``(x, y)``.
        arc_end: Arc end ``(x, y)``.
        arc_center: Arc center ``(x, y)``.
        clockwise: ``True`` for clockwise.
        polygons: Polygons (each a list of 2D/3D points or an ``(N, 2)``
            NumPy array; z ignored).

    Returns:
        ``True`` if every point on the arc is inside a polygon.

    Raises:
        ValueError: If a polygon point is not a 2- or 3-tuple.
        TypeError: If polygons are neither lists nor NumPy arrays.
    """
    ...


def is_angle_between(
    angle: float,
    start: float,
    end: float,
    clockwise: bool,
) -> bool:
    """Check whether *angle* lies within the arc sweep from *start* to
    *end*.

    Args:
        angle: Angle to test (radians).
        start: Arc start angle (radians).
        end: Arc end angle (radians).
        clockwise: ``True`` for clockwise sweep.

    Returns:
        ``True`` if *angle* is within the sweep.
    """
    ...


def normalize_angle(angle: float) -> float:
    """Normalize *angle* into the range ``[0, 2π)``.

    Args:
        angle: Angle in radians (any value).

    Returns:
        Equivalent angle in ``[0, 2π)``.
    """
    ...


def linearize_arc(
    arc_cmd: Union[List[float], object],
    start_point: Point3D,
    resolution: float = ...,
) -> List[_Segment3D]:
    """Convert an arc into line segments.

    Args:
        arc_cmd: An arc command row (8 floats) or an object with
            ``end``, ``center_offset``, and ``clockwise`` attributes.
        start_point: Arc start ``(x, y, z)``.
        resolution: Maximum segment length.  Defaults to ``0.1``.

    Returns:
        List of line segments ``((x1, y1, z1), (x2, y2, z2))``.

    Raises:
        TypeError: If *arc_cmd* cannot be parsed.
    """
    ...


def linearize_arc_from_array(
    data: List[float],
    start_point: Point3D,
    max_seg_length: float,
) -> List[List[float]]:
    """Linearise an arc from a raw 8-element command row.

    Args:
        data: Command row values (up to 8 floats).
        start_point: Arc start ``(x, y, z)``.
        max_seg_length: Maximum segment length.

    Returns:
        List of 6-element lists ``[x1, y1, z1, x2, y2, z2]``.
    """
    ...
