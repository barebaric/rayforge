"""Cubic Bezier curve operations — evaluation, splitting, bounding,
linearisation, and containment."""

from typing import List, Optional, Tuple, TypeAlias, Union

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo import Point, Point3D

_CubicBezier = Tuple[Point, Point, Point, Point]
_BezierSplit = Tuple[_CubicBezier, _CubicBezier]
_PolygonsInput: TypeAlias = Union[
    List[List[Tuple[float, float]]],
    List[NDArray[np.float64]],
]


def get_bezier_point_at(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
    t: float,
) -> Point:
    """Evaluate a cubic Bezier at parameter *t*.

    Only the ``x`` and ``y`` coordinates of the control points are
    used.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).
        t: Parameter in ``[0, 1]``.

    Returns:
        Point ``(x, y)`` on the curve.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def split_bezier(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
    t: float,
) -> _BezierSplit:
    """Split a cubic Bezier at parameter *t* using de Casteljau's
    algorithm.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).
        t: Split parameter in ``[0, 1]``.

    Returns:
        ``(first_half, second_half)`` where each half is a
        ``((p0, c1, c2, p1))`` tuple of ``(x, y)`` control points.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def get_bezier_bounds(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
) -> Tuple[float, float, float, float]:
    """Compute the 2D bounding box of a cubic Bezier curve.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).

    Returns:
        ``(x_min, y_min, x_max, y_max)``.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def get_bezier_rect_intersections(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
    rect: Tuple[float, float, float, float],
) -> List[float]:
    """Find parameter values where a Bezier intersects a rectangle.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        List of ``t`` values in ``[0, 1]`` at intersection points.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def clip_bezier_with_rect(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
    rect: Tuple[float, float, float, float],
) -> List[_CubicBezier]:
    """Clip a cubic Bezier curve to a rectangle.

    Returns one or more Bezier curves representing the portions of the
    original curve that lie inside the rectangle.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        List of clipped Bezier curves ``((p0, c1, c2, p1))``.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def convert_cubic_bezier_to_quadratic(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
) -> Tuple[Point, Point, Point]:
    """Approximate a cubic Bezier with a quadratic Bezier.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).

    Returns:
        ``(start, control, end)`` of the quadratic approximation.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def is_bezier_inside_polygons(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
    polygons: _PolygonsInput,
) -> bool:
    """Check whether a cubic Bezier lies entirely inside polygons.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).
        polygons: Polygons (each a list of 2D/3D points or an ``(N, 2)``
            NumPy array; z ignored).

    Returns:
        ``True`` if every point on the curve is inside a polygon.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
        TypeError: If polygons are neither lists nor NumPy arrays.
    """
    ...


def linearize_bezier(
    p0: Point3D,
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
    num_steps: int,
) -> List[Tuple[Point3D, Point3D]]:
    """Uniformly linearise a cubic Bezier into *num_steps* line segments.

    Args:
        p0: Start point ``(x, y, z)``.
        p1: First control point ``(x, y, z)``.
        p2: Second control point ``(x, y, z)``.
        p3: End point ``(x, y, z)``.
        num_steps: Number of uniform subdivisions.

    Returns:
        List of line segments ``((x1, y1, z1), (x2, y2, z2))``.
    """
    ...


def linearize_bezier_adaptive(
    p0: Union[Tuple[float, float], Tuple[float, float, float]],
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
    p3: Union[Tuple[float, float], Tuple[float, float, float]],
    tolerance_sq: float,
    max_subdivisions: int = ...,
) -> List[Point]:
    """Adaptively linearise a 2D cubic Bezier.

    Recursively subdivides until the flatness criterion or
    *max_subdivisions* is met.

    Args:
        p0: Start point (2D or 3D; z ignored).
        p1: First control point (2D or 3D; z ignored).
        p2: Second control point (2D or 3D; z ignored).
        p3: End point (2D or 3D; z ignored).
        tolerance_sq: Maximum squared flatness (distance from chord).
        max_subdivisions: Recursion depth limit.  Defaults to ``20``.

    Returns:
        List of ``(x, y)`` points along the linearised curve.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def linearize_bezier_from_array(
    bezier_row: List[float],
    start_point: Point3D,
    max_seg_length: float,
) -> List[List[float]]:
    """Linearise a Bezier from a raw 8-element command row.

    Args:
        bezier_row: Command row values (up to 8 floats).
        start_point: Bezier start ``(x, y, z)``.
        max_seg_length: Maximum segment length.

    Returns:
        List of 6-element lists ``[x1, y1, z1, x2, y2, z2]``.
    """
    ...


def linearize_bezier_segment(
    p0: Point3D,
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
    tolerance: float = ...,
) -> List[Point3D]:
    """Linearise a 3D Bezier segment with adaptive subdivision.

    Args:
        p0: Start point ``(x, y, z)``.
        p1: First control point ``(x, y, z)``.
        p2: Second control point ``(x, y, z)``.
        p3: End point ``(x, y, z)``.
        tolerance: Maximum deviation.  Defaults to ``0.1``.

    Returns:
        Ordered list of ``(x, y, z)`` points along the curve.
    """
    ...


def flatten_bezier(
    p0: Point3D,
    p1: Point3D,
    p2: Point3D,
    p3: Point3D,
    tolerance: float,
    max_subdivisions: int,
    pts: list,
) -> None:
    """Flatten a 3D Bezier by appending points to an existing list.

    Unlike :func:`linearize_bezier_segment`, this mutates *pts* in
    place rather than returning a new list.

    Args:
        p0: Start point ``(x, y, z)``.
        p1: First control point ``(x, y, z)``.
        p2: Second control point ``(x, y, z)``.
        p3: End point ``(x, y, z)``.
        tolerance: Maximum deviation.
        max_subdivisions: Recursion depth limit.
        pts: A Python list to append ``(x, y, z)`` points to.
    """
    ...


def bezier_flatness_sq(
    a: Union[Tuple[float, float], Tuple[float, float, float]],
    b: Union[Tuple[float, float], Tuple[float, float, float]],
    c: Union[Tuple[float, float], Tuple[float, float, float]],
    d: Union[Tuple[float, float], Tuple[float, float, float]],
) -> float:
    """Compute the squared flatness of a cubic Bezier.

    Flatness measures how close the curve is to a straight line.

    Args:
        a: Start point (2D or 3D; z ignored).
        b: First control point (2D or 3D; z ignored).
        c: Second control point (2D or 3D; z ignored).
        d: End point (2D or 3D; z ignored).

    Returns:
        Squared flatness value.  Zero means perfectly flat.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...


def perp_dist_sq(
    pt: Point3D,
    origin: Point3D,
    vx: float,
    vy: float,
    vz: float = ...,
    norm_sq: float = ...,
) -> float:
    """Compute the squared perpendicular distance from *pt* to a line
    defined by *origin* and direction ``(vx, vy, vz)``.

    Args:
        pt: Query point ``(x, y, z)``.
        origin: A point on the line ``(x, y, z)``.
        vx: Line direction x component.
        vy: Line direction y component.
        vz: Line direction z component.  Defaults to ``0.0``.
        norm_sq: Precomputed ``vx² + vy² + vz²``.  ``0.0`` triggers
            recalculation.  Defaults to ``0.0``.

    Returns:
        Squared perpendicular distance.
    """
    ...
