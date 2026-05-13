"""PyO3-based geometry engine for RayForge.

This module provides core geometric primitives and algorithms including
path construction, polygon operations, shape queries, and curve fitting.
It is implemented as a native Rust extension for performance.

Type Aliases:
    Point: A 2D point as ``(x, y)``.
    Point3D: A 3D point as ``(x, y, z)``.
    Rect: An axis-aligned bounding box as ``(x_min, y_min, x_max, y_max)``.
    Polygon: A 2D polygon as a list of :data:`Point` vertices.
    Polygon3D: A 3D polygon as a list of :data:`Point3D` vertices.
    IntPoint: An integer 2D point as ``(x, y)``.
    IntPolygon: An integer polygon as a list of :data:`IntPoint`.
    Edge: A line segment as a pair of :data:`Point` endpoints.
    CubicBezier: A cubic Bezier curve as four :data:`Point` control points.
    Point2DOr3D: A point that is either :data:`Point` or :data:`Point3D`.
"""

from typing import Tuple, List, Union, Optional, TypeAlias
from collections import namedtuple

from numpy.typing import NDArray
import numpy as np
from rayforge.core.geo.geometry import Geometry, _Point2DOr3D
from rayforge.core.geo.path import PyCommand

Point: TypeAlias = Tuple[float, float]
"""A 2D point represented as ``(x, y)``."""

Point3D: TypeAlias = Tuple[float, float, float]
"""A 3D point represented as ``(x, y, z)``."""

Rect: TypeAlias = Tuple[float, float, float, float]
"""An axis-aligned bounding box as ``(x_min, y_min, x_max, y_max)``."""

Polygon: TypeAlias = List[Tuple[float, float]]
"""A 2D polygon as an ordered list of :data:`Point` vertices."""

Polygon3D: TypeAlias = List[Tuple[float, float, float]]
"""A 3D polygon as an ordered list of :data:`Point3D` vertices."""

IntPoint: TypeAlias = Tuple[int, int]
"""An integer 2D point as ``(x, y)`` for grid-based operations."""

IntPolygon: TypeAlias = List[Tuple[int, int]]
"""An integer polygon as a list of :data:`IntPoint`."""

Edge: TypeAlias = Tuple[Tuple[float, float], Tuple[float, float]]
"""A line segment as a pair of :data:`Point` endpoints ``(start, end)``."""

CubicBezier: TypeAlias = Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
]
"""A cubic Bezier curve as four :data:`Point` control points ``(p0, c1, c2, p1)``."""

Point2DOr3D: TypeAlias = Union[Tuple[float, float], Tuple[float, float, float]]
"""A point that accepts either 2D ``(x, y)`` or 3D ``(x, y, z)``."""

_PolygonsInput: TypeAlias = Union[
    List[Polygon],
    List[NDArray[np.float64]],
]
"""A list of polygons — each polygon may be a list of ``(x, y)`` tuples
or an ``(N, 2)`` NumPy float64 array."""


class Rect3D(
    namedtuple(
        "Rect3D", ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
    )
):
    """A 3D axis-aligned bounding box with separate min/max for each axis.

    Attributes:
        x_min: Minimum x coordinate (left face).
        x_max: Maximum x coordinate (right face).
        y_min: Minimum y coordinate (bottom face).
        y_max: Maximum y coordinate (top face).
        z_min: Minimum z coordinate (front face).
        z_max: Maximum z coordinate (back face).
    """


CMD_TYPE_MOVE: int
"""Command type constant for a move-to operation."""

CMD_TYPE_LINE: int
"""Command type constant for a line-to operation."""

CMD_TYPE_ARC: int
"""Command type constant for an arc-to operation."""

CMD_TYPE_BEZIER: int
"""Command type constant for a cubic Bezier curve operation."""

COL_TYPE: int
"""Column index for the command type in the geometry data array."""

COL_X: int
"""Column index for the x coordinate."""

COL_Y: int
"""Column index for the y coordinate."""

COL_Z: int
"""Column index for the z coordinate."""

COL_I: int
"""Column index for the arc center x offset (i)."""

COL_J: int
"""Column index for the arc center y offset (j)."""

COL_CW: int
"""Column index for the arc clockwise flag."""

COL_C1X: int
"""Column index for the first Bezier control point x."""

COL_C1Y: int
"""Column index for the first Bezier control point y."""

COL_C2X: int
"""Column index for the second Bezier control point x."""

COL_C2Y: int
"""Column index for the second Bezier control point y."""

GEO_ARRAY_COLS: int
"""Total number of columns in the geometry data array."""

CLIPPER_SCALE: int
"""Default scale factor (10_000_000) for converting float polygons to
integer Clipper format."""


class constants:
    """Namespace holding geometry command and column-index constants.

    All attributes are integer values used for indexing into the
    :attr:`Geometry.data` NumPy array.

    Attributes:
        CMD_TYPE_MOVE: Command type for move-to.
        CMD_TYPE_LINE: Command type for line-to.
        CMD_TYPE_ARC: Command type for arc-to.
        CMD_TYPE_BEZIER: Command type for Bezier curve.
        COL_TYPE: Column index for command type.
        COL_X: Column index for x.
        COL_Y: Column index for y.
        COL_Z: Column index for z.
        COL_I: Column index for arc i offset.
        COL_J: Column index for arc j offset.
        COL_CW: Column index for arc clockwise flag.
        COL_C1X: Column index for Bezier control point 1 x.
        COL_C1Y: Column index for Bezier control point 1 y.
        COL_C2X: Column index for Bezier control point 2 x.
        COL_C2Y: Column index for Bezier control point 2 y.
        GEO_ARRAY_COLS: Number of columns in the data array.
    """

    CMD_TYPE_MOVE: int
    CMD_TYPE_LINE: int
    CMD_TYPE_ARC: int
    CMD_TYPE_BEZIER: int
    COL_TYPE: int
    COL_X: int
    COL_Y: int
    COL_Z: int
    COL_I: int
    COL_J: int
    COL_CW: int
    COL_C1X: int
    COL_C1Y: int
    COL_C2X: int
    COL_C2Y: int
    GEO_ARRAY_COLS: int


def clip_line_segment_with_polygons(
    p1: Point3D,
    p2: Point3D,
    regions: _PolygonsInput,
) -> List[Tuple[Point3D, Point3D]]:
    """Clip a 3D line segment against a set of 2D polygon regions.

    Returns only the portions of the segment that fall inside at least one
    of the given polygons.  The z coordinate of the endpoints is
    interpolated linearly along the original segment.

    Args:
        p1: Start point of the line segment as ``(x, y, z)``.
        p2: End point of the line segment as ``(x, y, z)``.
        regions: List of polygons, each a list of ``(x, y)`` vertices
            or an ``(N, 2)`` NumPy array defining a clipping region.

    Returns:
        A list of :data:`Segment3D` tuples ``(start, end)`` representing
        the visible portions of the original segment.
    """


def is_arc_inside_polygons(
    arc_start: Point,
    arc_end: Point,
    arc_center: Point,
    clockwise: bool,
    polygons: _PolygonsInput,
) -> bool:
    """Check whether an arc lies entirely inside a set of polygons.

    Args:
        arc_start: Start point of the arc as ``(x, y)``.
        arc_end: End point of the arc as ``(x, y)``.
        arc_center: Center point of the arc as ``(x, y)``.
        clockwise: Whether the arc runs clockwise.
        polygons: List of polygons, each a list of ``(x, y)`` vertices
            or an ``(N, 2)`` NumPy array.

    Returns:
        ``True`` if every point on the arc is inside at least one polygon.
    """


def is_bezier_inside_polygons(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    polygons: _PolygonsInput,
) -> bool:
    """Check whether a cubic Bezier curve lies entirely inside a set of
    polygons.

    Args:
        p0: Start point of the curve as ``(x, y)``.
        p1: First control point as ``(x, y)``.
        p2: Second control point as ``(x, y)``.
        p3: End point of the curve as ``(x, y)``.
        polygons: List of polygons, each a list of ``(x, y)`` vertices
            or an ``(N, 2)`` NumPy array.

    Returns:
        ``True`` if every point on the Bezier is inside at least one
        polygon.
    """


def fit_points_with_primitives(
    points: List[Point3D],
    tolerance: float,
) -> List[List[float]]:
    """Fit a sequence of 3D points to geometric primitives.

    Attempts to fit lines, arcs, and cubic Bezier curves to segments of
    the point sequence that best approximate the original path within the
    given tolerance.

    Args:
        points: Ordered list of 3D points to fit.
        tolerance: Maximum allowed deviation from the original points.

    Returns:
        A list of command rows, each an 8-element list of floats
        ``[type, x, y, z, aux1, aux2, aux3, aux4]``.
    """


def to_clipper(
    polygon: Union[Polygon, NDArray[np.float64]],
    scale: Optional[int] = ...,
) -> List[Tuple[int, int]]:
    """Convert a float polygon to integer Clipper format by scaling.

    Each vertex ``(x, y)`` is multiplied by *scale* and cast to ``int``.
    Accepts either a list of ``(x, y)`` tuples or an ``(N, 2)`` NumPy
    array.

    Args:
        polygon: A list of ``(x, y)`` float vertices or an ``(N, 2)``
            NumPy array.
        scale: Scale factor.  Defaults to :data:`CLIPPER_SCALE`
            (10_000_000).

    Returns:
        A list of ``(x, y)`` integer vertices.

    Raises:
        TypeError: If *polygon* is neither a list of tuples nor a NumPy
            array.
    """


def from_clipper(
    polygon: List[Tuple[int, int]],
    scale: Optional[int] = ...,
) -> Polygon:
    """Convert an integer Clipper polygon back to float format.

    Each vertex ``(x, y)`` is divided by *scale*.

    Args:
        polygon: A list of ``(x, y)`` integer vertices.
        scale: Scale factor.  Defaults to :data:`CLIPPER_SCALE`
            (10_000_000).

    Returns:
        A list of ``(x, y)`` float vertices.
    """
