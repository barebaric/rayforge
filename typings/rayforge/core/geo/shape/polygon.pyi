"""Polygon operations — area, bounds, Boolean ops, transforms, and
NumPy-accelerated variants."""

from typing import List, Optional, Tuple, TypeAlias, Union

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo import Point, Polygon

_ArrayLikePolygon: TypeAlias = Union[Polygon, NDArray[np.float64]]
_PolygonsInput: TypeAlias = Union[
    List[Polygon],
    List[NDArray[np.float64]],
]
"""A list of polygons — each polygon may be a list of ``(x, y)`` tuples
or an ``(N, 2)`` NumPy float64 array."""
_Rect = Tuple[float, float, float, float]

# -- pure-Python polygon functions -----------------------------------------

def clean_polygon(
    polygon: Polygon,
    tolerance: Optional[float] = ...,
) -> Optional[Polygon]:
    """Remove duplicate and near-duplicate vertices from a polygon.

    Args:
        polygon: Input polygon as a list of ``(x, y)`` vertices.
        tolerance: Minimum distance between consecutive vertices.
            Defaults to ``1e-6``.

    Returns:
        Cleaned polygon, or ``None`` if fewer than 3 vertices remain.
    """
    ...

def is_almost_equal(
    a: float,
    b: float,
    tolerance: Optional[float] = ...,
) -> bool:
    """Check whether two floats are equal within tolerance.

    Args:
        a: First value.
        b: Second value.
        tolerance: Maximum allowed difference.  Defaults to ``1e-9``.

    Returns:
        ``True`` if ``|a - b| < tolerance``.
    """
    ...

def normalize_polygons(
    polygons: _PolygonsInput,
) -> Tuple[List[Polygon], float, float]:
    """Normalize polygons by translating to the positive quadrant.

    Shifts all polygons so that the minimum x and y are near zero.

    Args:
        polygons: List of polygons (each a list of ``(x, y)`` tuples or
            an ``(N, 2)`` NumPy array).

    Returns:
        ``(normalized_polygons, min_x, min_y)`` — the shifted polygons
        and the original minimum coordinates.
    """
    ...

def translate_bounds(
    bounds: _Rect,
    dx: float,
    dy: float,
) -> _Rect:
    """Translate a bounding box by ``(dx, dy)``.

    Args:
        bounds: ``(x_min, y_min, x_max, y_max)``.
        dx: X offset.
        dy: Y offset.

    Returns:
        Shifted bounding box.
    """
    ...

def translate_polygons(
    polygons: _PolygonsInput,
    dx: float,
    dy: float,
) -> List[Polygon]:
    """Translate multiple polygons by ``(dx, dy)``.

    Args:
        polygons: List of polygons (each a list of ``(x, y)`` tuples or
            an ``(N, 2)`` NumPy array).
        dx: X offset.
        dy: Y offset.

    Returns:
        Translated polygons.
    """
    ...

def point_line_distance(
    point: Point,
    line_start: Point,
    line_end: Point,
) -> float:
    """Compute the perpendicular distance from a point to an infinite
    line.

    Args:
        point: Query point ``(x, y)``.
        line_start: A point on the line ``(x, y)``.
        line_end: Another point on the line ``(x, y)``.

    Returns:
        Perpendicular distance (always non-negative).
    """
    ...

def get_polygon_area(polygon: Polygon) -> float:
    """Compute the signed area of a polygon.

    Positive for CCW, negative for CW.

    Args:
        polygon: List of ``(x, y)`` vertices.

    Returns:
        Signed area.
    """
    ...

def get_polygon_signed_area(polygon: Polygon) -> float:
    """Compute the signed area of a polygon.

    Positive for CCW, negative for CW.

    Args:
        polygon: List of ``(x, y)`` vertices.

    Returns:
        Signed area.
    """
    ...

def get_polygon_perimeter(polygon: Polygon) -> float:
    """Compute the perimeter of a polygon.

    Args:
        polygon: List of ``(x, y)`` vertices.

    Returns:
        Perimeter length.
    """
    ...

def get_polygon_bounds(polygon: Polygon) -> _Rect:
    """Compute the axis-aligned bounding box.

    Args:
        polygon: List of ``(x, y)`` vertices.

    Returns:
        ``(x_min, y_min, x_max, y_max)``.
    """
    ...

def get_polygon_group_bounds(
    polygons: _PolygonsInput,
) -> _Rect:
    """Compute the bounding box enclosing multiple polygons.

    Args:
        polygons: List of polygons (each a list of ``(x, y)`` tuples or
            an ``(N, 2)`` NumPy array).

    Returns:
        ``(x_min, y_min, x_max, y_max)``.
    """
    ...

def get_polygon_centroid(polygon: Polygon) -> Point:
    """Compute the centroid of a polygon.

    Args:
        polygon: List of ``(x, y)`` vertices.

    Returns:
        Centroid ``(x, y)``.
    """
    ...

def is_polygon_convex(polygon: Polygon) -> bool:
    """Check whether a polygon is convex.

    Args:
        polygon: List of ``(x, y)`` vertices.

    Returns:
        ``True`` if the polygon is convex.
    """
    ...

def get_polygon_convex_hull(polygon: Polygon) -> Polygon:
    """Compute the convex hull of a polygon.

    Args:
        polygon: Input vertices ``(x, y)``.

    Returns:
        Convex hull vertices in CCW order.
    """
    ...

def get_polygon_edges(polygon: Polygon) -> List[Tuple[Point, Point]]:
    """Return the edges of a polygon as pairs of consecutive vertices.

    Args:
        polygon: List of ``(x, y)`` vertices.

    Returns:
        List of ``(start, end)`` edges.
    """
    ...

def is_point_inside_polygon(point: Point, polygon: Polygon) -> bool:
    """Check whether a point is inside a polygon using ray casting.

    Args:
        point: Query point ``(x, y)``.
        polygon: Polygon vertices ``(x, y)``.

    Returns:
        ``True`` if the point is inside (or on the boundary).
    """
    ...

def offset_polygon(
    polygon: Polygon,
    offset: float,
) -> List[Polygon]:
    """Offset (inflate/deflate) a polygon.

    Args:
        polygon: Input polygon.
        offset: Positive inflates, negative deflates.

    Returns:
        One or more offset polygons.
    """
    ...

def get_polygons_union(
    polygons: _PolygonsInput,
) -> List[Polygon]:
    """Compute the union of multiple polygons.

    Args:
        polygons: List of polygons (each a list of ``(x, y)`` tuples or
            an ``(N, 2)`` NumPy array).

    Returns:
        Result polygons after union.
    """
    ...

def get_polygons_intersection(
    poly1: Polygon,
    poly2: Polygon,
) -> List[Polygon]:
    """Compute the intersection of two polygons.

    Args:
        poly1: First polygon.
        poly2: Second polygon.

    Returns:
        Intersection polygons.
    """
    ...

def get_polygons_difference(
    poly1: Polygon,
    poly2: Polygon,
) -> List[Polygon]:
    """Compute the difference *poly1* \\ *poly2*.

    Args:
        poly1: Polygon to subtract from.
        poly2: Polygon to subtract.

    Returns:
        Difference polygons.
    """
    ...

def polygons_intersect(
    p1: Polygon,
    p2: Polygon,
    min_area: float = ...,
) -> bool:
    """Check whether two polygons intersect.

    Args:
        p1: First polygon.
        p2: Second polygon.
        min_area: Minimum overlap area to count as intersection.
            Defaults to ``0.0``.

    Returns:
        ``True`` if the polygons overlap by at least *min_area*.
    """
    ...

def flip_polygon(
    polygon: Polygon,
    flip_h: bool,
    flip_v: bool,
) -> Polygon:
    """Flip a polygon horizontally and/or vertically.

    Args:
        polygon: Input polygon.
        flip_h: ``True`` to negate x coordinates.
        flip_v: ``True`` to negate y coordinates.

    Returns:
        Flipped polygon.
    """
    ...

def flip_polygons(
    polygons: _PolygonsInput,
    flip_h: bool,
    flip_v: bool,
) -> List[Polygon]:
    """Flip multiple polygons.

    Args:
        polygons: List of polygons (each a list of ``(x, y)`` tuples or
            an ``(N, 2)`` NumPy array).
        flip_h: ``True`` to negate x coordinates.
        flip_v: ``True`` to negate y coordinates.

    Returns:
        Flipped polygons.
    """
    ...

def rotate_polygon(
    polygon: Polygon,
    angle: float,
) -> Polygon:
    """Rotate a polygon around the origin.

    Args:
        polygon: Input polygon.
        angle: Rotation angle in radians.

    Returns:
        Rotated polygon.
    """
    ...

def rotate_polygons(
    polygons: _PolygonsInput,
    angle: float,
) -> List[Polygon]:
    """Rotate multiple polygons around the origin.

    Args:
        polygons: List of polygons (each a list of ``(x, y)`` tuples or
            an ``(N, 2)`` NumPy array).
        angle: Rotation angle in radians.

    Returns:
        Rotated polygons.
    """
    ...

def scale_polygon(
    polygon: Polygon,
    scale: float,
    scale_y: Optional[float] = ...,
) -> Polygon:
    """Scale a polygon, optionally with separate x and y factors.

    Args:
        polygon: Input polygon.
        scale: Scale factor (applied to x; and to y if *scale_y* is
            ``None``).
        scale_y: Separate y scale factor.  ``None`` uses *scale*.

    Returns:
        Scaled polygon.
    """
    ...

def translate_polygon(
    polygon: Polygon,
    dx: float,
    dy: float,
) -> Polygon:
    """Translate a polygon by ``(dx, dy)``.

    Args:
        polygon: Input polygon.
        dx: X offset.
        dy: Y offset.

    Returns:
        Translated polygon.
    """
    ...

# -- NumPy-accelerated variants -------------------------------------------

def polygon_area_numpy(polygon: NDArray[np.float64]) -> float:
    """Compute signed area from a ``(N, 2)`` NumPy polygon.

    Args:
        polygon: ``(N, 2)`` float64 array of vertices.

    Returns:
        Signed area.
    """
    ...

def polygon_bounds_numpy(polygon: NDArray[np.float64]) -> _Rect:
    """Compute bounding box from a ``(N, 2)`` NumPy polygon.

    Args:
        polygon: ``(N, 2)`` float64 array of vertices.

    Returns:
        ``(x_min, y_min, x_max, y_max)``.
    """
    ...

def polygon_perimeter_numpy(polygon: NDArray[np.float64]) -> float:
    """Compute perimeter from a ``(N, 2)`` NumPy polygon.

    Args:
        polygon: ``(N, 2)`` float64 array of vertices.

    Returns:
        Perimeter length.
    """
    ...

def polygon_group_bounds_numpy(
    polygons: List[NDArray[np.float64]],
) -> _Rect:
    """Compute bounding box enclosing multiple NumPy polygons.

    Args:
        polygons: List of ``(N, 2)`` float64 arrays.

    Returns:
        ``(x_min, y_min, x_max, y_max)``.
    """
    ...

def flip_polygon_numpy(
    polygon: NDArray[np.float64],
    flip_h: bool,
    flip_v: bool,
) -> NDArray[np.float64]:
    """Flip a NumPy polygon.

    Args:
        polygon: ``(N, 2)`` float64 array.
        flip_h: ``True`` to negate x.
        flip_v: ``True`` to negate y.

    Returns:
        Flipped ``(N, 2)`` float64 array.
    """
    ...

def flip_polygons_numpy(
    polygons: list,
    flip_h: bool,
    flip_v: bool,
) -> list:
    """Flip multiple NumPy polygons.

    Args:
        polygons: List of ``(N, 2)`` float64 arrays.
        flip_h: ``True`` to negate x.
        flip_v: ``True`` to negate y.

    Returns:
        List of flipped ``(N, 2)`` float64 arrays.
    """
    ...

def normalize_polygons_numpy(
    polygons: List[NDArray[np.float64]],
) -> Tuple[List[NDArray[np.float64]], float, float]:
    """Normalize NumPy polygons to the positive quadrant.

    Args:
        polygons: List of ``(N, 2)`` float64 arrays.

    Returns:
        ``(normalized_arrays, min_x, min_y)``.
    """
    ...

def point_in_polygon_numpy(
    point: Point,
    polygon: NDArray[np.float64],
) -> bool:
    """Check whether a point is inside a NumPy polygon.

    Args:
        point: Query ``(x, y)``.
        polygon: ``(N, 2)`` float64 array.

    Returns:
        ``True`` if the point is inside.
    """
    ...

def polygons_intersect_numpy(
    poly1: NDArray[np.float64],
    poly2: NDArray[np.float64],
    min_area: float = ...,
) -> bool:
    """Check intersection of two NumPy polygons.

    Args:
        poly1: ``(N, 2)`` float64 array.
        poly2: ``(M, 2)`` float64 array.
        min_area: Minimum overlap area.  Defaults to ``0.0``.

    Returns:
        ``True`` if they intersect.
    """
    ...

def rotate_polygon_numpy(
    polygon: NDArray[np.float64],
    angle: float,
) -> NDArray[np.float64]:
    """Rotate a NumPy polygon around the origin.

    Args:
        polygon: ``(N, 2)`` float64 array.
        angle: Rotation angle in radians.

    Returns:
        Rotated ``(N, 2)`` float64 array.
    """
    ...

def rotate_polygons_numpy(
    polygons: List[NDArray[np.float64]],
    angle: float,
) -> List[NDArray[np.float64]]:
    """Rotate multiple NumPy polygons.

    Args:
        polygons: List of ``(N, 2)`` float64 arrays.
        angle: Rotation angle in radians.

    Returns:
        List of rotated arrays.
    """
    ...

def translate_polygon_numpy(
    polygon: NDArray[np.float64],
    dx: float,
    dy: float,
) -> NDArray[np.float64]:
    """Translate a NumPy polygon.

    Args:
        polygon: ``(N, 2)`` float64 array.
        dx: X offset.
        dy: Y offset.

    Returns:
        Translated ``(N, 2)`` float64 array.
    """
    ...

def translate_polygons_numpy(
    polygons: List[NDArray[np.float64]],
    dx: float,
    dy: float,
) -> List[NDArray[np.float64]]:
    """Translate multiple NumPy polygons.

    Args:
        polygons: List of ``(N, 2)`` float64 arrays.
        dx: X offset.
        dy: Y offset.

    Returns:
        List of translated arrays.
    """
    ...


def to_clipper_numpy(
    polygon: _ArrayLikePolygon,
    scale: int = ...,
) -> List[Tuple[int, int]]:
    """Convert a float polygon to integer Clipper format by scaling.

    Accepts either a list of ``(x, y)`` vertices or a ``(N, 2)`` NumPy
    array.

    Args:
        polygon: Input polygon vertices.
        scale: Scale factor.  Defaults to ``10_000_000``.

    Returns:
        A list of ``(x, y)`` integer vertices.

    Raises:
        TypeError: If *polygon* is neither a list of tuples nor a NumPy
            array.
    """
    ...
