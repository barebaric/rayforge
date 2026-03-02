"""
Polygon utilities using pyclipper for boolean operations.

This module provides basic polygon manipulation functions that operate
on simple polygon representations (lists of (x, y) tuples).
"""

import math
from typing import List, Tuple, Optional, TYPE_CHECKING
import pyclipper

if TYPE_CHECKING:
    from .geometry import Rect

Point = Tuple[float, float]
Polygon = List[Point]
IntPoint = Tuple[int, int]
IntPolygon = List[IntPoint]

CLIPPER_SCALE = int(1e7)


def almost_equal(a: float, b: float, tolerance: float = 1e-9) -> bool:
    """Check if two floats are approximately equal."""
    return abs(a - b) < tolerance


def polygon_area(polygon: Polygon) -> float:
    """
    Calculate the signed area of a polygon using the shoelace formula.

    Args:
        polygon: List of (x, y) tuples representing the polygon vertices.

    Returns:
        Signed area (positive for CCW, negative for CW).
    """
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return area / 2.0


def polygon_bounds(polygon: Polygon) -> "Rect":
    """
    Get the bounding box of a polygon.

    Args:
        polygon: List of (x, y) tuples.

    Returns:
        Tuple of (min_x, min_y, max_x, max_y).
    """
    if not polygon:
        return 0.0, 0.0, 0.0, 0.0
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def int_polygon_bounds(polygon: IntPolygon) -> Tuple[int, int, int, int]:
    """
    Calculate bounds of an integer polygon.

    Args:
        polygon: List of integer (x, y) tuples.

    Returns:
        Tuple of (min_x, min_y, max_x, max_y).
    """
    if not polygon:
        return 0, 0, 0, 0
    min_x = min(p[0] for p in polygon)
    max_x = max(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_y = max(p[1] for p in polygon)
    return min_x, min_y, max_x, max_y


def polygon_group_bounds(polygons: List[Polygon]) -> "Rect":
    """
    Get the bounding box of multiple polygons.

    Args:
        polygons: List of polygons.

    Returns:
        Tuple of (min_x, min_y, max_x, max_y).
    """
    if not polygons:
        return 0.0, 0.0, 0.0, 0.0

    all_points = [p for poly in polygons for p in poly]
    if not all_points:
        return 0.0, 0.0, 0.0, 0.0

    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)
    max_x = max(p[0] for p in all_points)
    max_y = max(p[1] for p in all_points)

    return min_x, min_y, max_x, max_y


def translate_bounds(bounds: "Rect", dx: float, dy: float) -> "Rect":
    """
    Translate a bounding box by a given offset.

    Args:
        bounds: Bounding box as (min_x, min_y, max_x, max_y).
        dx: X offset.
        dy: Y offset.

    Returns:
        Translated bounding box as (min_x, min_y, max_x, max_y).
    """
    return (bounds[0] + dx, bounds[1] + dy, bounds[2] + dx, bounds[3] + dy)


def normalize_polygons(
    polygons: List[Polygon],
) -> Tuple[List[Polygon], float, float]:
    """
    Normalize polygons so their minimum corner is at the origin.

    Args:
        polygons: List of polygons to normalize.

    Returns:
        Tuple of (normalized polygons, original min_x, original min_y).
        Returns (input, 0.0, 0.0) if polygons are empty or have no points.
    """
    if not polygons:
        return polygons, 0.0, 0.0

    all_points = [p for poly in polygons for p in poly]
    if not all_points:
        return polygons, 0.0, 0.0

    min_x = min(p[0] for p in all_points)
    min_y = min(p[1] for p in all_points)

    normalized = translate_polygons(polygons, -min_x, -min_y)
    return normalized, min_x, min_y


def polygon_centroid(polygon: Polygon) -> Point:
    """
    Calculate the centroid of a polygon.

    Args:
        polygon: List of (x, y) tuples.

    Returns:
        The centroid as (x, y) tuple.
    """
    if not polygon:
        return (0.0, 0.0)

    cx, cy = 0.0, 0.0
    signed_area = 0.0

    for i in range(len(polygon)):
        j = (i + 1) % len(polygon)
        cross = polygon[i][0] * polygon[j][1] - polygon[j][0] * polygon[i][1]
        signed_area += cross
        cx += (polygon[i][0] + polygon[j][0]) * cross
        cy += (polygon[i][1] + polygon[j][1]) * cross

    signed_area /= 2.0
    if abs(signed_area) < 1e-9:
        n = len(polygon)
        return (
            sum(p[0] for p in polygon) / n,
            sum(p[1] for p in polygon) / n,
        )

    cx /= 6.0 * signed_area
    cy /= 6.0 * signed_area
    return (cx, cy)


def rotate_polygon(polygon: Polygon, angle_degrees: float) -> Polygon:
    """
    Rotate a polygon around the origin.

    Args:
        polygon: List of (x, y) tuples.
        angle_degrees: Rotation angle in degrees (positive = CCW).

    Returns:
        Rotated polygon.
    """
    if len(polygon) < 3:
        return polygon

    angle_rad = math.radians(angle_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    return [(x * cos_a - y * sin_a, x * sin_a + y * cos_a) for x, y in polygon]


def translate_polygon(polygon: Polygon, dx: float, dy: float) -> Polygon:
    """
    Translate a polygon by a given offset.

    Args:
        polygon: List of (x, y) tuples.
        dx: X offset.
        dy: Y offset.

    Returns:
        Translated polygon.
    """
    return [(p[0] + dx, p[1] + dy) for p in polygon]


def rotate_polygons(
    polygons: List[Polygon], angle_degrees: float
) -> List[Polygon]:
    """
    Rotate multiple polygons around the origin.

    Args:
        polygons: List of polygons to rotate.
        angle_degrees: Rotation angle in degrees (positive = CCW).

    Returns:
        List of rotated polygons.
    """
    return [rotate_polygon(poly, angle_degrees) for poly in polygons]


def translate_polygons(
    polygons: List[Polygon], dx: float, dy: float
) -> List[Polygon]:
    """
    Translate multiple polygons by a given offset.

    Args:
        polygons: List of polygons to translate.
        dx: X offset.
        dy: Y offset.

    Returns:
        List of translated polygons.
    """
    return [translate_polygon(poly, dx, dy) for poly in polygons]


def scale_polygon(
    polygon: Polygon, sx: float, sy: Optional[float] = None
) -> Polygon:
    """
    Scale a polygon.

    Args:
        polygon: List of (x, y) tuples.
        sx: X scale factor (or uniform scale if sy not provided).
        sy: Y scale factor (optional, defaults to sx).

    Returns:
        Scaled polygon.
    """
    if sy is None:
        sy = sx
    return [(p[0] * sx, p[1] * sy) for p in polygon]


def convex_hull(polygon: Polygon) -> Polygon:
    """
    Compute the convex hull of a polygon using Andrew's monotone chain.

    Args:
        polygon: List of (x, y) tuples.

    Returns:
        Convex hull as a list of (x, y) tuples.
    """
    if len(polygon) < 3:
        return polygon

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    sorted_pts = sorted(polygon)

    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def to_clipper(polygon: Polygon, scale: int = CLIPPER_SCALE) -> IntPolygon:
    """Convert polygon to clipper integer coordinates."""
    return [(int(p[0] * scale), int(p[1] * scale)) for p in polygon]


def from_clipper(polygon: IntPolygon, scale: int = CLIPPER_SCALE) -> Polygon:
    """Convert clipper integer coordinates to polygon."""
    return [(p[0] / scale, p[1] / scale) for p in polygon]


def clean_polygon(
    polygon: Polygon, tolerance: float = 0.01
) -> Optional[Polygon]:
    """
    Clean a polygon by removing duplicate and near-collinear points.

    Args:
        polygon: List of (x, y) tuples.
        tolerance: Cleaning tolerance.

    Returns:
        Cleaned polygon, or None if result is degenerate.
    """
    if not polygon or len(polygon) < 3:
        return None

    clip_poly = to_clipper(polygon)

    simple = pyclipper.SimplifyPolygon(clip_poly, pyclipper.PFT_NONZERO)

    if not simple:
        return None

    biggest = simple[0]
    biggest_area = abs(pyclipper.Area(biggest))
    for i in range(1, len(simple)):
        area = abs(pyclipper.Area(simple[i]))
        if area > biggest_area:
            biggest = simple[i]
            biggest_area = area

    clean_tol = int(tolerance * CLIPPER_SCALE)
    clean = pyclipper.CleanPolygon(biggest, clean_tol)

    if not clean or len(clean) < 3:
        return None

    result = from_clipper(clean)

    if (
        len(result) > 1
        and almost_equal(result[0][0], result[-1][0])
        and almost_equal(result[0][1], result[-1][1])
    ):
        result.pop()

    return result


def polygon_offset(polygon: Polygon, offset: float) -> List[Polygon]:
    """
    Offset (inflate/deflate) a polygon.

    Args:
        polygon: List of (x, y) tuples.
        offset: Offset distance (positive = expand, negative = shrink).

    Returns:
        List of offset polygons.
    """
    if not polygon or len(polygon) < 3:
        return []

    if almost_equal(offset, 0):
        return [polygon]

    clip_poly = to_clipper(polygon)

    pco = pyclipper.PyclipperOffset()
    pco.AddPath(clip_poly, pyclipper.JT_MITER, pyclipper.ET_CLOSEDPOLYGON)

    scaled_offset = offset * CLIPPER_SCALE
    solution = pco.Execute(scaled_offset)

    result = []
    for path in solution:
        if len(path) >= 3:
            result.append(from_clipper(path))

    return result


def polygon_union(polygons: List[Polygon]) -> List[Polygon]:
    """
    Compute the union of multiple polygons.

    Args:
        polygons: List of polygons.

    Returns:
        List of resulting polygons.
    """
    if not polygons:
        return []

    clipper = pyclipper.Pyclipper()
    for poly in polygons:
        if poly and len(poly) >= 3:
            clipper.AddPath(to_clipper(poly), pyclipper.PT_SUBJECT, True)

    solution = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO)

    return [from_clipper(path) for path in solution if len(path) >= 3]


def polygon_intersection(poly1: Polygon, poly2: Polygon) -> List[Polygon]:
    """
    Compute the intersection of two polygons.

    Args:
        poly1: First polygon.
        poly2: Second polygon.

    Returns:
        List of resulting polygons.
    """
    if not poly1 or not poly2 or len(poly1) < 3 or len(poly2) < 3:
        return []

    clipper = pyclipper.Pyclipper()
    clipper.AddPath(to_clipper(poly1), pyclipper.PT_SUBJECT, True)
    clipper.AddPath(to_clipper(poly2), pyclipper.PT_CLIP, True)

    solution = clipper.Execute(
        pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO
    )

    return [from_clipper(path) for path in solution if len(path) >= 3]


def polygon_difference(poly1: Polygon, poly2: Polygon) -> List[Polygon]:
    """
    Compute the difference of two polygons (poly1 - poly2).

    Args:
        poly1: First polygon (subject).
        poly2: Second polygon (clip).

    Returns:
        List of resulting polygons.
    """
    if not poly1 or not poly2 or len(poly1) < 3 or len(poly2) < 3:
        return [poly1] if poly1 else []

    clipper = pyclipper.Pyclipper()
    clipper.AddPath(to_clipper(poly1), pyclipper.PT_SUBJECT, True)
    clipper.AddPath(to_clipper(poly2), pyclipper.PT_CLIP, True)

    solution = clipper.Execute(pyclipper.CT_DIFFERENCE, pyclipper.PFT_NONZERO)

    return [from_clipper(path) for path in solution if len(path) >= 3]


def point_in_polygon(
    point: Point,
    polygon: Polygon,
    scale: int = CLIPPER_SCALE,
) -> bool:
    """
    Check if a point is inside a polygon using pyclipper.

    Args:
        point: The (x, y) point to test.
        polygon: List of (x, y) tuples.
        scale: Scale factor for clipper integer conversion.

    Returns:
        True if point is inside the polygon.
    """
    if not polygon or len(polygon) < 3:
        return False

    clip_poly = to_clipper(polygon, scale)
    pt = (int(point[0] * scale), int(point[1] * scale))

    return pyclipper.PointInPolygon(pt, clip_poly) != 0


def polygons_intersect(
    poly1: Polygon,
    poly2: Polygon,
    min_area: float = 0.0,
) -> bool:
    """
    Check if two polygons intersect.

    Args:
        poly1: First polygon.
        poly2: Second polygon.
        min_area: Minimum intersection area threshold (in clipper integer
            coordinates). If > 0, only intersections with area > min_area
            are considered valid. Default is 0 (any intersection).

    Returns:
        True if polygons intersect (with area > min_area if specified).
    """
    if not poly1 or not poly2 or len(poly1) < 3 or len(poly2) < 3:
        return False

    clipper = pyclipper.Pyclipper()
    clipper.AddPath(to_clipper(poly1), pyclipper.PT_SUBJECT, True)
    clipper.AddPath(to_clipper(poly2), pyclipper.PT_CLIP, True)

    result = clipper.Execute(
        pyclipper.CT_INTERSECTION,
        pyclipper.PFT_NONZERO,
        pyclipper.PFT_NONZERO,
    )

    if not result:
        return False

    if min_area <= 0:
        return True

    for path in result:
        if abs(pyclipper.Area(path)) > min_area:
            return True
    return False
