"""
Minkowski sum utilities for polygon operations.

These functions implement Minkowski sum/difference operations used
in packing and nesting algorithms.
"""

from typing import List
from .polygon import (
    to_clipper,
    from_clipper,
    polygon_bounds,
    convex_hull,
    CLIPPER_SCALE,
)
from .types import IntPolygon, Point, Polygon


def calculate_input_scale(
    polygons: List[Polygon],
    max_int: int = 2147483647,
) -> float:
    """
    Calculate appropriate input scale based on geometry bounds.
    """
    if not polygons:
        return 0.1 * max_int

    max_abs = 0.0
    for polygon in polygons:
        for p in polygon:
            max_abs = max(max_abs, abs(float(p[0])), abs(float(p[1])))

    if max_abs < 1:
        max_abs = 1

    return (0.1 * max_int) / max_abs


def convolve_two_segments(
    a1: Point,
    a2: Point,
    b1: Point,
    b2: Point,
) -> Polygon:
    """
    Creates a parallelogram from two line segments.
    Matches the exact logic in minkowski.cc convolve_two_segments.
    """
    return [
        (a1[0] + b2[0], a1[1] + b2[1]),
        (a1[0] + b1[0], a1[1] + b1[1]),
        (a2[0] + b1[0], a2[1] + b1[1]),
        (a2[0] + b2[0], a2[1] + b2[1]),
    ]


def convolve_point_sequences(
    seq_a: IntPolygon, seq_b: IntPolygon
) -> List[IntPolygon]:
    """
    Calculates the convolution of two point sequences (polygons) by generating
    parallelograms from all pairs of edges. This is a core part of the
    Minkowski sum for general polygons.
    """
    parallelograms = []
    if not seq_a or len(seq_a) < 2 or not seq_b or len(seq_b) < 2:
        return parallelograms

    for i in range(len(seq_a)):
        p_a1 = seq_a[i - 1]
        p_a2 = seq_a[i]

        for j in range(len(seq_b)):
            p_b1 = seq_b[j - 1]
            p_b2 = seq_b[j]

            parallelograms.append(
                convolve_two_segments(p_a1, p_a2, p_b1, p_b2)
            )

    return parallelograms


def minkowski_sum_convex(
    poly_a: IntPolygon, poly_b: IntPolygon
) -> List[IntPolygon]:
    """
    Calculates the Minkowski Sum for two CONVEX polygons.
    Sum(A, B) = ConvexHull({a + b | a in Vertices(A), b in Vertices(B)})
    """
    if not poly_a or not poly_b:
        return []

    all_points = []
    for p1 in poly_a:
        for p2 in poly_b:
            all_points.append((p1[0] + p2[0], p1[1] + p2[1]))

    hull = convex_hull(all_points)
    return [[(int(p[0]), int(p[1])) for p in hull]] if len(hull) >= 3 else []


def calculate_nfp(
    stationary: Polygon,
    orbiting: Polygon,
    scale: int = CLIPPER_SCALE,
) -> List[Polygon]:
    """
    Calculate the No-Fit Polygon (NFP) for two polygons.
    Assumes polygons are convex for performance.
    """
    if not stationary or not orbiting:
        return []

    static_path = to_clipper(stationary, scale)
    orbiting_path = to_clipper(orbiting, scale)

    # Negate orbiting polygon (reflect through origin)
    orbiting_negated = [(-p[0], -p[1]) for p in orbiting_path]

    # NFP = MinkowskiSum(stationary, -orbiting)
    nfp_paths = minkowski_sum_convex(static_path, orbiting_negated)

    results = []
    for path in nfp_paths:
        start_x = orbiting_path[0][0]
        start_y = orbiting_path[0][1]
        shifted = [(p[0] + start_x, p[1] + start_y) for p in path]
        results.append(from_clipper(shifted, scale))

    return results


def calculate_ifp(
    container: Polygon,
    part: Polygon,
    scale: int = CLIPPER_SCALE,
) -> List[Polygon]:
    """
    Calculate the Inner-Fit Polygon (IFP) using a simple formula based on
    bounding boxes, which is exact for axis-aligned rectangles and a robust
    approximation for other convex shapes.
    """
    if not container or not part:
        return []

    c_min_x, c_min_y, c_max_x, c_max_y = polygon_bounds(container)
    p_min_x, p_min_y, p_max_x, p_max_y = polygon_bounds(part)

    p_width = p_max_x - p_min_x
    p_height = p_max_y - p_min_y
    c_width = c_max_x - c_min_x
    c_height = c_max_y - c_min_y

    # Check if part is larger than container in either dimension
    if p_width > c_width + 1e-9 or p_height > c_height + 1e-9:
        return []

    # This formula gives the valid locus for the part's reference point
    # (part[0])
    ifp_min_x = c_min_x - p_min_x
    ifp_max_x = c_max_x - p_max_x
    ifp_min_y = c_min_y - p_min_y
    ifp_max_y = c_max_y - p_max_y

    # If the resulting IFP has no area, return empty
    if ifp_min_x > ifp_max_x or ifp_min_y > ifp_max_y:
        return []

    ifp = [
        (ifp_min_x, ifp_min_y),
        (ifp_max_x, ifp_min_y),
        (ifp_max_x, ifp_max_y),
        (ifp_min_x, ifp_max_y),
    ]

    return [ifp]
