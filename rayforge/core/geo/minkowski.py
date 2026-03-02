"""
Minkowski sum utilities for polygon operations.

These functions implement Minkowski sum/difference operations used
in packing and nesting algorithms.
"""

from typing import List
from .polygon import IntPoint, IntPolygon, Polygon


def convolve_two_segments(
    a1: IntPoint,
    a2: IntPoint,
    b1: IntPoint,
    b2: IntPoint,
) -> IntPolygon:
    """
    Convolve two line segments to produce a parallelogram.

    Returns the 4 vertices of the resulting parallelogram.
    Order matches boost::polygon convolve_two_segments.

    Args:
        a1: First point of segment A
        a2: Second point of segment A
        b1: First point of segment B
        b2: Second point of segment B

    Returns:
        List of 4 vertices forming the parallelogram.
    """
    return [
        (a1[0] + b2[0], a1[1] + b2[1]),
        (a1[0] + b1[0], a1[1] + b1[1]),
        (a2[0] + b1[0], a2[1] + b1[1]),
        (a2[0] + b2[0], a2[1] + b2[1]),
    ]


def convolve_point_sequences(
    path_a: IntPolygon, path_b: IntPolygon
) -> List[IntPolygon]:
    """
    Convolve two point sequences (polygon outlines).

    Returns a list of parallelograms formed by convolving each edge pair.
    Matches boost::polygon convolve_two_point_sequences logic.

    Args:
        path_a: First polygon path
        path_b: Second polygon path

    Returns:
        List of parallelogram polygons.
    """
    if len(path_a) < 2 or len(path_b) < 2:
        return []

    result = []
    len_a = len(path_a)
    len_b = len(path_b)

    for i in range(len_a):
        a1 = path_a[i]
        a2 = path_a[(i + 1) % len_a]

        for j in range(len_b):
            b1 = path_b[j]
            b2 = path_b[(j + 1) % len_b]

            parallelogram = convolve_two_segments(a1, a2, b1, b2)
            result.append(parallelogram)

    return result


def calculate_input_scale(
    polygons: List[Polygon],
    max_int: int = 2147483647,
) -> float:
    """
    Calculate appropriate input scale based on geometry bounds.

    This determines a scale factor that keeps coordinates within
    integer limits when converted for clipper operations.

    Args:
        polygons: List of polygons to consider for bounds calculation
        max_int: Maximum integer value to scale to (default: 32-bit signed max)

    Returns:
        Appropriate scale factor for clipper conversion.
    """
    if not polygons:
        return 0.1 * max_int

    max_x = 0.0
    min_x = 0.0
    max_y = 0.0
    min_y = 0.0

    for polygon in polygons:
        for p in polygon:
            max_x = max(max_x, float(p[0]))
            min_x = min(min_x, float(p[0]))
            max_y = max(max_y, float(p[1]))
            min_y = min(min_y, float(p[1]))

    max_abs_x = max(abs(max_x), abs(min_x))
    max_abs_y = max(abs(max_y), abs(min_y))
    max_abs = max(max_abs_x, max_abs_y)

    if max_abs < 1:
        max_abs = 1

    return (0.1 * max_int) / max_abs
