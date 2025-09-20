from typing import List, Tuple, Optional
from .primitives import is_point_in_polygon, get_segment_region_intersections


# Cohen-Sutherland outcodes
_INSIDE = 0  # 0000
_LEFT = 1  # 0001
_RIGHT = 2  # 0010
_BOTTOM = 4  # 0100
_TOP = 8  # 1000


def _compute_outcode(
    x: float, y: float, rect: Tuple[float, float, float, float]
) -> int:
    x_min, y_min, x_max, y_max = rect
    code = _INSIDE
    if x < x_min:
        code |= _LEFT
    elif x > x_max:
        code |= _RIGHT
    if y < y_min:
        code |= _BOTTOM
    elif y > y_max:
        code |= _TOP
    return code


def clip_line_segment(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    rect: Tuple[float, float, float, float],
) -> Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Clips a 3D line segment to an axis-aligned 2D rectangle using the
    Cohen-Sutherland algorithm. Z-coordinates are interpolated.
    """
    x_min, y_min, x_max, y_max = rect
    (x1, y1, z1), (x2, y2, z2) = p1, p2
    outcode1 = _compute_outcode(x1, y1, rect)
    outcode2 = _compute_outcode(x2, y2, rect)
    dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

    while True:
        if not (outcode1 | outcode2):  # Trivial accept
            return (x1, y1, z1), (x2, y2, z2)
        if outcode1 & outcode2:  # Trivial reject
            return None

        outcode_out = outcode1 if outcode1 else outcode2
        x, y, z = 0.0, 0.0, 0.0

        if outcode_out & _TOP:
            y = y_max
            x = x1 + dx * (y_max - y1) / dy if dy != 0 else x1
            z = z1 + dz * (y_max - y1) / dy if dy != 0 else z1
        elif outcode_out & _BOTTOM:
            y = y_min
            x = x1 + dx * (y_min - y1) / dy if dy != 0 else x1
            z = z1 + dz * (y_min - y1) / dy if dy != 0 else z1
        elif outcode_out & _RIGHT:
            x = x_max
            y = y1 + dy * (x_max - x1) / dx if dx != 0 else y1
            z = z1 + dz * (x_max - x1) / dx if dx != 0 else z1
        elif outcode_out & _LEFT:
            x = x_min
            y = y1 + dy * (x_min - x1) / dx if dx != 0 else y1
            z = z1 + dz * (x_min - x1) / dx if dx != 0 else z1

        if outcode_out == outcode1:
            x1, y1, z1 = x, y, z
            outcode1 = _compute_outcode(x1, y1, rect)
        else:
            x2, y2, z2 = x, y, z
            outcode2 = _compute_outcode(x2, y2, rect)


def subtract_regions_from_line_segment(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
    regions: List[List[Tuple[float, float]]],
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Calculates the sub-segments of a line that lie outside a list of polygons.
    """
    kept_segments: List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = []
    sorted_cuts = get_segment_region_intersections(p1[:2], p2[:2], regions)

    for i in range(len(sorted_cuts) - 1):
        t1, t2 = sorted_cuts[i], sorted_cuts[i + 1]
        if abs(t1 - t2) < 1e-9:
            continue

        # Check if the midpoint of this sub-segment is inside any region
        mid_t = (t1 + t2) / 2.0
        mid_p = (
            p1[0] + (p2[0] - p1[0]) * mid_t,
            p1[1] + (p2[1] - p1[1]) * mid_t,
        )

        is_inside_any_region = any(
            is_point_in_polygon(mid_p, r) for r in regions
        )

        if not is_inside_any_region:
            # This sub-segment is kept. Interpolate its 3D coordinates.
            sub_p1: Tuple[float, float, float] = (
                p1[0] + (p2[0] - p1[0]) * t1,
                p1[1] + (p2[1] - p1[1]) * t1,
                p1[2] + (p2[2] - p1[2]) * t1,
            )
            sub_p2: Tuple[float, float, float] = (
                p1[0] + (p2[0] - p1[0]) * t2,
                p1[1] + (p2[1] - p1[1]) * t2,
                p1[2] + (p2[2] - p1[2]) * t2,
            )
            kept_segments.append((sub_p1, sub_p2))

    return kept_segments
