import math
from typing import List, Tuple, Any
import numpy as np
from .constants import (
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
    GEO_ARRAY_COLS,
)


def _linearize_arc_from_array(
    arc_row: np.ndarray,
    start_point: Tuple[float, float, float],
    resolution: float = 0.1,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """Internal, NumPy-native implementation for arc linearization."""
    segments: List[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ] = []
    p0 = start_point
    p1 = (arc_row[COL_X], arc_row[COL_Y], arc_row[COL_Z])
    center_offset = (arc_row[COL_I], arc_row[COL_J])
    clockwise = bool(arc_row[COL_CW])
    z0, z1 = p0[2], p1[2]

    center = (
        p0[0] + center_offset[0],
        p0[1] + center_offset[1],
    )

    radius_start = math.dist(p0[:2], center)
    radius_end = math.dist(p1[:2], center)

    # If the start point is the center, it's just a line to the end.
    if radius_start < 1e-9:
        return [(p0, p1)]

    start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
    end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
    angle_range = end_angle - start_angle
    if clockwise:
        if angle_range > 0:
            angle_range -= 2 * math.pi
    else:
        if angle_range < 0:
            angle_range += 2 * math.pi

    # Use the average radius to get a better estimate for arc length
    avg_radius = (radius_start + radius_end) / 2
    arc_len = abs(angle_range * avg_radius)
    num_segments = max(2, int(arc_len / resolution))

    prev_pt = p0
    for i in range(1, num_segments + 1):
        t = i / num_segments
        radius = radius_start + (radius_end - radius_start) * t
        angle = start_angle + angle_range * t
        z = z0 + (z1 - z0) * t
        next_pt = (
            center[0] + radius * math.cos(angle),
            center[1] + radius * math.sin(angle),
            z,
        )
        segments.append((prev_pt, next_pt))
        prev_pt = next_pt
    return segments


def linearize_arc(
    arc_input: Any,
    start_point: Tuple[float, float, float],
    resolution: float = 0.1,
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
    """
    Converts an arc into a list of line segments.
    This function is backward-compatible and accepts either a NumPy array row
    or an object with .end, .center_offset, and .clockwise attributes.
    """
    if isinstance(arc_input, np.ndarray):
        # Fast path for new NumPy-based code
        return _linearize_arc_from_array(arc_input, start_point, resolution)
    else:
        # Backward-compatibility path for legacy objects (e.g., ArcToCommand)
        # Create a temporary NumPy row from the object's attributes.
        temp_row = np.zeros(GEO_ARRAY_COLS, dtype=np.float64)
        temp_row[COL_TYPE] = CMD_TYPE_ARC
        if hasattr(arc_input, "end") and arc_input.end is not None:
            temp_row[COL_X] = arc_input.end[0]
            temp_row[COL_Y] = arc_input.end[1]
            temp_row[COL_Z] = arc_input.end[2]
        if hasattr(arc_input, "center_offset"):
            temp_row[COL_I] = arc_input.center_offset[0]
            temp_row[COL_J] = arc_input.center_offset[1]
        if hasattr(arc_input, "clockwise"):
            temp_row[COL_CW] = 1.0 if arc_input.clockwise else 0.0
        return _linearize_arc_from_array(temp_row, start_point, resolution)


def linearize_bezier(
    p0: Tuple[float, ...],
    c1: Tuple[float, ...],
    c2: Tuple[float, ...],
    p1: Tuple[float, ...],
    num_steps: int,
) -> List[Tuple[Tuple[float, ...], Tuple[float, ...]]]:
    """
    Converts a cubic Bézier curve into a list of line segments.
    This function is generic and supports points of any dimension (e.g., 2D
    or 3D).

    Args:
        p0: The starting point of the curve.
        c1: The first control point.
        c2: The second control point.
        p1: The ending point of the curve.
        num_steps: The number of line segments to approximate the curve with.

    Returns:
        A list of tuples, where each tuple is a line segment represented by
        (start_point, end_point).
    """
    if num_steps < 1:
        return []

    points_np = np.array([p0, c1, c2, p1])
    t_values = np.linspace(0, 1, num_steps + 1)

    # Evaluate the cubic Bézier formula for each value of t
    interpolated_points_np = np.array(
        [
            (1 - t) ** 3 * points_np[0]
            + 3 * (1 - t) ** 2 * t * points_np[1]
            + 3 * (1 - t) * t**2 * points_np[2]
            + t**3 * points_np[3]
            for t in t_values
        ]
    )
    # Convert back to a list of tuples
    interpolated_points = [tuple(p) for p in interpolated_points_np]

    # Create segments from the list of points
    return [
        (interpolated_points[i], interpolated_points[i + 1])
        for i in range(num_steps)
    ]


def linearize_bezier_adaptive(
    p0: Tuple[float, float],
    c1: Tuple[float, float],
    c2: Tuple[float, float],
    p1: Tuple[float, float],
    tolerance_sq: float,
    max_depth: int = 10,
) -> List[Tuple[float, float]]:
    """
    Recursively flattens a cubic Bezier curve based on geometric error.

    Args:
        p0, c1, c2, p1: 2D control points (x, y).
        tolerance_sq: The squared maximum allowable distance error.
        max_depth: Maximum recursion depth to prevent infinite loops on
                   singularities/cusps. 10 = max 1024 segments.

    Returns:
        A list of points (excluding p0, including p1) that approximate the
        curve.
    """
    points: List[Tuple[float, float]] = []

    def recursive_step(
        p0: Tuple[float, float],
        c1: Tuple[float, float],
        c2: Tuple[float, float],
        p1: Tuple[float, float],
        depth: int,
    ):
        # 1. Determine if the curve is flat enough to be a line.
        # We check the distance of control points c1 and c2 from the
        # baseline p0-p1. If both are within tolerance, we stop.

        # Vector from p0 to p1
        vx, vy = p1[0] - p0[0], p1[1] - p0[1]
        norm_sq = vx * vx + vy * vy

        is_flat = False

        if depth >= max_depth:
            is_flat = True
        elif norm_sq < 1e-9:
            # Endpoints are the same, check distance of controls to point p0
            d1_sq = (c1[0] - p0[0]) ** 2 + (c1[1] - p0[1]) ** 2
            d2_sq = (c2[0] - p0[0]) ** 2 + (c2[1] - p0[1]) ** 2
            if d1_sq < tolerance_sq and d2_sq < tolerance_sq:
                is_flat = True
        else:
            # Perpendicular distance from C to line P0-P1
            # dist = abs((y2-y1)x0 - (x2-x1)y0 + x2y1 - y2x1) / sqrt(...)
            # We compare squared distances to avoid sqrt
            # dist_sq = ((y2-y1)x0 - (x2-x1)y0 + cross_base)^2 / norm_sq

            # Precalc terms for the line equation
            term1 = -vy  # -(y1-y0)
            term2 = vx  # (x1-x0)
            # constant = cross product p0 x p1 = x0y1 - y0x1
            term3 = p0[0] * p1[1] - p0[1] * p1[0]

            # Distance for C1
            cross1 = abs(term1 * c1[0] + term2 * c1[1] - term3)
            # Distance for C2
            cross2 = abs(term1 * c2[0] + term2 * c2[1] - term3)

            # We compare squared distances to avoid sqrt
            # dist_sq = cross^2 / norm_sq
            # condition:
            #   dist_sq < tolerance_sq  => cross^2 < tolerance_sq * norm_sq
            limit = tolerance_sq * norm_sq
            if (cross1 * cross1) < limit and (cross2 * cross2) < limit:
                is_flat = True

        if is_flat:
            return

        # 2. If not flat, split using De Casteljau's algorithm
        # Midpoints of edges
        m01 = ((p0[0] + c1[0]) / 2, (p0[1] + c1[1]) / 2)
        m12 = ((c1[0] + c2[0]) / 2, (c1[1] + c2[1]) / 2)
        m23 = ((c2[0] + p1[0]) / 2, (c2[1] + p1[1]) / 2)

        # Midpoints of midpoints
        q01 = ((m01[0] + m12[0]) / 2, (m01[1] + m12[1]) / 2)
        q12 = ((m12[0] + m23[0]) / 2, (m12[1] + m23[1]) / 2)

        # Final midpoint on the curve
        r = ((q01[0] + q12[0]) / 2, (q01[1] + q12[1]) / 2)

        # Recurse Left
        recursive_step(p0, m01, q01, r, depth + 1)
        # Add the split point
        points.append(r)
        # Recurse Right
        recursive_step(r, q12, m23, p1, depth + 1)

    recursive_step(p0, c1, c2, p1, 0)
    points.append(p1)
    return points


def resample_polyline(
    points: List[Tuple[float, float, float]],
    max_segment_length: float,
    is_closed: bool,
) -> List[Tuple[float, float, float]]:
    """
    Resamples a polyline, adding points to increase its density such that
    no segment is longer than `max_segment_length`.
    """
    if not points:
        return []

    new_points = [points[0]]
    num_segments = len(points) if is_closed else len(points) - 1

    for i in range(num_segments):
        p1 = points[i]
        p2 = points[(i + 1) % len(points)]  # Wraps for closed paths
        dist = math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        if dist > max_segment_length:
            # If a segment is too long, subdivide it.
            num_sub = math.ceil(dist / max_segment_length)
            for j in range(1, int(num_sub)):
                t = j / num_sub
                # Linear interpolation to create new points.
                px = p1[0] * (1 - t) + p2[0] * t
                py = p1[1] * (1 - t) + p2[1] * t
                new_points.append((px, py, p1[2]))

        # Add the original endpoint, avoiding duplication for closed paths.
        if not (is_closed and i == num_segments - 1):
            new_points.append(p2)

    return new_points
