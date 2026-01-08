import math
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares
from .constants import (
    CMD_TYPE_BEZIER,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
    COL_X,
    COL_Z,
    COL_TYPE,
)
from .primitives import get_arc_angles


def are_collinear(
    points: List[Tuple[float, ...]], tolerance: float = 0.01
) -> bool:
    """
    Check if all points in a list are colinear within a given tolerance by
    checking the perpendicular distance of each point to the line formed by
    the first and last points.
    """
    if len(points) < 3:
        return True

    p1, p2 = points[0], points[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_length = math.hypot(dx, dy)

    if line_length < 1e-9:
        # All points are effectively coincident with p1
        return all(
            math.hypot(p[0] - p1[0], p[1] - p1[1]) < tolerance for p in points
        )

    # Check perpendicular distance of each intermediate point to the line p1-p2
    for p in points[1:-1]:
        # Vector from p1 to p
        vx = p[0] - p1[0]
        vy = p[1] - p1[1]
        # Perpendicular distance = |(p-p1) x (p2-p1)| / |p2-p1|
        # In 2D, this is |vx*dy - vy*dx| / line_length
        dist = abs(vx * dy - vy * dx) / line_length
        if dist > tolerance:
            return False
    return True


def fit_circle_to_points(
    points: List[Tuple[float, ...]],
) -> Optional[Tuple[Tuple[float, float], float, float]]:
    """
    Fits a circle to a list of 2D points using the least squares method.

    Args:
        points: A list of (x, y) or (x, y, z) tuples. Only x and y are used.

    Returns:
        A tuple containing (center, radius, max_error) if a fit is possible,
        otherwise None. The center is (xc, yc), radius is a float, and
        max_error is the maximum deviation of any point from the fitted arc.
    """
    if len(points) < 3 or are_collinear(points):
        return None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Initial guess: mean center and average radius
    x0, y0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x - x0) ** 2 + (y - y0) ** 2))

    # Define the residual function for least squares
    def residuals(p):
        return np.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2) - p[2]

    # Fit circle using least squares
    try:
        result = least_squares(residuals, [x0, y0, r0], method="lm")
        xc, yc, r = result.x
        center = (xc, yc)
    except Exception:
        return None

    # Calculate max deviation of points from the fitted circle's circumference
    distances = np.sqrt((x - xc) ** 2 + (y - yc) ** 2)
    point_error = np.max(np.abs(distances - r))

    return center, r, point_error


def get_arc_to_polyline_deviation(
    points: List[Tuple[float, ...]], center: Tuple[float, float], radius: float
) -> float:
    """
    Computes the maximum deviation of a circular arc from the original
    polyline that it is approximating.

    This checks how far the arc strays from the original line segments, which
    is a critical check for arc fitting algorithms. It calculates the sagitta
    for each segment.
    """
    if len(points) < 2:
        return 0.0
    xc, yc = center
    max_deviation = 0.0

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        x1, y1 = p1[:2]
        x2, y2 = p2[:2]
        dx = x2 - x1
        dy = y2 - y1
        segment_length = math.hypot(dx, dy)

        if segment_length < 1e-9:
            distance = math.hypot(x1 - xc, y1 - yc)
            deviation = abs(distance - radius)
            max_deviation = max(max_deviation, deviation)
            continue

        # Distances from center to endpoints
        d1 = math.hypot(x1 - xc, y1 - yc)
        d2 = math.hypot(x2 - xc, y2 - yc)

        # If segment is longer than diameter, it can't be a chord.
        # The deviation is just the endpoint deviation.
        if segment_length > 2 * radius:
            deviation = max(abs(d1 - radius), abs(d2 - radius))
        else:
            # Vectors from center to points
            v1x, v1y = x1 - xc, y1 - yc
            v2x, v2y = x2 - xc, y2 - yc

            # Angle between vectors using dot product
            dot = v1x * v2x + v1y * v2y
            mag1 = math.hypot(v1x, v1y)
            mag2 = math.hypot(v2x, v2y)

            if mag1 < 1e-9 or mag2 < 1e-9:
                deviation = (
                    abs(d1 - radius) if mag1 < 1e-9 else abs(d2 - radius)
                )
            else:
                # Clamp to avoid domain errors with acos
                cos_theta = min(1.0, max(-1.0, dot / (mag1 * mag2)))
                theta = math.acos(cos_theta)
                # Sagitta is the max distance from chord to arc
                sagitta = radius * (1 - math.cos(theta / 2.0))
                # Also consider if endpoints are not on the circle
                endpoint_dev = max(abs(d1 - radius), abs(d2 - radius))
                deviation = max(sagitta, endpoint_dev)

        max_deviation = max(max_deviation, deviation)
    return max_deviation


def convert_arc_to_beziers_from_array(
    start_point: Tuple[float, float, float],
    end_point: Tuple[float, float, float],
    center_offset: Tuple[float, float],
    clockwise: bool,
) -> List[np.ndarray]:
    """
    Approximates a circular arc with one or more cubic Bézier curves.

    An arc is split into segments of at most 90 degrees, as a single cubic
    Bézier can represent this with high precision. Z coordinates are linearly
    interpolated along the path.

    Args:
        start_point: The 3D start point (x, y, z) of the arc.
        end_point: The 3D end point (x, y, z) of the arc.
        center_offset: The 2D vector (i, j) from the start point to the
                       arc's center.
        clockwise: The direction of the arc.

    Returns:
        A list of numpy arrays, where each array is a row representing a
        single CMD_TYPE_BEZIER command. Returns an empty list for zero-length
        arcs.
    """
    p0_2d = start_point[:2]
    p_end_2d = end_point[:2]
    z_start, z_end = start_point[2], end_point[2]

    center = (p0_2d[0] + center_offset[0], p0_2d[1] + center_offset[1])
    radius = math.hypot(center_offset[0], center_offset[1])
    radius_end = math.hypot(p_end_2d[0] - center[0], p_end_2d[1] - center[1])

    if radius < 1e-9:
        return []  # Cannot create an arc with zero radius.

    # Strict check for full circles (coincident start/end)
    is_coincident = math.isclose(
        start_point[0], end_point[0], abs_tol=1e-12
    ) and math.isclose(start_point[1], end_point[1], abs_tol=1e-12)

    if is_coincident:
        # Standard convention: coincident points on a non-zero radius arc
        # define a full circle.
        total_sweep = -2 * math.pi if clockwise else 2 * math.pi
        start_angle = math.atan2(p0_2d[1] - center[1], p0_2d[0] - center[0])
    else:
        start_angle, _, total_sweep = get_arc_angles(
            p0_2d, p_end_2d, center, clockwise
        )

    # Threshold for treating an arc as zero-length noise.
    # 1e-8 radians is approx 0.00000057 degrees.
    if abs(total_sweep) < 1e-8:
        return []

    # Determine number of segments (max 90 degrees per segment)
    num_segments = max(1, math.ceil(abs(total_sweep) / (math.pi / 2)))
    segment_sweep = total_sweep / num_segments
    kappa = (4.0 / 3.0) * math.tan(abs(segment_sweep) / 4.0)

    bezier_rows: List[np.ndarray] = []
    current_p0 = np.array(start_point)

    for i in range(num_segments):
        angle1 = start_angle + (i + 1) * segment_sweep

        # The end point of the last segment must be the original end_point.
        if i == num_segments - 1:
            current_p3 = np.array(end_point)
        else:
            # Interpolate radius for spirals
            t1 = (i + 1) / num_segments
            radius1_interp = radius + t1 * (radius_end - radius)
            p3_x = center[0] + radius1_interp * math.cos(angle1)
            p3_y = center[1] + radius1_interp * math.sin(angle1)
            p3_z = z_start + t1 * (z_end - z_start)
            current_p3 = np.array([p3_x, p3_y, p3_z])

        # Tangent vectors (rotated radius vectors with length = radius)
        r_vec0 = (current_p0[0] - center[0], current_p0[1] - center[1])
        r_vec1 = (current_p3[0] - center[0], current_p3[1] - center[1])

        if clockwise:
            t_vec0 = (r_vec0[1], -r_vec0[0])
            t_vec1 = (r_vec1[1], -r_vec1[0])
        else:
            t_vec0 = (-r_vec0[1], r_vec0[0])
            t_vec1 = (-r_vec1[1], r_vec1[0])

        # P1 = P0 + kappa * T0
        c1 = (
            current_p0[0] + t_vec0[0] * kappa,
            current_p0[1] + t_vec0[1] * kappa,
        )
        # P2 = P3 - kappa * T1
        c2 = (
            current_p3[0] - t_vec1[0] * kappa,
            current_p3[1] - t_vec1[1] * kappa,
        )

        # Build the command row
        row = np.zeros(8, dtype=np.float64)
        row[COL_TYPE] = CMD_TYPE_BEZIER
        row[COL_X : COL_Z + 1] = current_p3
        row[COL_C1X] = c1[0]
        row[COL_C1Y] = c1[1]
        row[COL_C2X] = c2[0]
        row[COL_C2Y] = c2[1]
        bezier_rows.append(row)

        current_p0 = current_p3

    return bezier_rows
