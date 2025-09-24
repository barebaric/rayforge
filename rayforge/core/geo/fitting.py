import math
from typing import List, Tuple, Optional
import numpy as np
from scipy.optimize import least_squares


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
