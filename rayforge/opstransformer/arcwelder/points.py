import math
import numpy as np
from scipy.optimize import least_squares
from itertools import groupby


def remove_duplicates(segment):
    """
    Removes *consecutive* duplicates from a list of points.
    """
    return [k for (k, v) in groupby(segment)]


def are_colinear(points, tolerance=0.01):
    """
    Check if all points are colinear within a given tolerance.

    Args:
        points: List of (x, y) tuples.
        tolerance: Max perpendicular distance from the line (default 0.01).

    Returns:
        bool: True if all points are colinear within tolerance.
    """
    if len(points) < 2:
        return True  # Fewer than 2 points are trivially colinear
    if len(points) == 2:
        return True  # Two points define a line

    # Define line by first and last points
    p1, p2 = points[0], points[-1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    line_length = math.hypot(dx, dy)

    if line_length == 0:
        # Points are coincident, check if all are within tolerance
        return all(math.hypot(p[0] - p1[0], p[1] - p1[1]) < tolerance
                   for p in points)

    # Check perpendicular distance of each point to the line p1-p2
    for p in points[1:-1]:  # Skip endpoints as they define the line
        # Vector from p1 to p
        vx = p[0] - p1[0]
        vy = p[1] - p1[1]
        # Perpendicular distance = |ax + by + c| / sqrt(a^2 + b^2)
        # Line equation: ax + by + c = 0, where
        #                a=dy, b=-dx, c=-(dy*p1x - dx*p1y)
        dist = abs(dy * vx - dx * vy) / line_length
        if dist > tolerance:
            return False
    return True


def is_clockwise(points):
    """
    Determines direction using cross product.
    """
    if len(points) < 3:
        return False

    p1, p2, p3 = points[0], points[1], points[2]
    cross = (
        (p2[0] - p1[0]) * (p3[1] - p2[1])
        - (p2[1] - p1[1]) * (p3[0] - p2[0])
    )
    return cross < 0


def arc_direction(points, center):
    xc, yc = center
    cross_sum = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i]
        x1, y1 = points[i + 1]
        dx0 = x0 - xc
        dy0 = y0 - yc
        dx1 = x1 - xc
        dy1 = y1 - yc
        cross = dx0 * dy1 - dy0 * dx1
        cross_sum += cross
    return cross_sum < 0  # True for clockwise


def fit_circle(points):
    """
    Fit a circle to points, return (center, radius, error) or None.
    Error is max of point-to-arc deviation.
    """
    if len(points) < 3 or are_colinear(points):
        return None

    x = np.array([p[0] for p in points])
    y = np.array([p[1] for p in points])

    # Initial guess: mean center and average radius
    x0, y0 = np.mean(x), np.mean(y)
    r0 = np.mean(np.sqrt((x-x0)**2 + (y-y0)**2))

    # Fit circle using least squares
    result = least_squares(
        lambda p: np.sqrt((x-p[0])**2 + (y-p[1])**2) - p[2],
        [x0, y0, r0],
        method='lm'
    )
    xc, yc, r = result.x
    center = (xc, yc)

    # Point-to-arc error: max deviation of points from circle
    distances = np.sqrt((x-xc)**2 + (y-yc)**2)
    point_error = np.max(np.abs(distances - r))

    # Total error: max of point fit and arc deviation
    return center, r, point_error


def arc_to_polyline_deviation(points, center, radius):
    """
    Compute max deviation of an arc from the original polyline.
    Args:
        points: List of (x, y) tuples forming the polyline.
        center: (xc, yc) tuple, center of the fitted circle.
        radius: Radius of the fitted circle.
    Returns:
        float: Max perpendicular distance from arc to polyline segments.
    """
    if len(points) < 2:
        return 0.0
    xc, yc = center
    max_deviation = 0.0

    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        segment_length = math.hypot(dx, dy)

        if segment_length == 0:
            distance = math.hypot(x1 - xc, y1 - yc)
            deviation = abs(distance - radius)
            max_deviation = max(max_deviation, deviation)
            continue

        # Distances from center to endpoints
        d1 = math.hypot(x1 - xc, y1 - yc)
        d2 = math.hypot(x2 - xc, y2 - yc)

        # If segment exceeds diameter, use endpoint deviations
        if segment_length > 2 * radius:
            deviation = max(abs(d1 - radius), abs(d2 - radius))
        else:
            # Vectors from center to points
            v1x, v1y = x1 - xc, y1 - yc
            v2x, v2y = x2 - xc, y2 - yc

            # Dot product to find angle
            dot = v1x * v2x + v1y * v2y
            mag1 = math.hypot(v1x, v1y)
            mag2 = math.hypot(v2x, v2y)
            if mag1 < 1e-6 or mag2 < 1e-6:
                deviation = abs(d1 - radius) if mag1 < 1e-6 \
                       else abs(d2 - radius)
            else:
                cos_theta = min(1.0, max(-1.0, dot / (mag1 * mag2)))
                theta = math.acos(cos_theta)
                # Sagitta based on actual arc angle
                sagitta = radius * (1 - math.cos(theta / 2))
                # Endpoint deviations if off-arc
                endpoint_dev = max(abs(d1 - radius), abs(d2 - radius))
                deviation = max(sagitta, endpoint_dev)

        max_deviation = max(max_deviation, deviation)
    return max_deviation
