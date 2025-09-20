import math
from typing import List, Tuple, Any, Optional
from itertools import groupby
import numpy as np
from scipy.optimize import least_squares

from .linearize import linearize_arc


def get_subpath_vertices(
    commands: List[Any], start_cmd_index: int
) -> List[Tuple[float, float]]:
    """
    Extracts all 2D vertices for a single continuous subpath starting at a
    given MoveToCommand index, linearizing any arcs.
    """
    from .geometry import MoveToCommand, LineToCommand, ArcToCommand

    vertices: List[Tuple[float, float]] = []
    if start_cmd_index >= len(commands):
        return []
    last_pos_3d = commands[start_cmd_index].end or (0.0, 0.0, 0.0)
    vertices.append(last_pos_3d[:2])

    for i in range(start_cmd_index + 1, len(commands)):
        cmd = commands[i]
        if isinstance(cmd, MoveToCommand):
            # End of the subpath
            break
        if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
            continue

        if isinstance(cmd, LineToCommand):
            vertices.append(cmd.end[:2])
        elif isinstance(cmd, ArcToCommand):
            segments = linearize_arc(cmd, last_pos_3d)
            for _, p2 in segments:
                vertices.append(p2[:2])
        last_pos_3d = cmd.end

    return vertices


def get_path_winding_order(commands: List[Any], segment_index: int) -> str:
    """
    Determines winding order ('cw', 'ccw', 'unknown') for the subpath at a
    given index.
    """
    from .geometry import MoveToCommand

    # Find the start of the subpath for the given segment
    subpath_start_index = -1
    for i in range(segment_index, -1, -1):
        if isinstance(commands[i], MoveToCommand):
            subpath_start_index = i
            break
    if subpath_start_index == -1:
        return "unknown"

    vertices = get_subpath_vertices(commands, subpath_start_index)
    if len(vertices) < 3:
        return "unknown"  # Not a closed polygon

    # Shoelace formula to calculate signed area
    area = 0.0
    for i in range(len(vertices)):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % len(vertices)]
        area += (p1[0] * p2[1]) - (p2[0] * p1[1])

    # Convention: positive area is CCW, negative is CW in a Y-up system
    if abs(area) < 1e-9:
        return "unknown"
    elif area > 0:
        return "ccw"
    else:
        return "cw"


def get_point_and_tangent_at(
    commands: List[Any], segment_index: int, t: float
) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """
    Calculates the 2D point and tangent vector at a parameter 't' along a
    segment.
    """
    from .geometry import LineToCommand, ArcToCommand, MovingCommand

    cmd = commands[segment_index]
    if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
        return None

    # Find the start point of this segment
    start_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    for i in range(segment_index - 1, -1, -1):
        prev_cmd = commands[i]
        if isinstance(prev_cmd, MovingCommand) and prev_cmd.end:
            start_pos_3d = prev_cmd.end
            break

    p0 = start_pos_3d[:2]
    p1 = cmd.end[:2]

    if isinstance(cmd, LineToCommand):
        point = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        tangent_vec = (p1[0] - p0[0], p1[1] - p0[1])
    elif isinstance(cmd, ArcToCommand):
        center = (
            p0[0] + cmd.center_offset[0],
            p0[1] + cmd.center_offset[1],
        )
        start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
        end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angle_range = end_angle - start_angle

        if cmd.clockwise:
            if angle_range > 0:
                angle_range -= 2 * math.pi
        else:
            if angle_range < 0:
                angle_range += 2 * math.pi

        current_angle = start_angle + t * angle_range
        radius_start = math.dist(p0, center)
        radius_end = math.dist(p1, center)
        radius = radius_start + t * (radius_end - radius_start)

        point = (
            center[0] + radius * math.cos(current_angle),
            center[1] + radius * math.sin(current_angle),
        )

        radius_vec = (point[0] - center[0], point[1] - center[1])
        if cmd.clockwise:
            tangent_vec = (radius_vec[1], -radius_vec[0])
        else:
            tangent_vec = (-radius_vec[1], radius_vec[0])
    else:
        return None

    norm = math.hypot(tangent_vec[0], tangent_vec[1])
    if norm < 1e-9:
        return point, (1.0, 0.0)

    normalized_tangent = (tangent_vec[0] / norm, tangent_vec[1] / norm)
    return point, normalized_tangent


def get_outward_normal_at(
    commands: List[Any], segment_index: int, t: float
) -> Optional[Tuple[float, float]]:
    """
    Calculates the outward-pointing normal vector for a point on a closed
    path.
    """
    winding = get_path_winding_order(commands, segment_index)
    if winding == "unknown":
        return None

    result = get_point_and_tangent_at(commands, segment_index, t)
    if not result:
        return None

    _, tangent = result
    tx, ty = tangent

    if winding == "ccw":
        return (ty, -tx)
    else:  # winding == "cw"
        return (-ty, tx)


def get_angle_at_vertex(
    p0: Tuple[float, ...], p1: Tuple[float, ...], p2: Tuple[float, ...]
) -> float:
    """
    Calculates the internal angle of the corner at point p1 in the
    XY plane. Returns the angle in radians.
    """
    # Create vectors from p1 to p0 and p1 to p2.
    v1x, v1y = p0[0] - p1[0], p0[1] - p1[1]
    v2x, v2y = p2[0] - p1[0], p2[1] - p1[1]

    # Calculate magnitudes for normalization.
    mag_v1 = math.hypot(v1x, v1y)
    mag_v2 = math.hypot(v2x, v2y)
    mag_prod = mag_v1 * mag_v2
    if mag_prod < 1e-9:
        return math.pi  # Straight line if points are coincident.

    # Dot product and normalization to get cosine of the angle.
    dot = v1x * v2x + v1y * v2y
    cos_theta = min(1.0, max(-1.0, dot / mag_prod))

    # Return angle in radians.
    return math.acos(cos_theta)


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


def remove_duplicates(
    points: List[Tuple[float, ...]],
) -> List[Tuple[float, ...]]:
    """Removes consecutive duplicate points from a list."""
    return [k for k, v in groupby(points)]


def is_clockwise(points: List[Tuple[float, ...]]) -> bool:
    """
    Determines if the first three points in a list form a clockwise turn
    using the 2D cross product.
    """
    if len(points) < 3:
        return False  # Not enough points to determine direction

    p1, p2, p3 = points[0], points[1], points[2]
    cross_product = (p2[0] - p1[0]) * (p3[1] - p2[1]) - (p2[1] - p1[1]) * (
        p3[0] - p2[0]
    )
    return cross_product < 0


def arc_direction_is_clockwise(
    points: List[Tuple[float, ...]], center: Tuple[float, float]
) -> bool:
    """
    Determines the winding direction of a sequence of points around a center
    by summing the cross products of vectors from the center to consecutive
    points. A negative sum indicates a net clockwise rotation.
    """
    xc, yc = center
    cross_product_sum = 0.0
    for i in range(len(points) - 1):
        x0, y0 = points[i][:2]
        x1, y1 = points[i + 1][:2]
        # Vectors from center to points
        v0x, v0y = x0 - xc, y0 - yc
        v1x, v1y = x1 - xc, y1 - yc
        # 2D Cross product
        cross_product_sum += v0x * v1y - v0y * v1x

    return cross_product_sum < 0
