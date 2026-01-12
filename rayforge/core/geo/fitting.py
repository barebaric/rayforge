import logging
import math
from typing import List, Tuple, Optional, cast, Callable
import numpy as np
from scipy.optimize import least_squares
from .analysis import arc_direction_is_clockwise
from .constants import (
    CMD_TYPE_BEZIER,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_MOVE,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
    COL_X,
    COL_Y,
    COL_Z,
    COL_TYPE,
    COL_I,
    COL_J,
    COL_CW,
    GEO_ARRAY_COLS,
)
from .linearize import linearize_bezier_from_array, linearize_arc
from .primitives import get_arc_angles
from .simplify import simplify_points_to_array


logger = logging.getLogger(__name__)


Point2DOr3D = Tuple[float, float] | Tuple[float, float, float]


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


def fit_circle_3_points(
    p1: Tuple[float, ...], p2: Tuple[float, ...], p3: Tuple[float, ...]
) -> Optional[Tuple[Tuple[float, float], float]]:
    """
    Analytically calculates the center and radius of a circle passing through
    three 2D points. Returns None if the points are collinear.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    x3, y3 = p3[:2]

    # Check for collinearity using the area of the triangle.
    # A small area indicates the points are nearly collinear.
    area = x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)
    if abs(area) < 1e-9:
        return None

    # Using the perpendicular bisector method.
    # The center (xc, yc) is the intersection of the perpendicular bisectors
    # of the chords p1-p2 and p2-p3.
    d12 = 2 * ((y2 - y1) * (x3 - x2) - (y3 - y2) * (x2 - x1))
    if abs(d12) < 1e-9:
        return None  # Should be caught by collinearity check, but for safety.

    sq1 = x1**2 + y1**2
    sq2 = x2**2 + y2**2
    sq3 = x3**2 + y3**2

    xc = ((sq1 - sq2) * (y3 - y2) - (sq2 - sq3) * (y2 - y1)) / d12
    yc = ((x2 - x1) * (sq2 - sq3) - (x3 - x2) * (sq1 - sq2)) / d12

    center = (xc, yc)
    radius = math.hypot(x1 - xc, y1 - yc)

    return center, radius


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


def project_circle_center_to_bisector(
    p1: Tuple[float, ...], p2: Tuple[float, ...], center: Tuple[float, float]
) -> Tuple[float, float]:
    """
    Adjusts a circle center point so that it lies exactly on the perpendicular
    bisector of the segment p1-p2.

    This guarantees that dist(center, p1) == dist(center, p2), which is
    required for G-code arcs to avoid "invalid target" errors (Error 33).
    It minimizes the movement of the center point.
    """
    x1, y1 = p1[:2]
    x2, y2 = p2[:2]
    cx, cy = center

    dx = x2 - x1
    dy = y2 - y1
    chord_len_sq = dx * dx + dy * dy

    if chord_len_sq < 1e-12:
        # p1 and p2 are coincident; any center is equidistant.
        return center

    # Midpoint of the chord
    mx = (x1 + x2) / 2.0
    my = (y1 + y2) / 2.0

    # Vector from Midpoint to Center
    vx = cx - mx
    vy = cy - my

    # Project M->C vector onto the chord vector.
    # This gives us the component of the offset that is parallel to the chord.
    # We want to remove this component to place C on the perpendicular
    # bisector.
    dot = vx * dx + vy * dy
    proj_factor = dot / chord_len_sq

    proj_x = dx * proj_factor
    proj_y = dy * proj_factor

    # New center = Old center - parallel component
    return (cx - proj_x, cy - proj_y)


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


def get_max_line_deviation(
    pts: List[Tuple[float, ...]], start_idx: int, end_idx: int
) -> Tuple[float, int]:
    """
    Calculates the max deviation from the chord line between two points.

    Args:
        pts: List of points.
        start_idx: Index of the start point.
        end_idx: Index of the end point.

    Returns:
        Tuple of (max_distance, index_of_furthest_point).
    """
    p_start = pts[start_idx]
    p_end = pts[end_idx]
    dx = p_end[0] - p_start[0]
    dy = p_end[1] - p_start[1]
    line_len_sq = dx * dx + dy * dy

    max_dist_sq = 0.0
    max_idx = start_idx

    if line_len_sq < 1e-12:
        for i in range(start_idx + 1, end_idx):
            p = pts[i]
            d_sq = (p[0] - p_start[0]) ** 2 + (p[1] - p_start[1]) ** 2
            if d_sq > max_dist_sq:
                max_dist_sq = d_sq
                max_idx = i
        return math.sqrt(max_dist_sq), max_idx

    for i in range(start_idx + 1, end_idx):
        p = pts[i]
        cross_prod = (p[0] - p_start[0]) * dy - (p[1] - p_start[1]) * dx
        d_sq = (cross_prod * cross_prod) / line_len_sq
        if d_sq > max_dist_sq:
            max_dist_sq = d_sq
            max_idx = i

    return math.sqrt(max_dist_sq), max_idx


def create_line_cmd(end_point: Point2DOr3D) -> np.ndarray:
    """
    Creates a line command array.

    Args:
        end_point: The 3D end point (x, y, z) of the line.

    Returns:
        A numpy array representing a line command.
    """
    row = np.zeros(8, dtype=np.float64)
    row[COL_TYPE] = CMD_TYPE_LINE
    row[COL_X] = end_point[0]
    row[COL_Y] = end_point[1]
    row[COL_Z] = end_point[2] if len(end_point) > 2 else 0.0
    return row


def create_arc_cmd(
    end_point: Point2DOr3D,
    center: Tuple[float, float],
    start_point: Point2DOr3D,
) -> np.ndarray:
    """
    Creates an arc command array.

    Args:
        end_point: The 3D end point (x, y, z) of the arc.
        center: The 2D center (xc, yc) of the arc.
        start_point: The 3D start point (x, y, z) of the arc.

    Returns:
        A numpy array representing an arc command.
    """
    row = np.zeros(8, dtype=np.float64)
    row[COL_TYPE] = CMD_TYPE_ARC
    row[COL_X] = end_point[0]
    row[COL_Y] = end_point[1]
    row[COL_Z] = end_point[2] if len(end_point) > 2 else 0.0

    xc, yc = center
    row[COL_I] = xc - start_point[0]
    row[COL_J] = yc - start_point[1]

    v1x = start_point[0] - xc
    v1y = start_point[1] - yc
    v2x = end_point[0] - xc
    v2y = end_point[1] - yc
    cross = v1x * v2y - v1y * v2x
    row[COL_CW] = 1.0 if cross < 0 else 0.0

    return row


def fit_points_recursive(
    points: List[Tuple[float, ...]],
    tolerance: float,
    start: int,
    end: int,
) -> List[np.ndarray]:
    """
    Recursively fits primitives to a segment of points.

    Args:
        points: List of points.
        tolerance: Maximum allowable deviation.
        start: Start index of the segment.
        end: End index of the segment.

    Returns:
        List of command arrays.
    """
    if start >= end:
        return []

    # First, try to fit a straight line to the segment.
    max_dist, split_idx = get_max_line_deviation(points, start, end)
    if max_dist < tolerance:
        pt = points[end]
        if len(pt) < 2:
            raise ValueError(f"Point must have at least 2 coordinates: {pt}")
        return [create_line_cmd(cast(Point2DOr3D, pt))]

    # --- Fast path for 3-point segments ---
    # Use a faster analytical method instead of the general-purpose,
    # expensive least-squares solver for this common case.
    if end - start == 2:  # Segment has 3 points: start, start+1, end
        p1, p2, p3 = points[start], points[start + 1], points[end]
        fast_fit = fit_circle_3_points(p1, p2, p3)
        if fast_fit:
            center, radius = fast_fit

            # Ensure mathematically perfect arc (Start R == End R)
            center = project_circle_center_to_bisector(p1, p3, center)
            # Re-calculate radius to align perfectly with start/end
            radius = math.hypot(p1[0] - center[0], p1[1] - center[1])

            # We still must check if the arc deviates too far from the
            # original polyline segments (p1-p2 and p2-p3).
            arc_dev = get_arc_to_polyline_deviation(
                [p1, p2, p3], center, radius
            )
            if arc_dev < tolerance:
                row = create_arc_cmd(
                    cast(Point2DOr3D, p3), center, cast(Point2DOr3D, p1)
                )
                is_cw = arc_direction_is_clockwise([p1, p2, p3], center)
                row[COL_CW] = 1.0 if is_cw else 0.0
                return [row]
        # If fast fit fails, we fall through to the splitting logic below.

    # --- General case for > 3 points ---
    # If a line doesn't fit, try to fit a circular arc.
    subset = points[start : end + 1]
    fit_result = fit_circle_to_points(subset)
    if fit_result:
        center, _, _ = fit_result

        # Project center to bisector to guarantee R_start == R_end.
        # This prevents G-code error 33 ("invalid target") on strict
        # controllers.
        center = project_circle_center_to_bisector(
            points[start], points[end], center
        )
        # Recalculate radius based on start point and corrected center
        radius = math.hypot(
            points[start][0] - center[0], points[start][1] - center[1]
        )

        # Check deviation against the *corrected* arc
        arc_dev = get_arc_to_polyline_deviation(subset, center, radius)

        if arc_dev < tolerance:
            end_pt = points[end]
            start_pt = points[start]
            if len(end_pt) < 2 or len(start_pt) < 2:
                raise ValueError(
                    f"Points must have at least 2 coordinates: "
                    f"start={start_pt}, end={end_pt}"
                )
            row = create_arc_cmd(
                cast(Point2DOr3D, end_pt), center, cast(Point2DOr3D, start_pt)
            )
            is_cw = arc_direction_is_clockwise(subset, center)
            row[COL_CW] = 1.0 if is_cw else 0.0
            return [row]

    # If neither a line nor an arc fits, split the segment and recurse.
    if split_idx == start or split_idx == end:
        split_idx = (start + end) // 2

    left_cmds = fit_points_recursive(points, tolerance, start, split_idx)
    right_cmds = fit_points_recursive(points, tolerance, split_idx, end)
    return left_cmds + right_cmds


def fit_points_to_primitives(
    points: List[Tuple[float, ...]], tolerance: float
) -> List[np.ndarray]:
    """
    Approximates a list of points with a sequence of Line and Arc commands
    (CMD_TYPE_LINE, CMD_TYPE_ARC) stored in numpy arrays.

    This function recursively fits the best primitive to segments of the point
    list. It prefers straight lines if they fall within tolerance, otherwise
    it attempts to fit a circular arc. If neither fits, it subdivides the path.

    Args:
        points: A list of 2D or 3D points representing the path.
        tolerance: The maximum allowable deviation from the original path.

    Returns:
        A list of numpy arrays, where each array is a geometry command row.
    """
    if len(points) < 2:
        return []

    return fit_points_recursive(points, tolerance, 0, len(points) - 1)


def fit_arcs(
    data: Optional[np.ndarray],
    tolerance: float,
    on_progress: Optional[Callable[[float], None]] = None,
) -> Optional[np.ndarray]:
    """
    Reconstructs geometry data using an optimal set of Line and Arc
    commands. This function is optimized to handle both polylines and
    existing curves (like Beziers) efficiently, ensuring output
    contains only Lines and Arcs.

    Args:
        data: NumPy array of geometry commands.
        tolerance: The maximum allowable deviation.
        on_progress: An optional callback function that receives progress
                     updates from 0.0 to 1.0.

    Returns:
        A NumPy array of geometry commands.
    """
    if data is None or len(data) == 0:
        return data

    logger.debug("Starting optimized fit_arcs process...")
    new_rows: List[np.ndarray] = []
    line_point_chain: List[Tuple[float, float, float]] = []

    # Calculate linearization resolution.
    resolution = tolerance * 0.25
    total_rows = len(data)

    def flush_line_chain():
        """
        Processes the collected chain of line segment points with arc
        fitting.
        """
        nonlocal line_point_chain
        if len(line_point_chain) > 1:
            # 1. Convert to numpy for fast processing
            points_arr = np.array(line_point_chain, dtype=np.float64)

            # 2. Simplify points using Ramer-Douglas-Peucker.
            # This drastically reduces the point count (e.g. 1000 -> 20)
            # before the expensive curve fitting runs, while guaranteeing
            # the shape remains within 'tolerance'.
            simplified_arr = simplify_points_to_array(points_arr, tolerance)

            # 3. Convert back to list of tuples for the fitter
            simplified_points = [tuple(p) for p in simplified_arr.tolist()]

            # 4. Run the curve fitting on the simplified set
            primitives = fit_points_to_primitives(simplified_points, tolerance)
            new_rows.extend(primitives)

        line_point_chain = []

    last_pos = (0.0, 0.0, 0.0)

    for i, row in enumerate(data):
        if on_progress and i % 50 == 0:
            on_progress(i / total_rows)

        cmd_type = row[COL_TYPE]
        end_pos = (row[COL_X], row[COL_Y], row[COL_Z])

        # If we hit a Move, the previous continuous path ends.
        if cmd_type == CMD_TYPE_MOVE:
            flush_line_chain()
            new_rows.append(row)
            last_pos = end_pos
            continue

        # Initialize chain start if needed
        if not line_point_chain:
            line_point_chain.append(last_pos)

        if cmd_type == CMD_TYPE_LINE:
            line_point_chain.append(end_pos)

        elif cmd_type == CMD_TYPE_ARC:
            # Linearize Arc and add to chain
            segments = linearize_arc(row, last_pos, resolution)
            for _, p_end in segments:
                line_point_chain.append(p_end)

        elif cmd_type == CMD_TYPE_BEZIER:
            # Linearize Bezier and add to chain
            segments = linearize_bezier_from_array(row, last_pos, resolution)
            for _, p_end in segments:
                line_point_chain.append(p_end)

        last_pos = end_pos

    # Process the final chain
    flush_line_chain()

    if not new_rows:
        return np.empty((0, 8), dtype=np.float64)
    else:
        return np.array(new_rows, dtype=np.float64)


def optimize_path_from_array(
    data: Optional[np.ndarray], tolerance: float, fit_arcs: bool
) -> np.ndarray:
    """
    Optimizes a geometry numpy array by processing chains of line segments.
    It can either simplify these chains into fewer lines (RDP) or fit
    lines and arcs to them. Non-line commands (arcs, beziers) are preserved.
    """
    if data is None or len(data) == 0:
        return np.array([])

    optimized_rows: List[np.ndarray] = []
    point_chain: List[Tuple[float, float, float]] = []

    def flush_chain():
        nonlocal point_chain
        if len(point_chain) > 1:
            if fit_arcs:
                primitives = fit_points_to_primitives(point_chain, tolerance)
                optimized_rows.extend(primitives)
            else:
                points_arr = np.array(point_chain)
                simplified_arr = simplify_points_to_array(
                    points_arr, tolerance
                )
                for p in simplified_arr[1:]:
                    row = np.zeros(GEO_ARRAY_COLS, dtype=np.float64)
                    row[COL_TYPE] = CMD_TYPE_LINE
                    row[1:4] = p
                    optimized_rows.append(row)
        point_chain = []

    last_pos = (0.0, 0.0, 0.0)
    for row in data:
        cmd_type = row[COL_TYPE]
        end_pos = (row[COL_X], row[COL_Y], row[COL_Z])

        if cmd_type == CMD_TYPE_LINE:
            if not point_chain:
                point_chain.append(last_pos)
            point_chain.append(end_pos)
        else:
            flush_chain()
            optimized_rows.append(row)
            point_chain = [end_pos]

        last_pos = end_pos

    flush_chain()

    if not optimized_rows:
        return np.array([]).reshape(0, GEO_ARRAY_COLS)

    return np.array(optimized_rows)
