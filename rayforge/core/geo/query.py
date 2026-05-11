import math
from typing import List, Tuple, Optional
import numpy as np
from .arc import (
    find_closest_point_on_arc,
    get_arc_bounding_box,
)
from .bezier import find_closest_point_on_bezier, linearize_bezier_from_array
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
)
from .primitives import find_closest_point_on_line_segment
from .types import Point, Point3D, Rect


def _compute_cubic_bezier_bounds_1d(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes min and max values for 1D cubic Bezier coefficients vectorized.
    p0, p1, p2, p3 are 1D arrays of shape (N,).
    Returns (min_vals, max_vals).
    """
    # Start with endpoints
    local_min = np.minimum(p0, p3)
    local_max = np.maximum(p0, p3)

    # Derivatives coefficients: at^2 + bt + c = 0
    # B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3
    # B'(t) coefficients:
    a = 3 * (-p0 + 3 * p1 - 3 * p2 + p3)
    b = 6 * (p0 - 2 * p1 + p2)
    c = 3 * (p1 - p0)

    # Solve quadratic equation for t: at^2 + bt + c = 0
    discriminant = b**2 - 4 * a * c

    # Filter for valid quadratic solutions (discriminant >= 0 and a != 0)
    # We use a small epsilon for a != 0
    valid_mask = (discriminant >= 0) & (np.abs(a) > 1e-9)

    if np.any(valid_mask):
        sqrt_disc = np.sqrt(discriminant[valid_mask])
        a_valid = a[valid_mask]
        b_valid = b[valid_mask]

        t1 = (-b_valid - sqrt_disc) / (2 * a_valid)
        t2 = (-b_valid + sqrt_disc) / (2 * a_valid)

        # Check t1
        t1_valid_mask = (t1 > 0) & (t1 < 1)
        if np.any(t1_valid_mask):
            # Indices in the original arrays where t1 is valid
            full_indices = np.where(valid_mask)[0][t1_valid_mask]
            t = t1[t1_valid_mask]

            # Evaluate Bezier at t
            mt = 1 - t
            val = (
                mt**3 * p0[full_indices]
                + 3 * mt**2 * t * p1[full_indices]
                + 3 * mt * t**2 * p2[full_indices]
                + t**3 * p3[full_indices]
            )

            local_min[full_indices] = np.minimum(local_min[full_indices], val)
            local_max[full_indices] = np.maximum(local_max[full_indices], val)

        # Check t2
        t2_valid_mask = (t2 > 0) & (t2 < 1)
        if np.any(t2_valid_mask):
            full_indices = np.where(valid_mask)[0][t2_valid_mask]
            t = t2[t2_valid_mask]

            mt = 1 - t
            val = (
                mt**3 * p0[full_indices]
                + 3 * mt**2 * t * p1[full_indices]
                + 3 * mt * t**2 * p2[full_indices]
                + t**3 * p3[full_indices]
            )

            local_min[full_indices] = np.minimum(local_min[full_indices], val)
            local_max[full_indices] = np.maximum(local_max[full_indices], val)

    # Handle linear/quadratic cases where a ~ 0 but b != 0
    # bt + c = 0  => t = -c/b
    linear_mask = (np.abs(a) <= 1e-9) & (np.abs(b) > 1e-9)
    if np.any(linear_mask):
        t = -c[linear_mask] / b[linear_mask]
        t_valid_mask = (t > 0) & (t < 1)

        if np.any(t_valid_mask):
            full_indices = np.where(linear_mask)[0][t_valid_mask]
            t_val = t[t_valid_mask]

            mt = 1 - t_val
            val = (
                mt**3 * p0[full_indices]
                + 3 * mt**2 * t_val * p1[full_indices]
                + 3 * mt * t_val**2 * p2[full_indices]
                + t_val**3 * p3[full_indices]
            )

            local_min[full_indices] = np.minimum(local_min[full_indices], val)
            local_max[full_indices] = np.maximum(local_max[full_indices], val)

    return local_min, local_max


def get_bounding_rect_from_array(
    data: np.ndarray,
) -> Rect:
    """
    Calculates the bounding box (min_x, min_y, max_x, max_y) from the
    geometry array.
    """
    if data is None or data.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0

    # 1. Gather all endpoints from the array.
    # Columns 1 and 2 are X and Y.
    points_x = data[:, COL_X]
    points_y = data[:, COL_Y]

    min_x = np.min(points_x)
    max_x = np.max(points_x)
    min_y = np.min(points_y)
    max_y = np.max(points_y)

    # 2. Handle Arcs.
    # Arcs might bulge outside the bounding box defined by their endpoints.
    # We iterate only over Arc rows.
    arc_indices = np.where(data[:, COL_TYPE] == CMD_TYPE_ARC)[0]

    if len(arc_indices) > 0:
        # We need the start point for each arc.
        # The start point of row[i] is the end point of row[i-1].
        # If i=0, start is (0,0) implicitly.

        # Get start indices (previous row index)
        start_indices = arc_indices - 1

        # Construct start points array
        start_points = np.zeros((len(arc_indices), 2))

        valid_starts_mask = start_indices >= 0
        if np.any(valid_starts_mask):
            valid_start_idxs = start_indices[valid_starts_mask]
            # Fetch X, Y from previous rows
            start_points[valid_starts_mask, 0] = data[valid_start_idxs, COL_X]
            start_points[valid_starts_mask, 1] = data[valid_start_idxs, COL_Y]

        # Iterate over arcs to check bounds.
        # Vectorizing get_arc_bounding_box is difficult due to conditional
        # logic, so we loop explicitly. This loop iterates only over arcs,
        # which are typically sparse compared to lines.
        for i, row_idx in enumerate(arc_indices):
            row = data[row_idx]
            start = (start_points[i, 0], start_points[i, 1])
            end = (row[COL_X], row[COL_Y])
            center_offset = (row[COL_I], row[COL_J])
            clockwise = bool(row[COL_CW])

            ax1, ay1, ax2, ay2 = get_arc_bounding_box(
                start, end, center_offset, clockwise
            )

            if ax1 < min_x:
                min_x = ax1
            if ay1 < min_y:
                min_y = ay1
            if ax2 > max_x:
                max_x = ax2
            if ay2 > max_y:
                max_y = ay2

    # 3. Handle Beziers.
    # The curve is guaranteed to be contained within the convex hull of its
    # control points (P0, C1, C2, P3). We use this property for a fast
    # safe bounding box calculation, instead of linearizing.
    bezier_indices = np.where(data[:, COL_TYPE] == CMD_TYPE_BEZIER)[0]
    if len(bezier_indices) > 0:
        start_indices = bezier_indices - 1
        start_points = np.zeros((len(bezier_indices), 3))  # X, Y, Z
        valid_starts_mask = start_indices >= 0
        if np.any(valid_starts_mask):
            valid_start_idxs = start_indices[valid_starts_mask]
            start_points[valid_starts_mask] = data[
                valid_start_idxs, COL_X : COL_Z + 1
            ]

        # Extract P0, P1 (C1), P2 (C2), P3
        p0_x = start_points[:, 0]
        p0_y = start_points[:, 1]

        p1_x = data[bezier_indices, COL_C1X]
        p1_y = data[bezier_indices, COL_C1Y]

        p2_x = data[bezier_indices, COL_C2X]
        p2_y = data[bezier_indices, COL_C2Y]

        p3_x = data[bezier_indices, COL_X]
        p3_y = data[bezier_indices, COL_Y]

        # Calculate exact bounds
        bx_min, bx_max = _compute_cubic_bezier_bounds_1d(
            p0_x, p1_x, p2_x, p3_x
        )
        by_min, by_max = _compute_cubic_bezier_bounds_1d(
            p0_y, p1_y, p2_y, p3_y
        )

        min_x = min(min_x, np.min(bx_min))
        max_x = max(max_x, np.max(bx_max))
        min_y = min(min_y, np.min(by_min))
        max_y = max(max_y, np.max(by_max))

    return float(min_x), float(min_y), float(max_x), float(max_y)


def get_total_distance_from_array(data: np.ndarray) -> float:
    """
    Calculates the total 2D path length for all moving commands in a numpy
    array.
    """
    total_dist = 0.0
    last_point = (0.0, 0.0, 0.0)

    for i in range(len(data)):
        row = data[i]
        cmd_type = row[COL_TYPE]
        end_point = (row[COL_X], row[COL_Y], row[COL_Z])

        if cmd_type in (CMD_TYPE_MOVE, CMD_TYPE_LINE):
            total_dist += math.hypot(
                end_point[0] - last_point[0], end_point[1] - last_point[1]
            )
        elif cmd_type == CMD_TYPE_ARC:
            center_offset = (row[COL_I], row[COL_J])
            clockwise = bool(row[COL_CW])
            center_x = last_point[0] + center_offset[0]
            center_y = last_point[1] + center_offset[1]
            radius = math.hypot(center_offset[0], center_offset[1])

            if radius > 1e-9:
                start_angle = math.atan2(
                    last_point[1] - center_y, last_point[0] - center_x
                )
                end_angle = math.atan2(
                    end_point[1] - center_y, end_point[0] - center_x
                )
                angle_span = end_angle - start_angle
                if clockwise:
                    if angle_span > 1e-9:
                        angle_span -= 2 * math.pi
                else:
                    if angle_span < -1e-9:
                        angle_span += 2 * math.pi
                total_dist += abs(angle_span * radius)
        elif cmd_type == CMD_TYPE_BEZIER:
            segments = linearize_bezier_from_array(row, last_point)
            for p1, p2 in segments:
                total_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        last_point = end_point

    return total_dist


def find_closest_point_on_path_from_array(
    data: np.ndarray, x: float, y: float
) -> Optional[Tuple[int, float, Point]]:
    """
    Finds the closest point on an entire path to a given 2D coordinate from a
    numpy array.
    """
    min_dist_sq = float("inf")
    closest_info: Optional[Tuple[int, float, Point]] = None

    last_pos_3d: Point3D = (0.0, 0.0, 0.0)
    for i in range(len(data)):
        row = data[i]
        cmd_type = row[COL_TYPE]
        end_point_3d = (row[COL_X], row[COL_Y], row[COL_Z])

        if cmd_type == CMD_TYPE_MOVE:
            last_pos_3d = end_point_3d
            continue

        start_pos_3d = last_pos_3d

        if cmd_type == CMD_TYPE_LINE:
            t, pt, dist_sq = find_closest_point_on_line_segment(
                start_pos_3d[:2], end_point_3d[:2], x, y
            )
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_info = (i, t, pt)

        elif cmd_type == CMD_TYPE_ARC:
            result = find_closest_point_on_arc(row, start_pos_3d, x, y)
            if result:
                t_arc, pt_arc, dist_sq_arc = result
                if dist_sq_arc < min_dist_sq:
                    min_dist_sq = dist_sq_arc
                    closest_info = (i, t_arc, pt_arc)

        elif cmd_type == CMD_TYPE_BEZIER:
            result = find_closest_point_on_bezier(row, start_pos_3d, x, y)
            if result:
                t_bezier, pt_bezier, dist_sq_bezier = result
                if dist_sq_bezier < min_dist_sq:
                    min_dist_sq = dist_sq_bezier
                    closest_info = (i, t_bezier, pt_bezier)

        last_pos_3d = end_point_3d

    return closest_info


def _segment_length_from_row(row: np.ndarray, start_point: Point3D) -> float:
    cmd_type = row[COL_TYPE]
    sx, sy = start_point[0], start_point[1]
    ex, ey = row[COL_X], row[COL_Y]

    if cmd_type in (CMD_TYPE_MOVE, CMD_TYPE_LINE):
        return math.hypot(ex - sx, ey - sy)
    elif cmd_type == CMD_TYPE_ARC:
        i_off, j_off = row[COL_I], row[COL_J]
        clockwise = bool(row[COL_CW])
        cx = sx + i_off
        cy = sy + j_off
        radius = math.hypot(i_off, j_off)
        if radius < 1e-9:
            return 0.0
        start_angle = math.atan2(sy - cy, sx - cx)
        end_angle = math.atan2(ey - cy, ex - cx)
        angle_span = end_angle - start_angle
        if abs(angle_span) < 1e-9:
            angle_span = -2 * math.pi if clockwise else 2 * math.pi
        elif clockwise:
            if angle_span > 1e-9:
                angle_span -= 2 * math.pi
        else:
            if angle_span < -1e-9:
                angle_span += 2 * math.pi
        return abs(angle_span * radius)
    elif cmd_type == CMD_TYPE_BEZIER:
        segments = linearize_bezier_from_array(row, start_point)
        return sum(
            math.hypot(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in segments
        )
    return 0.0


def _partial_segment_from_row(
    row: np.ndarray, start_point: Point3D, t: float
) -> Optional[np.ndarray]:
    cmd_type = row[COL_TYPE]
    sx, sy, sz = start_point
    ex, ey, ez = row[COL_X], row[COL_Y], row[COL_Z]

    if cmd_type == CMD_TYPE_LINE:
        nx = sx + t * (ex - sx)
        ny = sy + t * (ey - sy)
        nz = sz + t * (ez - sz)
        return np.array([CMD_TYPE_LINE, nx, ny, nz, 0, 0, 0, 0])

    elif cmd_type == CMD_TYPE_ARC:
        i_off, j_off = row[COL_I], row[COL_J]
        clockwise = bool(row[COL_CW])
        cx = sx + i_off
        cy = sy + j_off
        radius_start = math.hypot(i_off, j_off)
        radius_end = math.hypot(ex - cx, ey - cy)

        start_angle = math.atan2(sy - cy, sx - cx)
        end_angle = math.atan2(ey - cy, ex - cx)
        angle_span = end_angle - start_angle
        if abs(angle_span) < 1e-9:
            angle_span = -2 * math.pi if clockwise else 2 * math.pi
        elif clockwise:
            if angle_span > 1e-9:
                angle_span -= 2 * math.pi
        else:
            if angle_span < -1e-9:
                angle_span += 2 * math.pi

        mid_angle = start_angle + t * angle_span
        radius = radius_start + t * (radius_end - radius_start)
        nx = cx + radius * math.cos(mid_angle)
        ny = cy + radius * math.sin(mid_angle)
        nz = sz + t * (ez - sz)
        return np.array(
            [CMD_TYPE_ARC, nx, ny, nz, i_off, j_off, row[COL_CW], 0]
        )

    elif cmd_type == CMD_TYPE_BEZIER:
        c1x, c1y = row[COL_C1X], row[COL_C1Y]
        c2x, c2y = row[COL_C2X], row[COL_C2Y]

        p01x = sx + t * (c1x - sx)
        p01y = sy + t * (c1y - sy)
        p12x = c1x + t * (c2x - c1x)
        p12y = c1y + t * (c2y - c1y)
        p23x = c2x + t * (ex - c2x)
        p23y = c2y + t * (ey - c2y)
        p012x = p01x + t * (p12x - p01x)
        p012y = p01y + t * (p12y - p01y)
        p123x = p12x + t * (p23x - p12x)
        p123y = p12y + t * (p23y - p12y)
        p0123x = p012x + t * (p123x - p012x)
        p0123y = p012y + t * (p123y - p012y)

        nz = sz + t * (ez - sz)
        return np.array(
            [
                CMD_TYPE_BEZIER,
                p0123x,
                p0123y,
                nz,
                p01x,
                p01y,
                p012x,
                p012y,
            ]
        )

    return None


def extract_overcut_rows(
    data: Optional[np.ndarray], max_length: float
) -> Optional[np.ndarray]:
    """
    Extract drawing command rows from the start of a closed contour's
    drawing path, up to *max_length* distance.

    For a closed contour (MoveTo at row 0, last endpoint ≈ MoveTo point)
    the returned rows can be appended directly to the data array.  Because
    the contour is closed the implicit start-point of every copied segment
    aligns with the continuation of the original path.

    Partial segments are interpolated when *max_length* does not fall on a
    segment boundary.

    Returns an (N, 8) array, or None if nothing to extract.
    """
    if data is None or len(data) < 2 or max_length <= 0:
        return None

    last_point: Point3D = (
        data[0, COL_X],
        data[0, COL_Y],
        data[0, COL_Z],
    )
    accumulated = 0.0
    collected: List[np.ndarray] = []

    for i in range(1, len(data)):
        row = data[i]
        end_point: Point3D = (row[COL_X], row[COL_Y], row[COL_Z])

        seg_length = _segment_length_from_row(row, last_point)
        if seg_length < 1e-9:
            last_point = end_point
            continue

        if accumulated + seg_length <= max_length + 1e-9:
            collected.append(row.copy())
            accumulated += seg_length
            last_point = end_point
        else:
            remaining = max_length - accumulated
            if remaining > 1e-9:
                t = remaining / seg_length
                partial = _partial_segment_from_row(row, last_point, t)
                if partial is not None:
                    collected.append(partial)
            break

    if not collected:
        return None
    return np.vstack(collected)


def bboxes_intersect(bbox1: "Rect", bbox2: "Rect") -> bool:
    """
    Check if two axis-aligned bounding boxes intersect or touch.

    Args:
        bbox1: First bounding box as (min_x, min_y, max_x, max_y).
        bbox2: Second bounding box as (min_x, min_y, max_x, max_y).

    Returns:
        True if the bounding boxes intersect or touch, False otherwise.
    """
    return not (
        bbox1[2] < bbox2[0]
        or bbox1[0] > bbox2[2]
        or bbox1[3] < bbox2[1]
        or bbox1[1] > bbox2[3]
    )
