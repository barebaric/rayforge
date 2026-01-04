import math
from typing import Tuple, Optional
import numpy as np
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
)
from .primitives import (
    find_closest_point_on_line_segment,
    find_closest_point_on_arc,
    get_arc_bounding_box,
)


def get_bounding_rect_from_array(
    data: np.ndarray,
) -> Tuple[float, float, float, float]:
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

        last_point = end_point

    return total_dist


def find_closest_point_on_path_from_array(
    data: np.ndarray, x: float, y: float
) -> Optional[Tuple[int, float, Tuple[float, float]]]:
    """
    Finds the closest point on an entire path to a given 2D coordinate from a
    numpy array.
    """
    min_dist_sq = float("inf")
    closest_info: Optional[Tuple[int, float, Tuple[float, float]]] = None

    last_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
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

        last_pos_3d = end_point_3d

    return closest_info
