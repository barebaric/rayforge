from typing import List, Optional
import numpy as np
from .arc import linearize_arc
from .bezier import linearize_bezier_from_array
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    GEO_ARRAY_COLS,
)
from .types import Point3D


def flatten_to_points(
    data: Optional[np.ndarray], resolution: float
) -> List[List[Point3D]]:
    """
    Converts geometry data into a list of dense point lists (one per
    subpath). Arcs and Beziers are linearized using the given resolution.

    Args:
        data: NumPy array of geometry commands.
        resolution: The resolution for linearizing curves.

    Returns:
        A list of subpaths, where each subpath is a list of (x, y, z) points.
    """
    if data is None or len(data) == 0:
        return []

    subpaths: List[List[Point3D]] = []
    current_subpath: List[Point3D] = []
    last_pos = (0.0, 0.0, 0.0)

    for row in data:
        cmd_type = row[COL_TYPE]
        end_pos = (row[COL_X], row[COL_Y], row[COL_Z])

        if cmd_type == CMD_TYPE_MOVE:
            if current_subpath:
                subpaths.append(current_subpath)
                current_subpath = []
            current_subpath.append(end_pos)
        elif cmd_type == CMD_TYPE_LINE:
            current_subpath.append(end_pos)
        elif cmd_type == CMD_TYPE_ARC:
            segments = linearize_arc(row, last_pos, resolution)
            for _, p_end in segments:
                current_subpath.append(p_end)
        elif cmd_type == CMD_TYPE_BEZIER:
            segments = linearize_bezier_from_array(row, last_pos, resolution)
            for _, p_end in segments:
                current_subpath.append(p_end)

        last_pos = end_pos

    if current_subpath:
        subpaths.append(current_subpath)

    return subpaths


def linearize_geometry(
    data: Optional[np.ndarray], tolerance: float
) -> np.ndarray:
    """
    Converts geometry data to a polyline approximation (Lines only),
    reducing vertex count using the Ramer-Douglas-Peucker algorithm.

    Args:
        data: NumPy array of geometry commands.
        tolerance: The maximum allowable deviation.

    Returns:
        A NumPy array containing only MOVE and LINE commands.
    """
    from .simplify import simplify_points_to_array

    if data is None or len(data) == 0:
        return np.array([]).reshape(0, GEO_ARRAY_COLS)

    resolution = tolerance * 0.25
    subpaths_points = flatten_to_points(data, resolution)

    new_rows = []
    for points in subpaths_points:
        if not points:
            continue

        pts_arr = np.array(points, dtype=np.float64)
        simplified_arr = simplify_points_to_array(pts_arr, tolerance)

        if len(simplified_arr) > 0:
            p0 = simplified_arr[0]
            new_rows.append([CMD_TYPE_MOVE, p0[0], p0[1], p0[2], 0, 0, 0, 0])

            for i in range(1, len(simplified_arr)):
                p = simplified_arr[i]
                new_rows.append([CMD_TYPE_LINE, p[0], p[1], p[2], 0, 0, 0, 0])

    if not new_rows:
        return np.array([]).reshape(0, GEO_ARRAY_COLS)

    return np.array(new_rows, dtype=np.float64)
