from typing import List, Tuple
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
)
from .primitives import line_segment_intersection
from .types import Point3D

_DRAW_CMDS = (CMD_TYPE_LINE, CMD_TYPE_ARC, CMD_TYPE_BEZIER)


def _get_segments_for_row(
    data: np.ndarray, index: int
) -> List[Tuple[Point3D, Point3D]]:
    """
    Returns a list of linearized line segments for a given row in a data array.
    """
    row = data[index]
    cmd_type = row[COL_TYPE]
    end_point = (row[COL_X], row[COL_Y], row[COL_Z])

    start_point = (0.0, 0.0, 0.0)
    if index > 0:
        prev_row = data[index - 1]
        start_point = (prev_row[COL_X], prev_row[COL_Y], prev_row[COL_Z])

    if cmd_type == CMD_TYPE_LINE:
        return [(start_point, end_point)]
    elif cmd_type == CMD_TYPE_ARC:
        return linearize_arc(row, start_point)
    elif cmd_type == CMD_TYPE_BEZIER:
        return linearize_bezier_from_array(row, start_point)
    return []


def _precompute_row_segments(data: np.ndarray):
    rows = []
    for i in range(len(data)):
        if data[i, COL_TYPE] not in _DRAW_CMDS:
            continue
        segments = _get_segments_for_row(data, i)
        if not segments:
            continue
        all_x = [c for p1, p2 in segments for c in (p1[0], p2[0])]
        all_y = [c for p1, p2 in segments for c in (p1[1], p2[1])]
        rows.append(
            (i, segments, (min(all_x), min(all_y), max(all_x), max(all_y)))
        )
    return rows


def _data_intersect(
    data1: np.ndarray,
    data2: np.ndarray,
    is_self_check: bool = False,
    fail_on_t_junction: bool = False,
) -> bool:
    """Core logic to check for intersections between two numpy data arrays."""
    rows1 = _precompute_row_segments(data1)
    rows2 = _precompute_row_segments(data2)

    for idx1, (i, segments1, bb1) in enumerate(rows1):
        for idx2, (j, segments2, bb2) in enumerate(rows2):
            if is_self_check and j <= i:
                continue

            if bb1[2] < bb2[0] or bb1[0] > bb2[2]:
                continue
            if bb1[3] < bb2[1] or bb1[1] > bb2[3]:
                continue

            for seg1_p1, seg1_p2 in segments1:
                for seg2_p1, seg2_p2 in segments2:
                    intersection = line_segment_intersection(
                        seg1_p1[:2],
                        seg1_p2[:2],
                        seg2_p1[:2],
                        seg2_p2[:2],
                    )

                    if intersection:
                        is_adjacent_check = is_self_check and (j == i + 1)
                        if is_adjacent_check:
                            shared_vertex = data1[i, COL_X : COL_Y + 1]
                            dist_sq = np.sum(
                                (np.array(intersection) - shared_vertex) ** 2
                            )
                            if dist_sq < 1e-12:
                                continue
                            else:
                                return True

                        is_at_endpoint1 = (
                            np.sum((np.array(intersection) - seg1_p1[:2]) ** 2)
                            < 1e-12
                            or np.sum(
                                (np.array(intersection) - seg1_p2[:2]) ** 2
                            )
                            < 1e-12
                        )
                        is_at_endpoint2 = (
                            np.sum((np.array(intersection) - seg2_p1[:2]) ** 2)
                            < 1e-12
                            or np.sum(
                                (np.array(intersection) - seg2_p2[:2]) ** 2
                            )
                            < 1e-12
                        )
                        is_at_vertex = is_at_endpoint1 or is_at_endpoint2

                        if (
                            is_self_check
                            and is_at_vertex
                            and not fail_on_t_junction
                        ):
                            continue
                        return True
    return False


def check_self_intersection_from_array(
    data: np.ndarray, fail_on_t_junction: bool = False
) -> bool:
    """Checks if a path defined by a numpy array self-intersects."""
    move_indices = np.where(data[:, COL_TYPE] == CMD_TYPE_MOVE)[0]
    subpaths = np.split(data, move_indices[1:])

    for subpath_data in subpaths:
        if len(subpath_data) > 1:
            if _data_intersect(
                subpath_data,
                subpath_data,
                is_self_check=True,
                fail_on_t_junction=fail_on_t_junction,
            ):
                return True
    return False


def check_intersection_from_array(
    data1: np.ndarray, data2: np.ndarray, fail_on_t_junction: bool = False
) -> bool:
    """Checks if two paths defined by numpy arrays intersect."""
    return _data_intersect(
        data1,
        data2,
        is_self_check=False,
        fail_on_t_junction=fail_on_t_junction,
    )
