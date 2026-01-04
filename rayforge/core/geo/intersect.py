from typing import List, Tuple
import numpy as np

from .linearize import linearize_arc
from .primitives import line_segment_intersection
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
)


def _get_segments_for_row(
    data: np.ndarray, index: int
) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
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
    return []


def _data_intersect(
    data1: np.ndarray,
    data2: np.ndarray,
    is_self_check: bool = False,
    fail_on_t_junction: bool = False,
) -> bool:
    """Core logic to check for intersections between two numpy data arrays."""
    for i in range(len(data1)):
        cmd_type1 = data1[i, COL_TYPE]
        if cmd_type1 not in (CMD_TYPE_LINE, CMD_TYPE_ARC):
            continue

        start_idx_j = i + 1 if is_self_check else 0
        for j in range(start_idx_j, len(data2)):
            cmd_type2 = data2[j, COL_TYPE]
            if cmd_type2 not in (CMD_TYPE_LINE, CMD_TYPE_ARC):
                continue

            segments1 = _get_segments_for_row(data1, i)
            segments2 = _get_segments_for_row(data2, j)

            for seg1_p1, seg1_p2 in segments1:
                for seg2_p1, seg2_p2 in segments2:
                    intersection = line_segment_intersection(
                        seg1_p1[:2], seg1_p2[:2], seg2_p1[:2], seg2_p2[:2]
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
