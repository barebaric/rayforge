from __future__ import annotations
import numpy as np
from typing import List, Tuple, Optional, TYPE_CHECKING

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


if TYPE_CHECKING:
    from .geometry import Geometry


def _are_points_equal(
    p1: Tuple[float, ...],
    p2: Tuple[float, ...],
    tolerance: float,
) -> bool:
    """Check if two points are equal within tolerance."""
    if len(p1) != len(p2):
        return False
    for i in range(len(p1)):
        if abs(p1[i] - p2[i]) > tolerance:
            return False
    return True


def _get_segment_key(
    data: np.ndarray,
    idx: int,
    tolerance: float,
) -> Optional[Tuple]:
    """Get a hashable key for a segment at the given index."""
    if idx >= len(data):
        return None

    row = data[idx]
    cmd_type = row[COL_TYPE]

    if cmd_type == CMD_TYPE_LINE:
        end_point = (row[COL_X], row[COL_Y], row[COL_Z])
        return ("LINE", end_point)

    if cmd_type == CMD_TYPE_ARC:
        end_point = (row[COL_X], row[COL_Y], row[COL_Z])
        center_offset = (row[COL_I], row[COL_J])
        clockwise = bool(row[COL_CW])
        return ("ARC", end_point, center_offset, clockwise)

    if cmd_type == CMD_TYPE_BEZIER:
        end_point = (row[COL_X], row[COL_Y], row[COL_Z])
        c1 = (row[COL_C1X], row[COL_C1Y])
        c2 = (row[COL_C2X], row[COL_C2Y])
        return ("BEZIER", end_point, c1, c2)

    return None


def _are_segments_equal(
    key1: Tuple,
    key2: Tuple,
    tolerance: float,
) -> bool:
    """Check if two segment keys represent identical segments."""
    if key1[0] != key2[0]:
        return False

    seg_type = key1[0]

    if seg_type == "LINE":
        return _are_points_equal(key1[1], key2[1], tolerance)

    if seg_type == "ARC":
        return (
            _are_points_equal(key1[1], key2[1], tolerance)
            and _are_points_equal(key1[2], key2[2], tolerance)
            and key1[3] == key2[3]
        )

    if seg_type == "BEZIER":
        return (
            _are_points_equal(key1[1], key2[1], tolerance)
            and _are_points_equal(key1[2], key2[2], tolerance)
            and _are_points_equal(key1[3], key2[3], tolerance)
        )

    return False


def remove_duplicate_segments(
    data: Optional[np.ndarray],
    tolerance: float = 1e-6,
) -> Optional[np.ndarray]:
    """
    Remove duplicate segments from geometry data.

    This function identifies and removes duplicate line, arc, and bezier
    segments. A segment is considered duplicate if it has the same type,
    endpoint, and parameters as another segment within the same path.

    Move commands are preserved as they define path boundaries and reset
    duplicate detection for new paths.

    Parameters
    ----------
    data : np.ndarray
        Geometry data array of shape (N, 8) where each row represents
        a command. Columns are [type, x, y, z, param1, param2, param3,
        param4].
    tolerance : float, optional
        Maximum distance for two points to be considered equal.
        Default is 1e-6.

    Returns
    -------
    np.ndarray
        Geometry data array with duplicate segments removed.

    Examples
    --------
    >>> import numpy as np
    >>> from rayforge.core.geo.constants import CMD_TYPE_MOVE, CMD_TYPE_LINE
    >>> data = np.array([
    ...     [CMD_TYPE_MOVE, 0, 0, 0, 0, 0, 0, 0],
    ...     [CMD_TYPE_LINE, 10, 0, 0, 0, 0, 0, 0],
    ...     [CMD_TYPE_LINE, 10, 0, 0, 0, 0, 0, 0],
    ... ])
    >>> result = remove_duplicate_segments(data)
    >>> len(result)
    2
    """
    if data is None or len(data) == 0:
        return data

    keep_mask = np.ones(len(data), dtype=bool)
    seen_segments: List[Tuple] = []

    for i in range(len(data)):
        row = data[i]
        cmd_type = row[COL_TYPE]

        if cmd_type == CMD_TYPE_MOVE:
            seen_segments = []
            continue

        key = _get_segment_key(data, i, tolerance)

        if key is None:
            continue

        is_duplicate = False
        for seen_key in seen_segments:
            if _are_segments_equal(key, seen_key, tolerance):
                is_duplicate = True
                break

        if is_duplicate:
            keep_mask[i] = False
        else:
            seen_segments.append(key)

    return data[keep_mask]


def close_geometry_gaps_from_array(
    data: np.ndarray, tolerance: float = 1e-6
) -> np.ndarray:
    """
    Closes small gaps in a geometry array to form clean, connected paths.

    Args:
        data: The input geometry numpy array.
        tolerance: The maximum distance between two points to be
                    considered "the same".

    Returns:
        A new, modified numpy array.
    """
    if data is None or len(data) < 2:
        return data if data is not None else np.array([])

    modified_data = data.copy()
    move_indices = np.where(modified_data[:, COL_TYPE] == CMD_TYPE_MOVE)[0]
    sub_arrays = np.split(modified_data, move_indices[1:])

    for sub in sub_arrays:
        if len(sub) < 2:
            continue
        start_pt = sub[0, COL_X : COL_Z + 1]
        end_pt = sub[-1, COL_X : COL_Z + 1]
        dist_sq = np.sum((start_pt - end_pt) ** 2)
        if dist_sq < tolerance * tolerance:
            sub[-1, COL_X : COL_Z + 1] = start_pt
    modified_data = np.vstack(sub_arrays)

    final_rows: List[np.ndarray] = []
    last_end_point: np.ndarray | None = None
    for row in modified_data:
        cmd_type = row[COL_TYPE]
        end_point = row[COL_X : COL_Z + 1]

        if cmd_type == CMD_TYPE_MOVE:
            if last_end_point is not None:
                dist_sq = np.sum((end_point - last_end_point) ** 2)
                if dist_sq < tolerance * tolerance:
                    new_row = row.copy()
                    new_row[COL_TYPE] = CMD_TYPE_LINE
                    new_row[COL_X : COL_Z + 1] = last_end_point
                    final_rows.append(new_row)
                else:
                    final_rows.append(row)
                    last_end_point = end_point
            else:
                final_rows.append(row)
                last_end_point = end_point
        else:
            final_rows.append(row)
            last_end_point = end_point

    if not final_rows:
        return np.array([])
    return np.array(final_rows)


def close_geometry_gaps(
    geometry: Geometry, tolerance: float = 1e-6
) -> Geometry:
    """
    Closes small gaps in a Geometry object to form clean, connected paths.

    This function creates a new Geometry object with the modifications.

    Args:
        geometry: The input Geometry object.
        tolerance: The maximum distance between two points to be
                    considered "the same".

    Returns:
        A new, modified Geometry object.
    """
    new_geo = geometry.copy()
    new_geo._sync_to_numpy()
    if new_geo.is_empty() or new_geo._data is None:
        return new_geo

    new_geo._data = close_geometry_gaps_from_array(
        new_geo._data, tolerance=tolerance
    )
    return new_geo
