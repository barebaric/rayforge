import numpy as np
from typing import List, Tuple
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    GEO_ARRAY_COLS,
    COL_TYPE,
)


def _ramer_douglas_peucker_numpy(
    points: np.ndarray, tolerance: float
) -> np.ndarray:
    """
    Vectorized Iterative Ramer-Douglas-Peucker using NumPy.
    """
    n = len(points)
    if n < 3:
        return points

    # Boolean mask of points to keep
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[n - 1] = True

    # Iterative stack to avoid recursion depth issues
    # Stack stores (start_index, end_index)
    stack: List[Tuple[int, int]] = [(0, n - 1)]

    while stack:
        start, end = stack.pop()

        # If segment is too small, skip
        if end - start < 2:
            continue

        # Get the segment endpoints (using only X,Y for calculation)
        p_start = points[start, :2]
        p_end = points[end, :2]

        # Vector of the chord
        chord_vec = p_end - p_start
        chord_len_sq = np.dot(chord_vec, chord_vec)

        # Points to check in this range
        # Note: slicing creates a view, so this is efficient
        check_points = points[start + 1 : end, :2]

        if chord_len_sq < 1e-12:
            # Start and End are practically the same
            # Dist is Euclidean dist from start
            dists_sq = np.sum((check_points - p_start) ** 2, axis=1)
        else:
            # Vectorized Perpendicular Distance
            # Distance = |CrossProduct(v_start_to_pt, chord)| / |chord|
            v_start_to_pts = check_points - p_start

            # 2D Cross Product: x1*y2 - x2*y1
            cross_prod = (
                v_start_to_pts[:, 0] * chord_vec[1]
                - v_start_to_pts[:, 1] * chord_vec[0]
            )

            # d^2 = cross^2 / chord^2
            dists_sq = (cross_prod**2) / chord_len_sq

        # Find max distance
        # argmax returns index relative to the sliced view
        max_idx_local = np.argmax(dists_sq)
        max_dist_sq = dists_sq[max_idx_local]

        if max_dist_sq > (tolerance * tolerance):
            # Convert local index back to global index
            # check_points started at start+1
            max_idx_global = start + 1 + int(max_idx_local)

            keep[max_idx_global] = True

            # Push sub-segments
            # Explicitly cast to int to satisfy type checkers against np.int64
            stack.append((int(start), int(max_idx_global)))
            stack.append((int(max_idx_global), int(end)))

    return points[keep]


def simplify_points_to_array(
    points: np.ndarray, tolerance: float
) -> np.ndarray:
    """
    Simplifies a numpy array of 2D points using the Ramer-Douglas-Peucker
    algorithm. Returns a numpy array of the kept points.
    """
    return _ramer_douglas_peucker_numpy(points, tolerance)


def simplify_points(
    points: List[Tuple[float, float]], tolerance: float
) -> List[Tuple[float, float]]:
    """
    Simplifies a list of 2D points using the Ramer-Douglas-Peucker algorithm.
    This bypasses Command object creation for raw buffer processing.
    """
    if len(points) < 3:
        return points

    # Convert list of tuples to numpy array
    arr = np.array(points, dtype=np.float64)
    simplified_arr = _ramer_douglas_peucker_numpy(arr, tolerance)

    # Convert back to list of tuples
    # simplify_arr is (N, 2)
    return [tuple(p) for p in simplified_arr.tolist()]  # type: ignore


def simplify_geometry_from_array(
    data: np.ndarray, tolerance: float
) -> np.ndarray:
    """
    Simplifies a geometry numpy array by applying RDP to linear chains.
    """
    if data is None or len(data) == 0:
        return np.array([])

    simplified_rows: List[np.ndarray] = []
    point_chain: List[np.ndarray] = []

    def flush_chain():
        nonlocal point_chain
        if len(point_chain) > 1:
            points_arr = np.array(point_chain)
            simplified_arr = _ramer_douglas_peucker_numpy(
                points_arr, tolerance
            )

            # Reconstruct LineTo commands for simplified points
            for p in simplified_arr[1:]:
                row = np.zeros(GEO_ARRAY_COLS)
                row[COL_TYPE] = CMD_TYPE_LINE
                row[1:4] = p
                simplified_rows.append(row)
        point_chain = []

    last_pos = np.array([0.0, 0.0, 0.0])

    for row in data:
        cmd_type = row[COL_TYPE]
        end_pos = row[1:4]

        if cmd_type == CMD_TYPE_MOVE:
            flush_chain()
            simplified_rows.append(row)
            last_pos = end_pos
            point_chain = [last_pos]
        elif cmd_type == CMD_TYPE_LINE:
            if not point_chain:
                point_chain = [last_pos]
            point_chain.append(end_pos)
            last_pos = end_pos
        elif cmd_type == CMD_TYPE_ARC:
            flush_chain()
            simplified_rows.append(row)
            last_pos = end_pos
            point_chain = [last_pos]
        else:
            flush_chain()
            simplified_rows.append(row)

    flush_chain()

    if not simplified_rows:
        return np.array([]).reshape(0, GEO_ARRAY_COLS)

    return np.array(simplified_rows)
