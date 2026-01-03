import numpy as np
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from .geometry import Command


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


def simplify_geometry(
    commands: List["Command"], tolerance: float = 0.01
) -> List["Command"]:
    """
    Reduces the number of commands in a command list using the
    Ramer-Douglas-Peucker algorithm. Uses NumPy for performance.

    Args:
        commands: The input list of geometric commands.
        tolerance: The maximum allowed perpendicular distance (in mm) from the
                   simplified line to the original points.

    Returns:
        A new list of simplified commands.
    """
    from .geometry import (
        MoveToCommand,
        LineToCommand,
        ArcToCommand,
        MovingCommand,
    )

    if not commands:
        return []

    simplified_commands: List["Command"] = []

    # We collect points for a continuous linear chain here
    current_chain: List[Tuple[float, float, float]] = []

    def flush_chain():
        nonlocal current_chain
        if len(current_chain) > 1:
            # Always use NumPy Acceleration
            arr = np.array(current_chain, dtype=np.float64)
            final_points_arr = _ramer_douglas_peucker_numpy(arr, tolerance)

            # Reconstruct LineTo commands
            # Skip the first point as it's the anchor (MoveTo or previous end)
            for p in final_points_arr[1:]:
                # Convert numpy row to standard list for cleaner indexing
                # checks
                p_list = p.tolist()

                if len(p_list) >= 3:
                    simplified_commands.append(
                        LineToCommand((p_list[0], p_list[1], p_list[2]))
                    )
                elif len(p_list) >= 2:
                    # Fallback if numpy array was only 2D (x, y)
                    simplified_commands.append(
                        LineToCommand((p_list[0], p_list[1], 0.0))
                    )

        current_chain = []

    last_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    for cmd in commands:
        if isinstance(cmd, MoveToCommand):
            # A MoveTo breaks any current chain
            flush_chain()
            simplified_commands.append(cmd)
            if cmd.end:
                last_pos = cmd.end
                # Start a new potential chain
                current_chain = [last_pos]

        elif isinstance(cmd, LineToCommand):
            if cmd.end:
                if not current_chain:
                    # Should ideally not happen if malformed geometry starts
                    # with LineTo, but we assume (0,0) start.
                    current_chain = [last_pos]

                current_chain.append(cmd.end)
                last_pos = cmd.end

        elif isinstance(cmd, ArcToCommand):
            # Arcs break the simplification chain.
            flush_chain()
            simplified_commands.append(cmd)
            if cmd.end:
                last_pos = cmd.end
                # Start a new potential chain after the arc
                current_chain = [last_pos]

        elif isinstance(cmd, MovingCommand):
            # Fallback for generic moving commands - treat as break
            flush_chain()
            simplified_commands.append(cmd)
            if cmd.end:
                last_pos = cmd.end
                current_chain = [last_pos]
        else:
            # Non-moving commands (state, markers) are preserved
            flush_chain()
            simplified_commands.append(cmd)

    flush_chain()
    return simplified_commands
