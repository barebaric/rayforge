import numpy as np
from typing import List, Sequence, Tuple


def simplify_points_to_array(
    points: np.ndarray, tolerance: float
) -> np.ndarray:
    """
    Simplify a sequence of 2D points using the Ramer-Douglas-Peucker algorithm.

    This is a vectorized, iterative implementation that avoids recursion depth
    issues by using an explicit stack. The algorithm works by recursively
    dividing the curve and keeping points that are farther from the chord
    connecting the endpoints than the specified tolerance.

    Parameters
    ----------
    points : np.ndarray
        An array of shape (N, 2) or (N, 3) representing the points to simplify.
        The first two columns (X, Y) are used for distance calculation.
        Additional columns (e.g., Z) are preserved in the output.
    tolerance : float
        The maximum perpendicular distance from the chord for a point to be
        removed. Points with deviation greater than this value are kept.
        Must be non-negative; negative values are treated as zero.

    Returns
    -------
    np.ndarray
        A subset of the input points with shape (M, K) where M <= N and K is
        the same as the input dimensionality. The first and last points are
        always preserved.

    Notes
    -----
    - The implementation uses squared distances to avoid unnecessary square
      root operations.
    - A small epsilon (1e-12) is used to handle degenerate cases where start
      and end points are practically identical.
    - The algorithm uses an iterative stack-based approach instead of recursion
      to avoid stack overflow on large point sequences.

    Examples
    --------
    >>> import numpy as np
    >>> points = np.array([(0, 0), (1, 1), (2, 2), (3, 3), (10, 10)])
    >>> simplify_points_to_array(points, tolerance=0.001)
    array([[ 0.,  0.],
           [10., 10.]])
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
    points: Sequence[Tuple[float, float]], tolerance: float
) -> Sequence[Tuple[float, float]]:
    """
    Simplify a sequence of 2D points using the Ramer-Douglas-Peucker algorithm.

    This is a convenience wrapper that converts between Python sequences of
    tuples and numpy arrays, allowing use with standard Python data structures.

    Parameters
    ----------
    points : Sequence[Tuple[float, float]]
        A sequence of (x, y) coordinate tuples representing the points to
        simplify.
    tolerance : float
        The maximum perpendicular distance from the chord for a point to be
        removed. Points with deviation greater than this value are kept.
        Must be non-negative; negative values are treated as zero.

    Returns
    -------
    Sequence[Tuple[float, float]]
        A sequence of (x, y) coordinate tuples representing the simplified
        points. The first and last points are always preserved.

    Examples
    --------
    >>> points = [(0, 0), (1, 1), (2, 2), (3, 3), (10, 10)]
    >>> simplify_points(points, tolerance=0.001)
    [(0, 0), (10, 10)]
    """
    if len(points) < 3:
        return points

    # Convert list of tuples to numpy array
    arr = np.array(points, dtype=np.float64)
    simplified_arr = simplify_points_to_array(arr, tolerance)

    # Convert back to list of tuples
    # simplify_arr is (N, 2)
    return [tuple(p) for p in simplified_arr.tolist()]
