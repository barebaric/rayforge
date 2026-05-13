"""Polyline simplification using the Douglas-Peucker algorithm."""

from typing import List, Sequence, Tuple, TypeAlias, Union

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo import Point

_ArrayLike: TypeAlias = Union[List[List[float]], NDArray[np.float64]]


def simplify_polyline(
    points: Sequence[Tuple[float, ...]],
    tolerance: float,
) -> List[Tuple[float, ...]]:
    """Simplify a 2D polyline using Douglas-Peucker.

    Points are projected to ``z=0`` for the computation.

    Args:
        points: Ordered ``(x, y)`` vertices.
        tolerance: Maximum perpendicular distance a point may have from
            the simplified line.

    Returns:
        Simplified polyline as a list of ``(x, y)`` vertices.
    """
    ...


def simplify_polyline_to_array(
    data: _ArrayLike,
    tolerance: float,
) -> List[List[float]]:
    """Simplify a polyline stored as a 2D array of coordinates.

    Each row is ``[x, y, ...]``; extra columns are preserved in the
    output.

    Args:
        data: List of coordinate rows (at least 2 elements each).
        tolerance: Maximum perpendicular deviation.

    Returns:
        Simplified coordinate rows with the same column count.
    """
    ...
