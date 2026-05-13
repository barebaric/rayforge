"""Minkowski sum and No-Fit Polygon algorithms for 2D packing."""

from typing import List, Tuple, TypeAlias, Union

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo import Point

_PolygonsInput: TypeAlias = Union[
    List[List[Point]],
    List[NDArray[np.float64]],
]

def get_polygon_minkowski_sum_convex(
    poly_a: List[Tuple[int, int]],
    poly_b: List[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """Compute the Minkowski sum of two **convex** integer polygons.

    Args:
        poly_a: First convex polygon as integer ``(x, y)`` vertices.
        poly_b: Second convex polygon as integer ``(x, y)`` vertices.

    Returns:
        List of result polygons (typically one for convex inputs).
    """
    ...


def get_inner_fit_polygon(
    outer: List[Point],
    inner: List[Point],
) -> List[List[Point]]:
    """Compute the Inner Fit Polygon (IFP) for 2D nesting.

    The IFP represents all valid placements of *inner* inside *outer*
    such that *inner* is fully contained.

    Args:
        outer: Outer boundary polygon.
        inner: Polygon to place inside.

    Returns:
        List of IFP polygons.
    """
    ...


def get_no_fit_polygon(
    subject: List[Point],
    tool: List[Point],
) -> List[List[Point]]:
    """Compute the No-Fit Polygon (NFP) for 2D nesting.

    The NFP describes all relative positions of *tool* that cause it
    to touch but not overlap *subject*.

    Args:
        subject: The stationary polygon.
        tool: The polygon being placed.

    Returns:
        List of NFP polygons.
    """
    ...


def calculate_input_scale(
    polygons: _PolygonsInput,
    max_int: int = ...,
) -> float:
    """Calculate an appropriate scale factor for integer-based
    computations.

    Ensures that scaled polygon coordinates stay within the integer
    range ``[-max_int, max_int]``.

    Args:
        polygons: Polygons to scale.  Each polygon may be a list of
            ``(x, y)`` tuples or an ``(N, 2)`` NumPy array.
        max_int: Maximum integer value.  Defaults to ``2147483647``.

    Returns:
        A float scale factor.
    """
    ...


def convolve_two_segments(
    a1: Tuple[int, int],
    a2: Tuple[int, int],
    b1: Tuple[int, int],
    b2: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Convolve two integer line segments (Minkowski sum of edges).

    Args:
        a1: Start of segment A.
        a2: End of segment A.
        b1: Start of segment B.
        b2: End of segment B.

    Returns:
        Resulting integer vertices.
    """
    ...


def convolve_point_sequences(
    seq_a: List[Tuple[int, int]],
    seq_b: List[Tuple[int, int]],
) -> List[List[Tuple[int, int]]]:
    """Convolve two integer point sequences (Minkowski sum).

    Args:
        seq_a: First point sequence.
        seq_b: Second point sequence.

    Returns:
        List of result polygons.
    """
    ...
