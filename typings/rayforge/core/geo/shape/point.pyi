"""Single-point utility functions."""

from typing import Tuple, Union

from rayforge.core.geo import Point3D


def midpoint(
    p1: Union[Tuple[float, float], Tuple[float, float, float]],
    p2: Union[Tuple[float, float], Tuple[float, float, float]],
) -> Point3D:
    """Compute the midpoint between two 3D points.

    Accepts 2D or 3D tuples; 2D inputs are treated as ``z=0``.

    Args:
        p1: First point ``(x, y)`` or ``(x, y, z)``.
        p2: Second point ``(x, y)`` or ``(x, y, z)``.

    Returns:
        The midpoint ``(x, y, z)``.

    Raises:
        ValueError: If a point is not a 2- or 3-tuple of floats.
    """
    ...
