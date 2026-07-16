"""
Array / Pattern tool strategies and parameters.

This package is purely geometric: it computes world-space transformation
deltas for array arrangements. Document mutation lives in
:class:`rayforge.doceditor.array_cmd.ArrayCmd`.
"""

from raygeo.geo.types import Rect

from .base import ArrayStrategy
from .circular import CircularArrayStrategy
from .grid import GridArrayStrategy
from .params import (
    ArrayMode,
    ArrayParams,
    CircularArrayParams,
    GridArrayParams,
    PointRotationParams,
    SpacingMode,
)
from .point_rotation import PointRotationStrategy

__all__ = [
    "ArrayMode",
    "ArrayParams",
    "ArrayStrategy",
    "CircularArrayParams",
    "CircularArrayStrategy",
    "GridArrayParams",
    "GridArrayStrategy",
    "PointRotationParams",
    "PointRotationStrategy",
    "SpacingMode",
    "make_array_strategy",
]


def make_array_strategy(unit_bbox: Rect, params: ArrayParams) -> ArrayStrategy:
    """Factory that builds the strategy for an :class:`ArrayParams`."""
    if params.mode == ArrayMode.POINT_ROTATION:
        return PointRotationStrategy(unit_bbox, params.point_rotation)
    if params.mode == ArrayMode.CIRCULAR:
        return CircularArrayStrategy(unit_bbox, params.circular)
    return GridArrayStrategy(unit_bbox, params.grid)
