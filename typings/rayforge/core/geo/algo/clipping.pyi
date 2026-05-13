"""Line-segment clipping against rectangles and polygons."""

from typing import Optional, List, Tuple, Union, TypeAlias

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo import Point, Point3D, Polygon

_Segment3D = Tuple[Point3D, Point3D]
_PolygonsInput: TypeAlias = Union[
    List[Polygon],
    List[NDArray[np.float64]],
]

def clip_line_segment_with_rect(
    p1: Point3D,
    p2: Point3D,
    rect: Tuple[float, float, float, float],
) -> Optional[_Segment3D]:
    """Clip a 3D line segment to a 2D axis-aligned rectangle.

    Args:
        p1: Start point ``(x, y, z)``.
        p2: End point ``(x, y, z)``.
        rect: ``(x_min, y_min, x_max, y_max)``.

    Returns:
        The clipped segment ``(start, end)`` or ``None`` if the segment
        falls entirely outside the rectangle.
    """
    ...


def subtract_polygons_from_line_segment(
    p1: Point3D,
    p2: Point3D,
    regions: _PolygonsInput,
) -> List[_Segment3D]:
    """Subtract polygon regions from a line segment.

    Returns only the portions of the segment that are **not** inside any
    of the given polygons.

    Args:
        p1: Start point ``(x, y, z)``.
        p2: End point ``(x, y, z)``.
        regions: Polygons defining the areas to subtract.  Each polygon
            may be a list of ``(x, y)`` tuples or an ``(N, 2)`` NumPy
            array.

    Returns:
        Remaining segments after subtraction.
    """
    ...


def clip_line_segment_with_polygons(
    p1: Point3D,
    p2: Point3D,
    regions: _PolygonsInput,
) -> List[_Segment3D]:
    """Clip a line segment to keep only the portions inside polygons.

    Args:
        p1: Start point ``(x, y, z)``.
        p2: End point ``(x, y, z)``.
        regions: Polygons defining the clipping regions.  Each polygon
            may be a list of ``(x, y)`` tuples or an ``(N, 2)`` NumPy
            array.

    Returns:
        Segments that fall inside at least one polygon.
    """
    ...
