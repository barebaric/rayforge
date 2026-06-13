from typing import List, Optional

import numpy as np
from raygeo import Geometry
from raygeo.geo.algo import hull as _hull


def _transform_geometry(
    geo: Geometry,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
) -> Geometry:
    """
    Transform a Geometry's vertex coordinates from pixel space to
    millimeter space, applying scaling, Y-axis inversion, and border
    offset.
    """
    data = geo.data
    if len(data) < 2:
        return geo

    new_geo = Geometry()
    for i, cmd in enumerate(data[:-1]):
        px = cmd.end[0] - border_size
        py = height_px - (cmd.end[1] - border_size)
        x = px / scale_x
        y = py / scale_y
        if i == 0:
            new_geo.move_to(x, y)
        else:
            new_geo.line_to(x, y)
    new_geo.close_path()
    return new_geo


def get_enclosing_hull(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
) -> Optional[Geometry]:
    """
    Calculates a single convex hull that encompasses all content in the image.

    Delegates to the raygeo Rust backend for contour tracing and convex hull
    computation, then transforms pixel coordinates to millimeter space.
    """
    geo = _hull.get_enclosing_hull(boolean_image)
    if geo is None:
        return None
    return _transform_geometry(geo, scale_x, scale_y, height_px, border_size)


def get_hulls_from_image(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
) -> List[Geometry]:
    """
    Finds all distinct contours in a boolean image, calculates the convex
    hull for each, and returns them as a list of Geometry objects.

    Args:
        boolean_image: The clean boolean image containing only major shapes.
        scale_x: Pixels per millimeter (X).
        scale_y: Pixels per millimeter (Y).
        height_px: Original height of the source surface in pixels.
        border_size: The pixel border size added during pre-processing.

    Returns:
        A list of Geometry objects, each representing a convex hull.
    """
    geometries = _hull.get_hulls_from_image(boolean_image)
    return [
        _transform_geometry(geo, scale_x, scale_y, height_px, border_size)
        for geo in geometries
    ]


def get_concave_hull(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
    gravity: float = 0.1,
) -> Optional[Geometry]:
    """
    Calculates a smooth, constrained concave hull that "shrink-wraps" the
    content geometrically, mimicking a physical rubber band using Bézier
    curves.

    Delegates to the raygeo Rust backend for the full algorithm, then
    transforms pixel coordinates to millimeter space.
    """
    geo = _hull.get_concave_hull(boolean_image, gravity)
    if geo is None:
        return None
    return _transform_geometry(geo, scale_x, scale_y, height_px, border_size)
