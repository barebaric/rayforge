import numpy as np
import cv2
from typing import Tuple, List
from ..core.geo import Geometry


def get_enclosing_hull(
    boolean_image: np.ndarray,
    scale_x: float,
    scale_y: float,
    height_px: int,
    border_size: int,
) -> List[Geometry]:
    """
    Calculates a single convex hull that encompasses all content in the image.

    This is often used as a fallback when more complex tracing methods fail or
    are not desired.

    Args:
        boolean_image: The boolean image containing all shapes.
        scale_x: Pixels per millimeter (X).
        scale_y: Pixels per millimeter (Y).
        height_px: Original height of the source surface in pixels.
        border_size: The size of the border added during pre-processing, which
                     must be accounted for in coordinate transformation.

    Returns:
        A list containing a single Geometry object for the enclosing hull,
        or an empty list if no content was found.
    """
    img_uint8 = boolean_image.astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        img_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Combine all points from all contours into a single array
    all_points = np.vstack(contours)
    if len(all_points) < 3:
        return []

    hull_points = cv2.convexHull(all_points).squeeze(axis=1)

    def _transform_point(p: Tuple[float, float]) -> Tuple[float, float]:
        px, py = p
        ops_px = px - border_size
        ops_py = height_px - (py - border_size)
        return ops_px / scale_x, ops_py / scale_y

    geo = Geometry()
    start_pt = _transform_point(tuple(hull_points[0]))
    geo.move_to(start_pt[0], start_pt[1])

    for point in hull_points[1:]:
        pt = _transform_point(tuple(point))
        geo.line_to(pt[0], pt[1])

    geo.close_path()
    return [geo]
