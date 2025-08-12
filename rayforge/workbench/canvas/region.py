from __future__ import annotations
from enum import Enum, auto
from typing import Tuple


class ElementRegion(Enum):
    """Defines interactive regions for selection frames."""

    NONE = auto()
    BODY = auto()
    TOP_LEFT = auto()
    TOP_MIDDLE = auto()
    TOP_RIGHT = auto()
    MIDDLE_LEFT = auto()
    MIDDLE_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_MIDDLE = auto()
    BOTTOM_RIGHT = auto()
    ROTATION_HANDLE = auto()


def get_region_rect(
    region: ElementRegion,
    width: float,
    height: float,
    base_handle_size: float,
    max_handle_size: float = 20.0,
    scale_compensation: float = 1.0,
) -> Tuple[float, float, float, float]:
    """
    A generic function to calculate the rectangle (x, y, w, h) for a given
    region, relative to a bounding box of a given width and height.

    It compensates for scale to keep handle sizes visually consistent.

    Args:
        region: The ElementRegion to calculate.
        width: The width of the bounding box.
        height: The height of the bounding box.
        base_handle_size: The desired base size of the handles in pixels.
        max_handle_size: The maximum visual size of the handles in pixels.
        scale_compensation: The scale factor of the context the handles
                            will be drawn on.

    Returns:
        A tuple (x, y, width, height) for the region's rectangle.
    """
    w, h = width, height
    # To keep the visual size constant, we calculate the required size in
    # the element's local space.
    # Formula: local_size = min(base_size, max_size / scale)
    # This caps the visual growth while allowing shrinking.
    local_handle_size = min(
        base_handle_size, max_handle_size / scale_compensation
    )

    # Dynamically calculate handle size to prevent overlap on small elements.
    # If the element is very small, this can result in a size of 0, which
    # correctly hides the handles.
    effective_hs = min(local_handle_size, w / 3.0, h / 3.0)

    if region == ElementRegion.ROTATION_HANDLE:
        handle_dist = 20.0 / scale_compensation  # Keep distance constant too
        cx = w / 2.0
        # The rotation handle also uses the effective size to scale down.
        return (
            cx - effective_hs / 2.0,
            -handle_dist - effective_hs,
            effective_hs,
            effective_hs,
        )

    # Corner regions are effective_hs x effective_hs squares
    if region == ElementRegion.TOP_LEFT:
        return 0.0, 0.0, effective_hs, effective_hs
    if region == ElementRegion.TOP_RIGHT:
        return w - effective_hs, 0.0, effective_hs, effective_hs
    if region == ElementRegion.BOTTOM_LEFT:
        return 0.0, h - effective_hs, effective_hs, effective_hs
    if region == ElementRegion.BOTTOM_RIGHT:
        return (
            w - effective_hs,
            h - effective_hs,
            effective_hs,
            effective_hs,
        )

    # Edge regions are between the corners
    if region == ElementRegion.TOP_MIDDLE:
        return effective_hs, 0.0, w - 2.0 * effective_hs, effective_hs
    if region == ElementRegion.BOTTOM_MIDDLE:
        return (
            effective_hs,
            h - effective_hs,
            w - 2.0 * effective_hs,
            effective_hs,
        )
    if region == ElementRegion.MIDDLE_LEFT:
        return 0.0, effective_hs, effective_hs, h - 2.0 * effective_hs
    if region == ElementRegion.MIDDLE_RIGHT:
        return (
            w - effective_hs,
            effective_hs,
            effective_hs,
            h - 2.0 * effective_hs,
        )

    if region == ElementRegion.BODY:
        return 0.0, 0.0, w, h

    return 0.0, 0.0, 0.0, 0.0  # For NONE or other cases


def check_region_hit(
    local_x: float,
    local_y: float,
    width: float,
    height: float,
    base_handle_size: float,
) -> ElementRegion:
    """
    Checks which interactive region is hit by a point in LOCAL coordinates.
    This function does not need to handle rotation or translation.
    """
    # For simplicity in this context, we will use the base size for hit
    # detection, as it provides a consistent target for the user.
    # A more complex implementation might pass the scale here as well.
    regions_to_check = [
        ElementRegion.ROTATION_HANDLE,
        ElementRegion.TOP_LEFT,
        ElementRegion.TOP_RIGHT,
        ElementRegion.BOTTOM_LEFT,
        ElementRegion.BOTTOM_RIGHT,
        ElementRegion.TOP_MIDDLE,
        ElementRegion.BOTTOM_MIDDLE,
        ElementRegion.MIDDLE_LEFT,
        ElementRegion.MIDDLE_RIGHT,
    ]
    for region in regions_to_check:
        # Use a fixed scale for hit testing for a consistent feel
        rx, ry, rw, rh = get_region_rect(
            region, width, height, base_handle_size
        )
        if (
            rw > 0
            and rh > 0
            and rx <= local_x < rx + rw
            and ry <= local_y < ry + rh
        ):
            return region

    # If no handle is hit, check the body (the entire 0,0,w,h area).
    if 0 <= local_x < width and 0 <= local_y < height:
        return ElementRegion.BODY

    return ElementRegion.NONE
