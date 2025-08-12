from __future__ import annotations
from enum import Enum, auto
from typing import Tuple, Union


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
    scale_compensation: Union[float, Tuple[float, float]] = 1.0,
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
        scale_compensation: The scale factor(s) of the context. Can be a
                            single float for uniform scale or a tuple (sx, sy)
                            for non-uniform scale to ensure handles remain
                            square.
    """
    w, h = width, height

    if isinstance(scale_compensation, tuple):
        scale_x, scale_y = scale_compensation
    else:
        scale_x = scale_y = scale_compensation

    # Avoid division by zero for extremely small or invalid scales
    if abs(scale_x) < 1e-6 or abs(scale_y) < 1e-6:
        return (0.0, 0.0, 0.0, 0.0)

    # Calculate local handle dimensions by dividing the desired
    # visual size by the scale factors.
    local_handle_w = base_handle_size / scale_x
    local_handle_h = base_handle_size / scale_y

    # Dynamically calculate handle size to prevent overlap on small elements.
    effective_hw = min(local_handle_w, w / 3.0)
    effective_hh = min(local_handle_h, h / 3.0)

    avg_scale = (scale_x + scale_y) / 2.0

    if region == ElementRegion.ROTATION_HANDLE:
        handle_dist = 20.0 / avg_scale  # Keep distance visually constant
        cx = w / 2.0
        # The rotation handle also uses the effective sizes to appear square.
        return (
            cx - effective_hw / 2.0,
            -handle_dist - effective_hh,
            effective_hw,
            effective_hh,
        )

    # Corner regions are rectangles that will appear square after scaling
    if region == ElementRegion.TOP_LEFT:
        return 0.0, 0.0, effective_hw, effective_hh
    if region == ElementRegion.TOP_RIGHT:
        return w - effective_hw, 0.0, effective_hw, effective_hh
    if region == ElementRegion.BOTTOM_LEFT:
        return 0.0, h - effective_hh, effective_hw, effective_hh
    if region == ElementRegion.BOTTOM_RIGHT:
        return (
            w - effective_hw,
            h - effective_hh,
            effective_hw,
            effective_hh,
        )

    # Edge regions are between the corners
    if region == ElementRegion.TOP_MIDDLE:
        return effective_hw, 0.0, w - 2.0 * effective_hw, effective_hh
    if region == ElementRegion.BOTTOM_MIDDLE:
        return (
            effective_hw,
            h - effective_hh,
            w - 2.0 * effective_hw,
            effective_hh,
        )
    if region == ElementRegion.MIDDLE_LEFT:
        return 0.0, effective_hh, effective_hw, h - 2.0 * effective_hh
    if region == ElementRegion.MIDDLE_RIGHT:
        return (
            w - effective_hw,
            effective_hh,
            effective_hw,
            h - 2.0 * effective_hh,
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
    scale_compensation: Union[float, Tuple[float, float]] = 1.0,
) -> ElementRegion:
    """
    Checks which interactive region is hit by a point in LOCAL coordinates.
    This function does not need to handle rotation or translation.
    """
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
        # Use the provided scale compensation to calculate the hit rectangle.
        # This ensures the hit-test area matches the rendered handle size.
        rx, ry, rw, rh = get_region_rect(
            region, width, height, base_handle_size, scale_compensation
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
