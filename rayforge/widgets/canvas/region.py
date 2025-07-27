from __future__ import annotations
import math
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
    region: ElementRegion, width: float, height: float, base_handle_size: float
) -> Tuple[float, float, float, float]:
    """
    A generic function to calculate the rectangle (x, y, w, h) for a given
    region, relative to a bounding box of a given width and height.

    Args:
        region: The ElementRegion to calculate.
        width: The width of the bounding box.
        height: The height of the bounding box.
        base_handle_size: The desired size of the handles in pixels.

    Returns:
        A tuple (x, y, width, height) for the region's rectangle.
    """
    w, h = width, height
    # Dynamically calculate handle size to prevent overlap on small elements.
    # If the element is very small, this can result in a size of 0, which
    # correctly hides the handles.
    effective_hs = min(base_handle_size, w / 3.0, h / 3.0)

    if region == ElementRegion.ROTATION_HANDLE:
        handle_dist = 20.0
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
    x: float,
    y: float,
    bounding_box_x: float,
    bounding_box_y: float,
    width: float,
    height: float,
    angle: float,
    center_x: float,
    center_y: float,
    base_handle_size: float,
) -> ElementRegion:
    """
    Generic function to check which interactive region is hit by a point.
    It accounts for the object's rotation.

    Args:
        x: The absolute x-coordinate of the hit point (e.g., mouse cursor).
        y: The absolute y-coordinate of the hit point.
        bounding_box_x: The absolute x-coordinate of the object's bounding box
         top-left.
        bounding_box_y: The absolute y-coordinate of the object's bounding box
         top-left.
        width: The width of the object's bounding box.
        height: The height of the object's bounding box.
        angle: The rotation of the object in degrees.
        center_x: The absolute x-coordinate of the object's rotation center.
        center_y: The absolute y-coordinate of the object's rotation center.
        base_handle_size: The desired size of the handles in pixels.

    Returns:
        The ElementRegion that was hit.
    """
    # 1. Un-rotate the hit point if there's an angle
    if angle != 0:
        angle_rad = math.radians(-angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        rot_x = center_x + (x - center_x) * cos_a - (y - center_y) * sin_a
        rot_y = center_y + (x - center_x) * sin_a + (y - center_y) * cos_a
        x, y = rot_x, rot_y

    # 2. Convert absolute, un-rotated coordinates to be local to the bounding
    # box
    local_x, local_y = x - bounding_box_x, y - bounding_box_y

    # 3. Check handles first
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

    # 4. If no handle is hit, check the body
    bx, by, bw, bh = get_region_rect(
        ElementRegion.BODY, width, height, base_handle_size
    )
    if bx <= local_x < bx + bw and by <= local_y < by + bh:
        return ElementRegion.BODY

    return ElementRegion.NONE
