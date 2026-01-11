from __future__ import annotations
from enum import Enum, auto
from typing import Tuple, Union, Set, Optional


class ElementRegion(Enum):
    """Defines interactive regions for selection frames."""

    NONE = auto()
    BODY = auto()
    # Resize handles (inside)
    TOP_LEFT = auto()
    TOP_MIDDLE = auto()
    TOP_RIGHT = auto()
    MIDDLE_LEFT = auto()
    MIDDLE_RIGHT = auto()
    BOTTOM_LEFT = auto()
    BOTTOM_MIDDLE = auto()
    BOTTOM_RIGHT = auto()
    # Rotate & Shear handles (outside)
    ROTATE_TOP_LEFT = auto()
    ROTATE_TOP_RIGHT = auto()
    ROTATE_BOTTOM_LEFT = auto()
    ROTATE_BOTTOM_RIGHT = auto()
    SHEAR_TOP = auto()
    SHEAR_RIGHT = auto()
    SHEAR_BOTTOM = auto()
    SHEAR_LEFT = auto()


RESIZE_HANDLES: Set[ElementRegion] = {
    ElementRegion.TOP_LEFT,
    ElementRegion.TOP_MIDDLE,
    ElementRegion.TOP_RIGHT,
    ElementRegion.MIDDLE_LEFT,
    ElementRegion.MIDDLE_RIGHT,
    ElementRegion.BOTTOM_LEFT,
    ElementRegion.BOTTOM_MIDDLE,
    ElementRegion.BOTTOM_RIGHT,
}

BBOX_REGIONS: Set[ElementRegion] = {ElementRegion.BODY} | RESIZE_HANDLES

ROTATE_HANDLES: Set[ElementRegion] = {
    ElementRegion.ROTATE_TOP_LEFT,
    ElementRegion.ROTATE_TOP_RIGHT,
    ElementRegion.ROTATE_BOTTOM_LEFT,
    ElementRegion.ROTATE_BOTTOM_RIGHT,
}

SHEAR_HANDLES: Set[ElementRegion] = {
    ElementRegion.SHEAR_TOP,
    ElementRegion.SHEAR_RIGHT,
    ElementRegion.SHEAR_BOTTOM,
    ElementRegion.SHEAR_LEFT,
}

ROTATE_SHEAR_HANDLES: Set[ElementRegion] = ROTATE_HANDLES | SHEAR_HANDLES

LEFT_HANDLES: Set[ElementRegion] = {
    ElementRegion.TOP_LEFT,
    ElementRegion.MIDDLE_LEFT,
    ElementRegion.BOTTOM_LEFT,
}

RIGHT_HANDLES: Set[ElementRegion] = {
    ElementRegion.TOP_RIGHT,
    ElementRegion.MIDDLE_RIGHT,
    ElementRegion.BOTTOM_RIGHT,
}

TOP_HANDLES: Set[ElementRegion] = {
    ElementRegion.TOP_LEFT,
    ElementRegion.TOP_MIDDLE,
    ElementRegion.TOP_RIGHT,
}

BOTTOM_HANDLES: Set[ElementRegion] = {
    ElementRegion.BOTTOM_LEFT,
    ElementRegion.BOTTOM_MIDDLE,
    ElementRegion.BOTTOM_RIGHT,
}

CORNER_RESIZE_HANDLES: Set[ElementRegion] = (TOP_HANDLES | BOTTOM_HANDLES) & (
    LEFT_HANDLES | RIGHT_HANDLES
)

MIDDLE_RESIZE_HANDLES: Set[ElementRegion] = (
    RESIZE_HANDLES - CORNER_RESIZE_HANDLES
)


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

    It compensates for scale to keep handle sizes visually consistent and
    adapts to flipped coordinate systems by checking the sign of the
    scale_compensation.

    Args:
        region: The ElementRegion to calculate.
        width: The width of the bounding box.
        height: The height of the bounding box.
        base_handle_size: The desired base size of the handles in pixels.
        scale_compensation: The signed scale factor(s) of the context.
                            A negative y-scale indicates a flipped axis.
    """
    w, h = width, height

    if isinstance(scale_compensation, tuple):
        scale_x, scale_y = scale_compensation
    else:
        scale_x = scale_y = scale_compensation

    # Check for a flipped Y-axis BEFORE taking the absolute value.
    is_flipped_y = scale_y < 0

    # Use absolute scale for calculating handle *dimensions*.
    abs_scale_x = abs(scale_x)
    abs_scale_y = abs(scale_y)

    if abs_scale_x < 1e-6 or abs_scale_y < 1e-6:
        return (0.0, 0.0, 0.0, 0.0)

    # Calculate local handle dimensions by dividing the desired
    # visual size by the scale factors.
    local_handle_w = base_handle_size / abs_scale_x
    local_handle_h = base_handle_size / abs_scale_y

    # Dynamically calculate handle size to prevent overlap on small elements.
    effective_hw = min(local_handle_w, w / 3.0)
    effective_hh = min(local_handle_h, h / 3.0)

    # Use average scale for distance calculation of rotation handle
    avg_abs_scale = (abs_scale_x + abs_scale_y) / 2.0
    handle_dist = 5.0 / avg_abs_scale  # Visual distance for external handles

    # Conditionally calculate Y positions based on the axis orientation.
    if is_flipped_y:
        # In a flipped system (like WorkSurface), the visual "top" starts
        # at y=h.
        y_start_top = h - effective_hh
        # And the visual "bottom" starts at y=0.
        y_start_bottom = 0.0
    else:
        # In a standard Y-down system, the visual "top" is at y=0.
        y_start_top = 0.0
        # And the visual "bottom" is at y=h.
        y_start_bottom = h - effective_hh

    # Side handles always start below the top corner handle's space.
    y_start_middle = effective_hh
    middle_height = h - 2.0 * effective_hh
    if middle_height < 0:
        middle_height = 0

    # Resize handles
    if region == ElementRegion.TOP_LEFT:
        return 0.0, y_start_top, effective_hw, effective_hh
    if region == ElementRegion.TOP_RIGHT:
        return w - effective_hw, y_start_top, effective_hw, effective_hh
    if region == ElementRegion.BOTTOM_LEFT:
        return 0.0, y_start_bottom, effective_hw, effective_hh
    if region == ElementRegion.BOTTOM_RIGHT:
        return w - effective_hw, y_start_bottom, effective_hw, effective_hh

    if region == ElementRegion.TOP_MIDDLE:
        return effective_hw, y_start_top, w - 2.0 * effective_hw, effective_hh
    if region == ElementRegion.BOTTOM_MIDDLE:
        return (
            effective_hw,
            y_start_bottom,
            w - 2.0 * effective_hw,
            effective_hh,
        )
    if region == ElementRegion.MIDDLE_LEFT:
        return 0.0, y_start_middle, effective_hw, middle_height
    if region == ElementRegion.MIDDLE_RIGHT:
        return w - effective_hw, y_start_middle, effective_hw, middle_height

    # Rotate/Shear handles (external)
    if region == ElementRegion.ROTATE_TOP_LEFT:
        rot_w = min((base_handle_size * 1.4) / abs_scale_x, w / 2.0)
        rot_h = min((base_handle_size * 1.4) / abs_scale_y, h / 2.0)
        center_y = h if is_flipped_y else 0.0
        return -rot_w / 2, center_y - rot_h / 2, rot_w, rot_h
    if region == ElementRegion.ROTATE_TOP_RIGHT:
        rot_w = min((base_handle_size * 1.4) / abs_scale_x, w / 2.0)
        rot_h = min((base_handle_size * 1.4) / abs_scale_y, h / 2.0)
        center_y = h if is_flipped_y else 0.0
        return w - rot_w / 2, center_y - rot_h / 2, rot_w, rot_h
    if region == ElementRegion.ROTATE_BOTTOM_LEFT:
        rot_w = min((base_handle_size * 1.4) / abs_scale_x, w / 2.0)
        rot_h = min((base_handle_size * 1.4) / abs_scale_y, h / 2.0)
        center_y = 0.0 if is_flipped_y else h
        return -rot_w / 2, center_y - rot_h / 2, rot_w, rot_h
    if region == ElementRegion.ROTATE_BOTTOM_RIGHT:
        rot_w = min((base_handle_size * 1.4) / abs_scale_x, w / 2.0)
        rot_h = min((base_handle_size * 1.4) / abs_scale_y, h / 2.0)
        center_y = 0.0 if is_flipped_y else h
        return w - rot_w / 2, center_y - rot_h / 2, rot_w, rot_h

    if region == ElementRegion.SHEAR_TOP:
        y_pos = (
            h + handle_dist if is_flipped_y else -handle_dist - effective_hh
        )
        return w / 2 - effective_hw / 2, y_pos, effective_hw, effective_hh
    if region == ElementRegion.SHEAR_BOTTOM:
        y_pos = (
            -effective_hh - handle_dist if is_flipped_y else h + handle_dist
        )
        return w / 2 - effective_hw / 2, y_pos, effective_hw, effective_hh
    if region == ElementRegion.SHEAR_LEFT:
        y_pos = h / 2 - effective_hh / 2
        return -effective_hw - handle_dist, y_pos, effective_hw, effective_hh
    if region == ElementRegion.SHEAR_RIGHT:
        y_pos = h / 2 - effective_hh / 2
        return w + handle_dist, y_pos, effective_hw, effective_hh

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
    candidates: Optional[Set[ElementRegion]] = None,
) -> ElementRegion:
    """
    Checks which interactive region is hit by a point in LOCAL coordinates.
    If `candidates` is provided, it will only check against regions in that
    set.

    Args:
        local_x: The x-coordinate in LOCAL coordinates.
        local_y: The y-coordinate in LOCAL coordinates.
        width: The width of the bounding box.
        height: The height of the bounding box.
        base_handle_size: The desired base size of the handles in pixels.
        scale_compensation: The signed scale factor(s) of the context.
        candidates: Optional set of regions to check against.
    """
    # Determine which handle regions to check based on the candidates.
    # The order of _HIT_TEST_ORDER is crucial to resolve overlap ambiguity.
    regions_to_check = candidates if candidates is not None else BBOX_REGIONS

    for region in regions_to_check:
        # Calculate the hit rectangle for the current region. This ensures
        # the hit-test area matches the rendered handle size.
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

    # If no handle is hit, check the body if it's a candidate.
    if ElementRegion.BODY in regions_to_check:
        if 0 <= local_x < width and 0 <= local_y < height:
            return ElementRegion.BODY

    return ElementRegion.NONE
