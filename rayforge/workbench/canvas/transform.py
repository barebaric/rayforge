from __future__ import annotations
import math
from typing import Tuple
from ...core.matrix import Matrix
from .element import CanvasElement
from .region import ElementRegion

# This data structure defines the behavior for a standard Y-DOWN view.
RESIZE_BEHAVIORS = {
    ElementRegion.TOP_LEFT: {"scale": (-1, -1), "anchor": (1.0, 1.0)},
    ElementRegion.TOP_MIDDLE: {"scale": (0, -1), "anchor": (0.5, 1.0)},
    ElementRegion.TOP_RIGHT: {"scale": (1, -1), "anchor": (0.0, 1.0)},
    ElementRegion.MIDDLE_LEFT: {"scale": (-1, 0), "anchor": (1.0, 0.5)},
    ElementRegion.MIDDLE_RIGHT: {"scale": (1, 0), "anchor": (0.0, 0.5)},
    ElementRegion.BOTTOM_LEFT: {"scale": (-1, 1), "anchor": (1.0, 0.0)},
    ElementRegion.BOTTOM_MIDDLE: {"scale": (0, 1), "anchor": (0.5, 0.0)},
    ElementRegion.BOTTOM_RIGHT: {"scale": (1, 1), "anchor": (0.0, 0.0)},
}


def calculate_resized_box(
    original_box: Tuple[float, float, float, float],
    active_region: ElementRegion,
    drag_delta: Tuple[float, float],
    is_flipped: bool,
    constrain_aspect: bool = False,
    from_center: bool = False,
    min_size: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[float, float, float, float]:
    """
    Calculates a new bounding box based on a resize operation.
    This is the central, data-driven logic for all resizing, and it
    now correctly handles flipped coordinate systems.
    """
    if active_region not in RESIZE_BEHAVIORS:
        return original_box

    orig_x, orig_y, orig_w, orig_h = original_box
    delta_x, delta_y = drag_delta
    min_w, min_h = min_size

    # 1. Get base behavior for a Y-down view
    base_behavior = RESIZE_BEHAVIORS[active_region]

    # 2. Determine the effective geometric behavior based on view orientation
    effective_scale = list(base_behavior["scale"])
    effective_anchor = list(base_behavior["anchor"])

    if is_flipped:
        # Invert the vertical scale's effect
        effective_scale[1] *= -1
        # Invert the vertical anchor's position (top becomes bottom)
        effective_anchor[1] = 1.0 - effective_anchor[1]

    if from_center:
        effective_anchor = [0.5, 0.5]

    # 3. Calculate raw change in width and height (dw, dh)
    dw = delta_x * effective_scale[0]
    dh = delta_y * effective_scale[1]

    if from_center:
        dw *= 2
        dh *= 2

    # 4. Apply aspect ratio constraint
    if constrain_aspect and orig_w > 0 and orig_h > 0:
        aspect = orig_w / orig_h
        is_corner = effective_scale[0] != 0 and effective_scale[1] != 0

        if (is_corner and abs(dw) * aspect > abs(dh)) or (
            not is_corner and effective_scale[0] != 0
        ):
            dh = dw / aspect
        else:
            dw = dh * aspect

    # 5. Calculate final size, enforcing minimums
    new_w = max(orig_w + dw, min_w)
    new_h = max(orig_h + dh, min_h)

    # 6. Calculate new origin based on the fixed anchor point
    anchor_world_x = orig_x + effective_anchor[0] * orig_w
    anchor_world_y = orig_y + effective_anchor[1] * orig_h

    new_x = anchor_world_x - effective_anchor[0] * new_w
    new_y = anchor_world_y - effective_anchor[1] * new_h

    return new_x, new_y, new_w, new_h


def move_element(
    element: CanvasElement,
    world_dx: float,
    world_dy: float,
    initial_world_transform: Matrix,
):
    """
    Calculates the new local transform for an element being moved.

    Args:
        element: The element to move.
        world_dx: The horizontal drag distance in world coordinates.
        world_dy: The vertical drag distance in world coordinates.
        initial_world_transform: The element's world transform at the
                                 start of the drag.
    """
    # Apply the drag translation to the initial world transform
    new_world_transform = initial_world_transform.pre_translate(
        world_dx, world_dy
    )

    # Convert back to the element's local space
    parent_inv_world = Matrix.identity()
    if isinstance(element.parent, CanvasElement):
        parent_inv_world = element.parent.get_world_transform().invert()

    new_local_transform = parent_inv_world @ new_world_transform

    # Set the final transform, preserving all components
    element.set_transform(new_local_transform)


def resize_element(
    element: CanvasElement,
    world_dx: float,
    world_dy: float,
    initial_local_transform: Matrix,
    initial_world_transform: Matrix,
    active_region: ElementRegion,
    view_transform: Matrix,
    shift_pressed: bool,
    ctrl_pressed: bool,
):
    """
    Calculates and applies the new local transform for a resizing element
    by using the centralized `calculate_resized_box` logic.
    """
    # 1. Convert world drag delta to the element's local, unrotated space.
    initial_world_no_trans = initial_world_transform.without_translation()
    inv_rot_scale = initial_world_no_trans.invert()
    local_delta = inv_rot_scale.transform_vector((world_dx, world_dy))

    # 2. Define minimum size in local units
    min_size_world = 2.0
    world_scale_x, world_scale_y = initial_world_transform.get_abs_scale()
    min_size_local = (
        min_size_world / world_scale_x if world_scale_x > 1e-6 else 0,
        min_size_world / world_scale_y if world_scale_y > 1e-6 else 0,
    )

    # 3. Delegate calculation to the central function
    original_box_local = (0, 0, element.width, element.height)
    _, _, new_w, new_h = calculate_resized_box(
        original_box=original_box_local,
        active_region=active_region,
        drag_delta=local_delta,
        is_flipped=view_transform.is_flipped(),  # Pass the flag
        constrain_aspect=shift_pressed,
        from_center=ctrl_pressed,
        min_size=min_size_local,
    )

    # 4. Build the correct delta transform: a scale around the fixed
    # anchor point.
    orig_w, orig_h = element.width, element.height
    base_behavior = RESIZE_BEHAVIORS[active_region]

    anchor_norm = list(base_behavior["anchor"])
    if view_transform.is_flipped():
        anchor_norm[1] = 1.0 - anchor_norm[1]
    if ctrl_pressed:
        anchor_norm = [0.5, 0.5]

    anchor_x = anchor_norm[0] * orig_w
    anchor_y = anchor_norm[1] * orig_h

    scale_x = new_w / orig_w if orig_w > 0 else 1.0
    scale_y = new_h / orig_h if orig_h > 0 else 1.0

    t_to_anchor = Matrix.translation(-anchor_x, -anchor_y)
    m_scale = Matrix.scale(scale_x, scale_y)
    t_from_anchor = Matrix.translation(anchor_x, anchor_y)
    delta_transform_local = t_from_anchor @ m_scale @ t_to_anchor

    # 5. Apply the delta to the initial transform and set it
    new_local_transform = initial_local_transform @ delta_transform_local
    element.set_transform(new_local_transform)


def rotate_element(
    element: CanvasElement,
    world_x: float,
    world_y: float,
    initial_world_transform: Matrix,
    rotation_pivot: Tuple[float, float],
    drag_start_angle: float,
):
    """
    Calculates the new local transform for a rotating element.
    """
    # 1. Calculate the angle of the current mouse position around the pivot
    current_angle = math.degrees(
        math.atan2(
            world_y - rotation_pivot[1],
            world_x - rotation_pivot[0],
        )
    )

    # 2. Find the change in angle since the drag started.
    angle_diff = current_angle - drag_start_angle

    # 3. Apply this delta to the element's initial world state.
    new_world_transform = initial_world_transform.pre_rotate(
        angle_diff, center=rotation_pivot
    )

    # 4. Convert the new world transform back to a local transform.
    parent_inv_world = Matrix.identity()
    if isinstance(element.parent, CanvasElement):
        parent_inv_world = element.parent.get_world_transform().invert()
    new_local_transform = parent_inv_world @ new_world_transform

    # 5. Set the new matrix directly.
    element.set_transform(new_local_transform)


def shear_element(
    element: CanvasElement,
    world_dx: float,
    world_dy: float,
    initial_local_transform: Matrix,
    initial_world_transform: Matrix,
    active_region: ElementRegion,
    view_transform: Matrix,
):
    """Calculates the new local transform for a shearing element."""
    # Transform world delta into element's unrotated local space.
    init_world_no_trans = initial_world_transform.without_translation()
    inv_rot_scale = init_world_no_trans.invert()
    local_dx, local_dy = inv_rot_scale.transform_vector((world_dx, world_dy))

    is_view_flipped = view_transform.is_flipped()

    # If the view is flipped, the coordinate system of the local
    # drag vector is inverted relative to the user's visual perception.
    # We must flip the x-component back to match the visual drag direction.
    if is_view_flipped:
        local_dx = -local_dx

    w, h = element.width, element.height
    shx, shy = 0.0, 0.0
    anchor_x, anchor_y = 0.0, 0.0

    semantic_is_top = active_region == ElementRegion.SHEAR_TOP
    semantic_is_bottom = active_region == ElementRegion.SHEAR_BOTTOM
    semantic_is_left = active_region == ElementRegion.SHEAR_LEFT
    semantic_is_right = active_region == ElementRegion.SHEAR_RIGHT

    if semantic_is_top or semantic_is_bottom:
        geom_is_top_edge = (
            semantic_is_bottom if is_view_flipped else semantic_is_top
        )
        anchor_y = h if geom_is_top_edge else 0
        anchor_x = w / 2

        if semantic_is_top:
            shx = -local_dx / h if h > 1e-6 else 0.0
        else:
            shx = local_dx / h if h > 1e-6 else 0.0

    elif semantic_is_left or semantic_is_right:
        anchor_x = w if semantic_is_left else 0
        anchor_y = h / 2

        if semantic_is_left:
            shy = -local_dy / w if w > 1e-6 else 0.0
        else:
            shy = local_dy / w if w > 1e-6 else 0.0

    # Construct delta shear matrix around local anchor
    t_to = Matrix.translation(-anchor_x, -anchor_y)
    m_shear = Matrix.shear(shx, shy)
    t_from = Matrix.translation(anchor_x, anchor_y)
    delta_local = t_from @ m_shear @ t_to

    new_local_transform = initial_local_transform @ delta_local
    element.set_transform(new_local_transform)
