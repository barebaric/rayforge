from __future__ import annotations
from typing import TYPE_CHECKING, Union
import cairo
from ...core.matrix import Matrix
from .region import (
    ElementRegion,
    ROTATE_SHEAR_HANDLES,
    CORNER_RESIZE_HANDLES,
    MIDDLE_RESIZE_HANDLES,
)

if TYPE_CHECKING:
    from .canvas import CanvasElement, MultiSelectionGroup, SelectionMode


# Debug mapping for handles. Each gets a unique letter.
_region_letters = {
    ElementRegion.TOP_LEFT: "A",
    ElementRegion.TOP_MIDDLE: "B",
    ElementRegion.TOP_RIGHT: "C",
    ElementRegion.MIDDLE_LEFT: "D",
    ElementRegion.MIDDLE_RIGHT: "E",
    ElementRegion.BOTTOM_LEFT: "F",
    ElementRegion.BOTTOM_MIDDLE: "G",
    ElementRegion.BOTTOM_RIGHT: "H",
    ElementRegion.ROTATE_TOP_LEFT: "I",
    ElementRegion.ROTATE_TOP_RIGHT: "J",
    ElementRegion.ROTATE_BOTTOM_LEFT: "K",
    ElementRegion.ROTATE_BOTTOM_RIGHT: "L",
    ElementRegion.SHEAR_TOP: "M",
    ElementRegion.SHEAR_BOTTOM: "N",
    ElementRegion.SHEAR_LEFT: "O",
    ElementRegion.SHEAR_RIGHT: "P",
}


def render_selection_frame(
    ctx: cairo.Context,
    target: Union[CanvasElement, MultiSelectionGroup],
    transform_to_screen: Matrix,
):
    """
    Draws the dashed selection frame for a target.

    Args:
        ctx: The cairo context (in screen space).
        target: The CanvasElement or MultiSelectionGroup to draw frame for.
        transform_to_screen: The matrix to transform from local to screen.
    """
    ctx.save()
    w, h = target.width, target.height
    corners_local = [(0, 0), (w, 0), (w, h), (0, h)]
    corners_screen = [
        transform_to_screen.transform_point(p) for p in corners_local
    ]

    # Draw the dashed outline connecting the screen-space corners.
    # Line width and dash pattern are now in fixed pixels.
    ctx.set_source_rgb(0.4, 0.4, 0.4)
    ctx.set_line_width(1.0)
    ctx.set_dash((5, 5))

    ctx.move_to(*corners_screen[0])
    ctx.line_to(*corners_screen[1])
    ctx.line_to(*corners_screen[2])
    ctx.line_to(*corners_screen[3])
    ctx.close_path()
    ctx.stroke()
    ctx.restore()


def render_selection_handles(
    ctx: cairo.Context,
    target: Union[CanvasElement, MultiSelectionGroup],
    transform_to_screen: Matrix,
    mode: SelectionMode,
    hovered_region: ElementRegion,
    base_handle_size: float,
    with_labels: bool = False,
):
    """
    Renders selection handles for a target based on the current interaction
    mode.

    This function understands the application logic (modes, regions) but is
    "dumb" regarding transformations; it requires a pre-computed matrix to
    map the target's local coordinates to the screen.

    Args:
        ctx: The cairo context (in screen space).
        target: The CanvasElement or MultiSelectionGroup to draw handles for.
        transform_to_screen: The matrix to transform from local to screen.
        mode: The current SelectionMode.
        hovered_region: The currently hovered region, for hover effects.
        base_handle_size: The base pixel size for the handles.
        with_labels: If True, draws debug text labels on the handles.
    """
    from .canvas import SelectionMode  # Avoid circular import at module level

    if transform_to_screen.has_zero_scale():
        return
    sx_abs, sy_abs = transform_to_screen.get_abs_scale()
    is_view_flipped = transform_to_screen.is_flipped()
    scale_compensation = (sx_abs, -sy_abs if is_view_flipped else sy_abs)

    # Determine regions to draw
    regions_to_draw = []
    if mode == SelectionMode.RESIZE:
        regions_to_draw.extend(CORNER_RESIZE_HANDLES)
        if hovered_region in MIDDLE_RESIZE_HANDLES:
            regions_to_draw.append(hovered_region)
    elif mode == SelectionMode.ROTATE_SHEAR:
        regions_to_draw.extend(ROTATE_SHEAR_HANDLES)

    # Render handles
    ctx.save()
    ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)
    for region in regions_to_draw:
        lx, ly, lw, lh = target.get_region_rect(
            region, base_handle_size, scale_compensation
        )
        if lw <= 0 or lh <= 0:
            continue

        local_corners = [
            (lx, ly), (lx + lw, ly), (lx + lw, ly + lh), (lx, ly + lh)
        ]
        screen_corners = [
            transform_to_screen.transform_point(p) for p in local_corners
        ]
        ctx.move_to(*screen_corners[0])
        for i in range(1, 4):
            ctx.line_to(*screen_corners[i])
        ctx.close_path()
        ctx.fill()
    ctx.restore()

    # Render labels
    if not with_labels:
        return

    ctx.save()
    ctx.set_source_rgb(1, 1, 1)
    ctx.select_font_face(
        "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
    )
    ctx.set_font_size(10)
    for region in regions_to_draw:
        letter = _region_letters.get(region)
        if not letter:
            continue

        lx, ly, lw, lh = target.get_region_rect(
            region, base_handle_size, scale_compensation
        )
        center_local_x = lx + lw / 2
        center_local_y = ly + lh / 2
        screen_x, screen_y = transform_to_screen.transform_point(
            (center_local_x, center_local_y)
        )
        extents = ctx.text_extents(letter)
        text_x = screen_x - (extents.width / 2 + extents.x_bearing)
        text_y = screen_y - (extents.height / 2 + extents.y_bearing)
        ctx.move_to(text_x, text_y)
        ctx.show_text(letter)
    ctx.restore()
