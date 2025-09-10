from __future__ import annotations
from typing import TYPE_CHECKING, Tuple, Union, Dict, Any, List
import cairo
import math
from ...core.matrix import Matrix
from .region import (
    ElementRegion,
    ROTATE_SHEAR_HANDLES,
    CORNER_RESIZE_HANDLES,
    MIDDLE_RESIZE_HANDLES,
    ROTATE_HANDLES,
    RESIZE_HANDLES,
)

if TYPE_CHECKING:
    from .canvas import CanvasElement, MultiSelectionGroup, SelectionMode


def _draw_quad_handle(
    ctx: cairo.Context,
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    p4: Tuple[float, float],
    is_hovered: bool,
):
    """Draws a quadrilateral handle given four screen-space points."""
    if is_hovered:
        ctx.set_source_rgba(0.3, 0.6, 0.9, 0.9)
    else:
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)

    ctx.move_to(*p1)
    ctx.line_to(*p2)
    ctx.line_to(*p3)
    ctx.line_to(*p4)
    ctx.close_path()
    ctx.fill()


def _draw_square_handle(
    ctx: cairo.Context, width: float, height: float, is_hovered: bool
):
    """Draws a square handle. Uses the smaller of width/height for size."""
    size = min(width, height)
    if is_hovered:
        ctx.set_source_rgba(0.3, 0.6, 0.9, 0.9)
    else:
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)

    half_size = size / 2
    ctx.rectangle(-half_size, -half_size, size, size)
    ctx.fill()


def _draw_rectangle_handle(
    ctx: cairo.Context, width: float, height: float, is_hovered: bool
):
    """Draws a rectangular handle, perfect for stretched edge handles."""
    if is_hovered:
        ctx.set_source_rgba(0.3, 0.6, 0.9, 0.9)
    else:
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.7)

    ctx.rectangle(-width / 2, -height / 2, width, height)
    ctx.fill()


def _draw_arc_handle(
    ctx: cairo.Context, width: float, height: float, is_hovered: bool
):
    """Draws a rotation arc handle. Uses average of width/height for size."""
    size = (width + height) / 2.0
    if is_hovered:
        ctx.set_source_rgba(0.3, 0.6, 0.9, 0.95)
    else:
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.8)

    ctx.set_line_width(2.0)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    radius = size * 0.5
    start_angle, end_angle = math.radians(45), math.radians(-45)
    ctx.arc_negative(0, 0, radius, start_angle, end_angle)

    def draw_arrowhead(point_angle: float, is_start_arrow: bool):
        arrow_len, arrow_width = size * 0.18, size * 0.2
        ctx.save()
        px, py = radius * math.cos(point_angle), radius * math.sin(point_angle)
        ctx.translate(px, py)
        tangent = point_angle - math.pi / 2.0
        ctx.rotate(tangent + (math.pi if is_start_arrow else 0))
        ctx.move_to(0, 0)

        # Draw an asymmetric arrowhead. The inner tine is flatter.
        if is_start_arrow:
            # Outer tine (slightly more pronounced)
            ctx.line_to(-arrow_len, -arrow_width * 0.9)
            ctx.move_to(0, 0)
            # Inner tine (flatter)
            ctx.line_to(-arrow_len * 0.9, arrow_width * 1.2)
        else:
            # Inner tine (flatter)
            ctx.line_to(-arrow_len * 0.9, -arrow_width * 1.2)
            ctx.move_to(0, 0)
            # Outer tine (slightly more pronounced)
            ctx.line_to(-arrow_width, arrow_len * 0.9)
        ctx.restore()

    draw_arrowhead(start_angle, True)
    draw_arrowhead(end_angle, False)
    ctx.stroke()


def _draw_arrow_handle(
    ctx: cairo.Context, width: float, height: float, is_hovered: bool
):
    """Draws a bidirectional arrow. Uses average of width/height for size."""
    size = (width + height) / 2.0
    if is_hovered:
        ctx.set_source_rgba(0.3, 0.6, 0.9, 0.95)
    else:
        ctx.set_source_rgba(0.2, 0.5, 0.8, 0.8)

    ctx.set_line_width(2.0)
    ctx.set_line_cap(cairo.LINE_CAP_ROUND)
    ctx.set_line_join(cairo.LINE_JOIN_ROUND)
    length, arrow_size = size * 0.4, size * 0.25

    ctx.move_to(-length, 0)
    ctx.line_to(length, 0)
    ctx.move_to(length, 0)
    ctx.line_to(length - arrow_size, -arrow_size * 0.7)
    ctx.move_to(length, 0)
    ctx.line_to(length - arrow_size, arrow_size * 0.7)
    ctx.move_to(-length, 0)
    ctx.line_to(-length + arrow_size, -arrow_size * 0.7)
    ctx.move_to(-length, 0)
    ctx.line_to(-length + arrow_size, arrow_size * 0.7)
    ctx.stroke()


_ARC_HANDLE_BASE_ANGLES_DEG = {
    ElementRegion.ROTATE_TOP_RIGHT: 315,
    ElementRegion.ROTATE_TOP_LEFT: 225,
    ElementRegion.ROTATE_BOTTOM_LEFT: 135,
    ElementRegion.ROTATE_BOTTOM_RIGHT: 45,
}

HANDLE_DRAW_INFO: Dict[ElementRegion, Dict[str, Any]] = {
    region: {
        "draw": _draw_square_handle,
        "get_angle": lambda t, r: t.get_x_axis_angle(),
    }
    for region in CORNER_RESIZE_HANDLES
}
HANDLE_DRAW_INFO.update(
    {
        ElementRegion.TOP_MIDDLE: {
            "draw": _draw_rectangle_handle,
            "get_angle": lambda t, r: t.get_x_axis_angle(),
        },
        ElementRegion.BOTTOM_MIDDLE: {
            "draw": _draw_rectangle_handle,
            "get_angle": lambda t, r: t.get_x_axis_angle(),
        },
        ElementRegion.MIDDLE_LEFT: {
            "draw": _draw_rectangle_handle,
            "get_angle": lambda t, r: t.get_y_axis_angle(),
            "swap_dims": True,
        },
        ElementRegion.MIDDLE_RIGHT: {
            "draw": _draw_rectangle_handle,
            "get_angle": lambda t, r: t.get_y_axis_angle(),
            "swap_dims": True,
        },
    }
)
HANDLE_DRAW_INFO.update(
    {
        region: {
            "draw": _draw_arc_handle,
            "get_angle": lambda t, r: t.get_rotation()
            + _ARC_HANDLE_BASE_ANGLES_DEG[r],
        }
        for region in ROTATE_HANDLES
    }
)
HANDLE_DRAW_INFO.update(
    {
        ElementRegion.SHEAR_TOP: {
            "draw": _draw_arrow_handle,
            "get_angle": lambda t, r: t.get_x_axis_angle(),
        },
        ElementRegion.SHEAR_BOTTOM: {
            "draw": _draw_arrow_handle,
            "get_angle": lambda t, r: t.get_x_axis_angle(),
        },
        ElementRegion.SHEAR_LEFT: {
            "draw": _draw_arrow_handle,
            "get_angle": lambda t, r: t.get_y_axis_angle(),
            "swap_dims": True,
        },
        ElementRegion.SHEAR_RIGHT: {
            "draw": _draw_arrow_handle,
            "get_angle": lambda t, r: t.get_y_axis_angle(),
            "swap_dims": True,
        },
    }
)


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


def _render_handles(
    ctx: cairo.Context,
    target: Union[CanvasElement, MultiSelectionGroup],
    transform_to_screen: Matrix,
    regions: List[ElementRegion],
    hovered_region: ElementRegion,
    base_handle_size: float,
    scale_compensation: Tuple[float, float],
):
    sx_abs, sy_abs = transform_to_screen.get_abs_scale()

    for region in regions:
        # Resize handles must be drawn as transformed quads to
        # account for shear. Rotate/Shear handles are glyphs that are only
        # rotated to align with the frame.
        if region in RESIZE_HANDLES:
            lx, ly, lw, lh = target.get_region_rect(
                region, base_handle_size, scale_compensation
            )
            if lw <= 0 or lh <= 0:
                continue

            # Get the 4 corners of the handle's rectangle in local space
            corners_local = [
                (lx, ly),
                (lx + lw, ly),
                (lx + lw, ly + lh),
                (lx, ly + lh),
            ]
            # Transform them to screen space to get the final skewed quad
            corners_screen = [
                transform_to_screen.transform_point(p) for p in corners_local
            ]
            _draw_quad_handle(
                ctx,
                *corners_screen,
                is_hovered=(region == hovered_region),
            )
        else:  # Rotate or Shear handles
            draw_info = HANDLE_DRAW_INFO.get(region)
            if not draw_info:
                continue

            lx, ly, lw, lh = target.get_region_rect(
                region, base_handle_size, scale_compensation
            )
            if lw <= 0 or lh <= 0:
                continue

            center_local = (lx + lw / 2, ly + lh / 2)
            screen_x, screen_y = transform_to_screen.transform_point(
                center_local
            )
            angle_rad = math.radians(
                draw_info["get_angle"](transform_to_screen, region)
            )

            screen_width = lw * sx_abs
            screen_height = lh * sy_abs

            draw_w, draw_h = screen_width, screen_height
            if draw_info.get("swap_dims", False):
                draw_w, draw_h = screen_height, screen_width

            ctx.save()
            ctx.translate(screen_x, screen_y)
            ctx.rotate(angle_rad)
            draw_info["draw"](
                ctx, draw_w, draw_h, is_hovered=(region == hovered_region)
            )
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

    if regions_to_draw:
        _render_handles(
            ctx,
            target,
            transform_to_screen,
            regions_to_draw,
            hovered_region,
            base_handle_size,
            scale_compensation,
        )

    if with_labels:
        _render_debug_labels(
            ctx,
            target,
            transform_to_screen,
            regions_to_draw,
            base_handle_size,
            scale_compensation,
        )


def _render_debug_labels(
    ctx, target, transform, regions, base_size, scale_comp
):
    """Helper to draw debug text labels on handles."""
    _region_letters = {
        r: chr(ord("A") + i)
        for i, r in enumerate(RESIZE_HANDLES | ROTATE_SHEAR_HANDLES)
    }
    ctx.save()
    ctx.set_source_rgb(1, 0, 0)
    ctx.select_font_face(
        "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
    )
    ctx.set_font_size(10)
    for region in regions:
        letter = _region_letters.get(region)
        if not letter:
            continue
        lx, ly, lw, lh = target.get_region_rect(region, base_size, scale_comp)
        sx, sy = transform.transform_point((lx + lw / 2, ly + lh / 2))
        ext = ctx.text_extents(letter)
        ctx.move_to(
            sx - (ext.width / 2 + ext.x_bearing),
            sy - (ext.height / 2 + ext.y_bearing),
        )
        ctx.show_text(letter)
    ctx.restore()
