from __future__ import annotations
import logging
import math
import numpy as np
import cairo
from typing import Optional, Tuple, Dict, Any, List
from multiprocessing import shared_memory
from ...shared.tasker.proxy import ExecutionContextProxy
from ...context import get_context
from ..artifact import (
    WorkPieceArtifact,
    create_handle_from_dict,
)
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
)
from ...shared.util.colors import ColorSet

logger = logging.getLogger(__name__)

# Cairo has a hard limit on surface dimensions.
CAIRO_MAX_DIMENSION = 30000


def _draw_vertices(
    ctx: cairo.Context,
    vertex_data,
    color_set: ColorSet,
    show_travel: bool,
    line_width_mm: float,
):
    """Draws all vertex data onto the provided cairo context."""
    ctx.save()
    ctx.set_line_width(line_width_mm)
    ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

    # --- Draw Travel & Zero-Power Moves ---
    if show_travel:
        if vertex_data.travel_vertices.size > 0:
            travel_v = vertex_data.travel_vertices.reshape(-1, 2, 3)
            ctx.set_source_rgba(*color_set.get_rgba("travel"))
            for start, end in travel_v:
                ctx.move_to(start[0], start[1])
                ctx.line_to(end[0], end[1])
            ctx.stroke()

        if vertex_data.zero_power_vertices.size > 0:
            zero_v = vertex_data.zero_power_vertices.reshape(-1, 2, 3)
            ctx.set_source_rgba(*color_set.get_rgba("zero_power"))
            for start, end in zero_v:
                ctx.move_to(start[0], start[1])
                ctx.line_to(end[0], end[1])
            ctx.stroke()

    # --- Draw Powered Moves (Grouped by Color for performance) ---
    if vertex_data.powered_vertices.size > 0:
        powered_v = vertex_data.powered_vertices.reshape(-1, 2, 3)
        powered_c = vertex_data.powered_colors
        cut_lut = color_set.get_lut("cut")

        # Use power from the first vertex of each segment for color.
        power_indices = (powered_c[::2, 0] * 255.0).astype(np.uint8)
        themed_colors_per_segment = cut_lut[power_indices]
        unique_colors, inverse_indices = np.unique(
            themed_colors_per_segment, axis=0, return_inverse=True
        )

        for i, color in enumerate(unique_colors):
            ctx.set_source_rgba(*color)
            segment_indices = np.where(inverse_indices == i)[0]
            for seg_idx in segment_indices:
                start, end = powered_v[seg_idx]
                ctx.move_to(start[0], start[1])
                ctx.line_to(end[0], end[1])
            ctx.stroke()
    ctx.restore()


def _draw_texture(
    ctx: cairo.Context,
    texture_data,
    color_set: ColorSet,
):
    """Draws themed texture data onto the provided cairo context."""
    power_data = texture_data.power_texture_data
    if power_data.size == 0:
        return

    engrave_lut = color_set.get_lut("engrave")
    rgba_texture = engrave_lut[power_data]
    rgba_texture[power_data == 0, 3] = 0.0  # Set alpha for zero power

    h, w = rgba_texture.shape[:2]
    # Create pre-multiplied BGRA data for Cairo
    alpha_ch = rgba_texture[..., 3, np.newaxis]
    rgb_ch = rgba_texture[..., :3]
    bgra_texture = np.empty((h, w, 4), dtype=np.uint8)
    premultiplied_rgb = rgb_ch * alpha_ch * 255.0
    # Round, clip, and cast to prevent truncation artifacts
    premultiplied_rgb_int = np.clip(np.round(premultiplied_rgb), 0, 255)
    premultiplied_rgb_int = premultiplied_rgb_int.astype(np.uint8)

    bgra_texture[..., 0] = premultiplied_rgb_int[..., 2]  # B
    bgra_texture[..., 1] = premultiplied_rgb_int[..., 1]  # G
    bgra_texture[..., 2] = premultiplied_rgb_int[..., 0]  # R
    bgra_texture[..., 3] = (alpha_ch.squeeze() * 255).astype(np.uint8)

    # The `create_for_data` surface is just a VIEW on the numpy buffer. The
    # buffer can be garbage collected after this function returns, creating
    # a race condition. To solve this, we create a new surface and paint
    # the view onto it, forcing Cairo to make its own copy of the pixel data.
    view_surface = cairo.ImageSurface.create_for_data(
        memoryview(np.ascontiguousarray(bgra_texture)),
        cairo.FORMAT_ARGB32,
        w,
        h,
    )
    texture_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, w, h)
    copy_ctx = cairo.Context(texture_surface)
    copy_ctx.set_source_surface(view_surface, 0, 0)
    copy_ctx.paint()
    # Now, `texture_surface` has its own data, and the `bgra_texture` and
    # `view_surface` can be safely garbage collected.

    ctx.save()
    pos_mm = texture_data.position_mm
    dim_mm = texture_data.dimensions_mm

    # We are in a Y-up context. To draw the Y-down texture correctly:
    # 1. Go to the top-left corner of the destination rectangle.
    ctx.translate(pos_mm[0], pos_mm[1] + dim_mm[1])
    # 2. Scale Y by -1 to flip the drawing space locally.
    ctx.scale(1, -1)
    # 3. Scale the flipped space to the correct dimensions.
    ctx.scale(dim_mm[0] / w, dim_mm[1] / h)

    ctx.set_source_surface(texture_surface, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()


def _get_content_bbox(
    artifact: WorkPieceArtifact, show_travel: bool
) -> Optional[Tuple[float, float, float, float]]:
    """Calculate the union bounding box of all visual content."""
    bbox_min = np.array([math.inf, math.inf])
    bbox_max = np.array([-math.inf, -math.inf])
    has_content = False

    # Union the bounding box of any vertices
    if artifact.vertex_data:
        all_vertices: List[np.ndarray] = []
        v_data = artifact.vertex_data
        if v_data.powered_vertices.size > 0:
            all_vertices.append(v_data.powered_vertices)
        if show_travel:
            if v_data.travel_vertices.size > 0:
                all_vertices.append(v_data.travel_vertices)
            if v_data.zero_power_vertices.size > 0:
                all_vertices.append(v_data.zero_power_vertices)

        if all_vertices:
            v_stack = np.vstack(all_vertices)
            v_min = np.min(v_stack, axis=0)[:2]
            v_max = np.max(v_stack, axis=0)[:2]
            bbox_min = np.minimum(bbox_min, v_min)
            bbox_max = np.maximum(bbox_max, v_max)
            has_content = True

    # Union the bounding box of the texture, if it has pixels
    if (
        artifact.texture_data
        and artifact.texture_data.power_texture_data.size > 0
    ):
        tex = artifact.texture_data
        t_min = np.array(tex.position_mm)
        t_max = t_min + np.array(tex.dimensions_mm)
        bbox_min = np.minimum(bbox_min, t_min)
        bbox_max = np.maximum(bbox_max, t_max)
        has_content = True

    if not has_content:
        return None

    width, height = bbox_max - bbox_min
    return (bbox_min[0], bbox_min[1], width, height)


def make_workpiece_view_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    workpiece_artifact_handle_dict: Dict[str, Any],
    render_context_dict: Dict[str, Any],
    generation_id: int,
    creator_tag: str,
) -> Optional[Dict[str, Any]]:
    """
    Renders a WorkPieceArtifact to a bitmap in a background process.
    """
    proxy.set_message(_("Preparing 2D preview..."))
    context = RenderContext.from_dict(render_context_dict)
    handle = create_handle_from_dict(workpiece_artifact_handle_dict)
    artifact_store = get_context().artifact_store
    artifact = artifact_store.get(handle)
    if not isinstance(artifact, WorkPieceArtifact):
        logger.error("Runner received incorrect artifact type.")
        return None

    bbox = _get_content_bbox(artifact, context.show_travel_moves)
    if not bbox or bbox[2] <= 1e-9 or bbox[3] <= 1e-9:
        return None  # No content to render

    x_mm, y_mm, w_mm, h_mm = bbox
    ppm_x, ppm_y = context.pixels_per_mm
    margin = context.margin_px
    width_px = min(round(w_mm * ppm_x) + 2 * margin, CAIRO_MAX_DIMENSION)
    height_px = min(round(h_mm * ppm_y) + 2 * margin, CAIRO_MAX_DIMENSION)
    if width_px <= 2 * margin or height_px <= 2 * margin:
        return None

    # --- Phase 1: Create and emit the shared artifact upfront ---
    bitmap = np.zeros(shape=(height_px, width_px, 4), dtype=np.uint8)
    view_artifact = WorkPieceViewArtifact(bitmap_data=bitmap, bbox_mm=bbox)
    view_handle = artifact_store.put(view_artifact, creator_tag=creator_tag)

    logger.debug(
        f"View artifact created (shm_name={view_handle.shm_name}, "
        f"gen_id={generation_id}). "
        "Bitmap is currently empty. Sending 'view_artifact_created' event."
    )
    proxy.send_event(
        "view_artifact_created",
        {"handle_dict": view_handle.to_dict(), "generation_id": generation_id},
    )

    logger.debug("Event sent. Beginning render into shared memory.")

    shm = None
    try:
        # --- Phase 2: Render directly into the shared memory buffer ---
        shm = shared_memory.SharedMemory(name=view_handle.shm_name)
        shm_bitmap = np.ndarray(
            shape=(height_px, width_px, 4),
            dtype=np.uint8,
            buffer=shm.buf,
        )
        surface = cairo.ImageSurface.create_for_data(
            memoryview(shm_bitmap), cairo.FORMAT_ARGB32, width_px, height_px
        )
        ctx = cairo.Context(surface)

        # Set up transform: Y-down pixel space from Y-up mm space
        ctx.translate(margin, height_px - margin)
        ctx.scale(ppm_x, -ppm_y)
        ctx.translate(-x_mm, -y_mm)

        color_set = ColorSet.from_dict(context.color_set_dict)

        # --- Phase 3: Signal updates after each chunk of work ---
        if artifact.texture_data:
            logger.debug(f"Drawing texture for {view_handle.shm_name}...")
            _draw_texture(ctx, artifact.texture_data, color_set)
            surface.flush()
            logger.debug(
                f"Texture drawn for {view_handle.shm_name}. "
                f"Sending 'view_artifact_updated'."
            )
            proxy.send_event(
                "view_artifact_updated", {"generation_id": generation_id}
            )

        if artifact.vertex_data:
            logger.debug(f"Drawing vertices for {view_handle.shm_name}...")
            # Set line width to be 1 pixel wide in device space
            line_width_mm = 1.0 / ppm_x if ppm_x > 0 else 1.0
            _draw_vertices(
                ctx,
                artifact.vertex_data,
                color_set,
                context.show_travel_moves,
                line_width_mm,
            )
            surface.flush()
            logger.debug(
                f"Vertices drawn for {view_handle.shm_name}. "
                f"Sending 'view_artifact_updated'."
            )
            proxy.send_event(
                "view_artifact_updated", {"generation_id": generation_id}
            )

        logger.debug(f"Render complete for {view_handle.shm_name}.")

    finally:
        if shm:
            shm.close()

    proxy.set_progress(1.0)
    # The result is communicated via events; return None.
    return None
