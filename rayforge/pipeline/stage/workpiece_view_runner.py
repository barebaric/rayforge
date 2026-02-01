from __future__ import annotations
import cairo
import logging
import math
import numpy as np
import time
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any, List
from multiprocessing import shared_memory
from ...shared.tasker.proxy import ExecutionContextProxy
from ...shared.util.colors import ColorSet
from ..artifact import (
    WorkPieceArtifact,
    create_handle_from_dict,
)
from ..artifact.store import ArtifactStore
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
)

if TYPE_CHECKING:
    import threading

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


def _draw_vertices_progressive(
    ctx: cairo.Context,
    vertex_data,
    color_set: ColorSet,
    show_travel: bool,
    line_width_mm: float,
    surface,
    proxy: ExecutionContextProxy,
    num_batches: int = 3,
):
    """
    Draws vertex data in multiple batches for progressive rendering.
    Sends view_artifact_updated events after each batch.
    """
    ctx.save()
    ctx.set_line_width(line_width_mm)
    ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

    batch_num = 0

    # --- Batch 1: Draw Travel & Zero-Power Moves ---
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

        surface.flush()
        proxy.send_event("view_artifact_updated")
        logger.debug("Worker: Sent view_artifact_updated after travel/zero")
        batch_num += 1

    # --- Draw Powered Moves in Batches ---
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

        # Calculate remaining batches
        remaining_batches = num_batches - batch_num
        if remaining_batches < 1:
            remaining_batches = 1

        # Calculate total number of segments
        total_segments = len(powered_v)

        # Split segments into batches (not colors)
        segments_per_batch = max(1, total_segments // remaining_batches)

        for batch_idx in range(remaining_batches):
            start_seg = batch_idx * segments_per_batch
            end_seg = (
                start_seg + segments_per_batch
                if batch_idx < remaining_batches - 1
                else total_segments
            )

            # Skip if start_seg is beyond the available segments
            if start_seg >= total_segments:
                continue

            # Draw all segments in this batch
            for seg_idx in range(start_seg, end_seg):
                start, end = powered_v[seg_idx]
                # Get color for this segment
                color_idx = inverse_indices[seg_idx]
                color = unique_colors[color_idx]
                ctx.set_source_rgba(*color)
                ctx.move_to(start[0], start[1])
                ctx.line_to(end[0], end[1])
                ctx.stroke()

            surface.flush()
            proxy.send_event("view_artifact_updated")
            logger.debug(
                f"Worker: Sent view_artifact_updated after powered "
                f"batch {batch_idx + 1}/{remaining_batches} "
                f"(segments {start_seg}-{end_seg}/{total_segments})"
            )
            # Small delay to ensure each event gets processed in a separate
            # GTK frame, enabling progressive rendering
            time.sleep(0.05)

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
    premultiplied_rgb = rgb_ch * alpha_ch * 255
    # Round, clip, and cast to prevent truncation artifacts
    premultiplied_rgb_int = np.clip(np.round(premultiplied_rgb), 0, 255)
    premultiplied_rgb_int = premultiplied_rgb_int.astype(np.uint8)

    bgra_texture[..., 0] = premultiplied_rgb_int[..., 2]  # B
    bgra_texture[..., 1] = premultiplied_rgb_int[..., 1]  # G
    bgra_texture[..., 2] = premultiplied_rgb_int[..., 0]  # R
    bgra_texture[..., 3] = (alpha_ch.squeeze() * 255).astype(np.uint8)

    texture_surface = cairo.ImageSurface.create_for_data(
        memoryview(np.ascontiguousarray(bgra_texture)),
        cairo.FORMAT_ARGB32,
        w,
        h,
    )
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
    all_vertices: List[np.ndarray] = []
    has_content = False

    if artifact.vertex_data:
        v_data = artifact.vertex_data
        if v_data.powered_vertices.size > 0:
            all_vertices.append(v_data.powered_vertices)
        if show_travel:
            if v_data.travel_vertices.size > 0:
                all_vertices.append(v_data.travel_vertices)
            if v_data.zero_power_vertices.size > 0:
                all_vertices.append(v_data.zero_power_vertices)

    v_stack = np.vstack(all_vertices) if all_vertices else None
    if v_stack is not None:
        v_x1, v_y1, _ = np.min(v_stack, axis=0)
        v_x2, v_y2, _ = np.max(v_stack, axis=0)
        has_content = True
    else:
        v_x1, v_y1, v_x2, v_y2 = math.inf, math.inf, -math.inf, -math.inf

    if artifact.texture_data:
        tex = artifact.texture_data
        t_x1, t_y1 = tex.position_mm
        t_x2, t_y2 = t_x1 + tex.dimensions_mm[0], t_y1 + tex.dimensions_mm[1]
        v_x1, v_x2 = min(v_x1, t_x1), max(v_x2, t_x2)
        v_y1, v_y2 = min(v_y1, t_y1), max(v_y2, t_y2)
        has_content = True

    if not has_content:
        return None

    return (v_x1, v_y1, v_x2 - v_x1, v_y2 - v_y1)


def make_workpiece_view_artifact_in_subprocess(
    proxy: ExecutionContextProxy,
    artifact_store: ArtifactStore,
    workpiece_artifact_handle_dict: Dict[str, Any],
    render_context_dict: Dict[str, Any],
    creator_tag: str,
    adoption_event: Optional["threading.Event"] = None,
) -> Optional[Dict[str, Any]]:
    """
    Renders a WorkPieceArtifact to a bitmap in a background process.
    """
    logger.debug("Worker: Starting view artifact rendering...")
    proxy.set_message(_("Preparing 2D preview..."))
    context = RenderContext.from_dict(render_context_dict)
    handle = create_handle_from_dict(workpiece_artifact_handle_dict)
    logger.debug(f"Worker: Retrieved handle {handle.shm_name}")
    artifact = artifact_store.get(handle)
    if not isinstance(artifact, WorkPieceArtifact):
        logger.error("Runner received incorrect artifact type.")
        return None

    logger.debug("Worker: Calculating content bbox...")
    bbox = _get_content_bbox(artifact, context.show_travel_moves)
    if not bbox or bbox[2] <= 1e-9 or bbox[3] <= 1e-9:
        logger.warning(
            f"Worker: No content to render (bbox={bbox}). Returning None."
        )
        return None  # No content to render
    logger.debug(f"Worker: Content bbox calculated: {bbox}")

    x_mm, y_mm, w_mm, h_mm = bbox
    ppm_x, ppm_y = context.pixels_per_mm
    margin = context.margin_px

    # Calculate requested pixel dimensions
    requested_width_px = round(w_mm * ppm_x) + 2 * margin
    requested_height_px = round(h_mm * ppm_y) + 2 * margin

    # Clamp dimensions to Cairo limit
    width_px = min(requested_width_px, CAIRO_MAX_DIMENSION)
    height_px = min(requested_height_px, CAIRO_MAX_DIMENSION)

    if width_px <= 2 * margin or height_px <= 2 * margin:
        return None

    # Calculate effective PPM based on the final (possibly clamped) dimensions.
    # This ensures the content is scaled down to fit if the max dimension is
    # hit.
    effective_ppm_x = (width_px - 2 * margin) / w_mm if w_mm > 0 else ppm_x
    effective_ppm_y = (height_px - 2 * margin) / h_mm if h_mm > 0 else ppm_y

    # --- Phase 1: Create the shared artifact upfront ---
    # (don't send event yet - wait until drawing is complete)
    logger.debug(
        f"Worker: Creating view artifact with dimensions "
        f"{width_px}x{height_px}"
    )
    bitmap = np.zeros(shape=(height_px, width_px, 4), dtype=np.uint8)
    view_artifact = WorkPieceViewArtifact(bitmap_data=bitmap, bbox_mm=bbox)
    view_handle = artifact_store.put(view_artifact, creator_tag=creator_tag)
    logger.debug(f"Worker: Created view artifact {view_handle.shm_name}")

    shm = None
    try:
        # --- Phase 2: Render directly into the shared memory buffer ---
        logger.debug("Worker: Opening shared memory for rendering...")
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

        # Set up transform: Y-down pixel space from Y-up mm space.
        ctx.translate(margin, height_px - margin)
        ctx.scale(effective_ppm_x, -effective_ppm_y)
        ctx.translate(-x_mm, -y_mm)

        color_set = ColorSet.from_dict(context.color_set_dict)

        # --- Phase 3: Draw all content progressively ---
        # Send created event AFTER first drawing phase (if texture exists)
        # or BEFORE drawing (if no texture) so the stage adopts the artifact
        if artifact.texture_data:
            logger.debug("Worker: Drawing texture...")
            _draw_texture(ctx, artifact.texture_data, color_set)
            surface.flush()
            # Send created event after texture is drawn
            proxy.send_event(
                "view_artifact_created",
                {"handle_dict": view_handle.to_dict()},
            )
            logger.debug("Worker: Sent view_artifact_created after texture")
            # Send intermediate update after texture
            proxy.send_event("view_artifact_updated")
            logger.debug("Worker: Sent view_artifact_updated after texture")
        else:
            # No texture, send created event before drawing vertices
            proxy.send_event(
                "view_artifact_created",
                {"handle_dict": view_handle.to_dict()},
            )
            logger.debug("Worker: Sent view_artifact_created before vertices")

        if artifact.vertex_data:
            logger.debug(
                f"Worker: Drawing vertices progressively... "
                f"show_travel={context.show_travel_moves}"
            )
            # Set line width to be 1 pixel wide in device space
            line_width_mm = (
                1.0 / effective_ppm_x if effective_ppm_x > 0 else 1.0
            )
            logger.debug("Worker: About to call _draw_vertices_progressive")
            _draw_vertices_progressive(
                ctx,
                artifact.vertex_data,
                color_set,
                context.show_travel_moves,
                line_width_mm,
                surface,
                proxy,
                num_batches=3,
            )
            logger.debug("Worker: _draw_vertices_progressive returned")

    finally:
        if shm:
            shm.close()

    # Send final update event after all drawing is complete
    proxy.send_event("view_artifact_updated")
    logger.debug("Worker: Sent final view_artifact_updated event")

    logger.debug("Worker: View artifact rendering complete.")
    proxy.set_progress(1.0)
    # The result is communicated via events; return None.
    return None
