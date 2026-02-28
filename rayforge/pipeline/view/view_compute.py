from __future__ import annotations
import cairo
import logging
import math
import numpy as np
from typing import Optional, Tuple, Iterator, TYPE_CHECKING
from gettext import gettext as _
from ...shared.tasker.progress import ProgressContext
from ...shared.util.colors import ColorSet
from ..artifact import WorkPieceArtifact
from ..artifact.base import TextureData, VertexData
from ..artifact.workpiece_view import (
    RenderContext,
    WorkPieceViewArtifact,
)
from ..encoder.textureencoder import TextureEncoder
from ..encoder.vertexencoder import VertexEncoder

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

CAIRO_MAX_DIMENSION = 30000


def compute_view_dimensions(
    artifact: WorkPieceArtifact,
    context: RenderContext,
) -> Optional[Tuple[float, float, float, float, int, int]]:
    """
    Computes the bounding box and dimensions for rendering a workpiece view.

    This function calculates the content bounding box and the required
    pixel dimensions for rendering, handling all geometric calculations.

    Args:
        artifact: The WorkPieceArtifact to calculate dimensions for.
        context: The RenderContext containing rendering parameters.

    Returns:
        A tuple of (x_mm, y_mm, width_mm, height_mm, width_px, height_px),
        or None if no content to render.
    """
    encoder_vertex = VertexEncoder()
    encoder_texture = TextureEncoder()

    vertex_data = encoder_vertex.encode(artifact.ops)

    texture_data = None
    if not artifact.is_scalable:
        if artifact.source_dimensions:
            px_per_mm_x, px_per_mm_y = context.pixels_per_mm
            width_px = int(round(artifact.generation_size[0] * px_per_mm_x))
            height_px = int(round(artifact.generation_size[1] * px_per_mm_y))
            logger.debug(
                f"compute_view_dimensions: Using generation_size for "
                f"non-scalable artifact: "
                f"src_dims={artifact.source_dimensions}, "
                f"gen_size={artifact.generation_size}, "
                f"calc_px=({width_px}x{height_px}), "
                f"calc_ppm=({px_per_mm_x:.2f}, {px_per_mm_y:.2f})"
            )
        else:
            px_per_mm_x, px_per_mm_y = context.pixels_per_mm
            width_px = int(round(artifact.generation_size[0] * px_per_mm_x))
            height_px = int(round(artifact.generation_size[1] * px_per_mm_y))

        if width_px > 0 and height_px > 0:
            texture_buffer = encoder_texture.encode(
                artifact.ops,
                width_px,
                height_px,
                (px_per_mm_x, px_per_mm_y),
            )
            texture_data = TextureData(
                power_texture_data=texture_buffer,
                dimensions_mm=artifact.generation_size,
                position_mm=(0.0, 0.0),
            )

    bbox = _get_content_bbox(
        vertex_data, texture_data, context.show_travel_moves
    )
    if not bbox or bbox[2] <= 1e-9 or bbox[3] <= 1e-9:
        return None

    x_mm, y_mm, w_mm, h_mm = bbox

    dimensions = calculate_render_dimensions(bbox, context)
    if dimensions is None:
        return None

    width_px, height_px, _, _ = dimensions

    return (x_mm, y_mm, w_mm, h_mm, width_px, height_px)


def _encode_vertex_and_texture_data(
    artifact: WorkPieceArtifact,
    render_context: RenderContext,
) -> Tuple[VertexData, Optional[TextureData]]:
    """
    Encodes vertex and texture data from a WorkPieceArtifact.

    Args:
        artifact: The WorkPieceArtifact to encode.
        render_context: The RenderContext containing rendering parameters.

    Returns:
        A tuple of (vertex_data, texture_data). texture_data is None for
        scalable artifacts.
    """
    encoder_vertex = VertexEncoder()
    encoder_texture = TextureEncoder()

    vertex_data = encoder_vertex.encode(artifact.ops)

    texture_data = None
    if not artifact.is_scalable:
        if artifact.source_dimensions:
            source_px_per_mm_x = (
                artifact.source_dimensions[0] / artifact.generation_size[0]
            )
            source_px_per_mm_y = (
                artifact.source_dimensions[1] / artifact.generation_size[1]
            )
            context_px_per_mm_x, context_px_per_mm_y = (
                render_context.pixels_per_mm
            )
            px_per_mm_x = max(source_px_per_mm_x, context_px_per_mm_x)
            px_per_mm_y = max(source_px_per_mm_y, context_px_per_mm_y)
            width_px = int(round(artifact.generation_size[0] * px_per_mm_x))
            height_px = int(round(artifact.generation_size[1] * px_per_mm_y))
        else:
            px_per_mm_x, px_per_mm_y = render_context.pixels_per_mm
            width_px = int(round(artifact.generation_size[0] * px_per_mm_x))
            height_px = int(round(artifact.generation_size[1] * px_per_mm_y))

        if width_px > 0 and height_px > 0:
            texture_buffer = encoder_texture.encode(
                artifact.ops,
                width_px,
                height_px,
                (px_per_mm_x, px_per_mm_y),
            )
            texture_data = TextureData(
                power_texture_data=texture_buffer,
                dimensions_mm=artifact.generation_size,
                position_mm=(0.0, 0.0),
            )

    return vertex_data, texture_data


def calculate_render_dimensions(
    bbox: Tuple[float, float, float, float],
    render_context: RenderContext,
) -> Optional[Tuple[int, int, float, float]]:
    """
    Calculates pixel dimensions and effective pixels-per-mm for rendering.

    Args:
        bbox: The content bounding box (x, y, width, height) in mm.
        render_context: The RenderContext containing rendering parameters.

    Returns:
        A tuple of (width_px, height_px, effective_ppm_x, effective_ppm_y),
        or None if dimensions are invalid.
    """
    x_mm, y_mm, w_mm, h_mm = bbox
    ppm_x, ppm_y = render_context.pixels_per_mm
    margin = render_context.margin_px

    requested_width_px = round(w_mm * ppm_x) + 2 * margin
    requested_height_px = round(h_mm * ppm_y) + 2 * margin

    width_px = min(requested_width_px, CAIRO_MAX_DIMENSION)
    height_px = min(requested_height_px, CAIRO_MAX_DIMENSION)

    if width_px <= 2 * margin or height_px <= 2 * margin:
        return None

    effective_ppm_x = (width_px - 2 * margin) / w_mm if w_mm > 0 else ppm_x
    effective_ppm_y = (height_px - 2 * margin) / h_mm if h_mm > 0 else ppm_y

    return width_px, height_px, effective_ppm_x, effective_ppm_y


def _setup_cairo_context(
    bitmap: np.ndarray,
    bbox: Tuple[float, float, float, float],
    render_context: RenderContext,
) -> Tuple[cairo.Context, float]:
    """
    Sets up a cairo context with appropriate transforms for rendering.

    Args:
        bitmap: The numpy array to create the surface from.
        bbox: The content bounding box (x, y, width, height) in mm.
        render_context: The RenderContext containing rendering parameters.

    Returns:
        A tuple of (cairo_context, line_width_mm).
    """
    height_px, width_px = bitmap.shape[:2]
    x_mm, y_mm, w_mm, h_mm = bbox
    margin = render_context.margin_px
    ppm_x, ppm_y = render_context.pixels_per_mm

    effective_ppm_x = (width_px - 2 * margin) / w_mm if w_mm > 0 else ppm_x
    effective_ppm_y = (height_px - 2 * margin) / h_mm if h_mm > 0 else ppm_y

    surface = cairo.ImageSurface.create_for_data(
        memoryview(bitmap), cairo.FORMAT_ARGB32, width_px, height_px
    )
    ctx = cairo.Context(surface)

    ctx.translate(margin, height_px - margin)
    ctx.scale(effective_ppm_x, -effective_ppm_y)
    ctx.translate(-x_mm, -y_mm)

    line_width_mm = 1.0 / effective_ppm_x if effective_ppm_x > 0 else 1.0

    return ctx, line_width_mm


def _render_to_cairo(
    ctx: cairo.Context,
    vertex_data: VertexData,
    texture_data: Optional[TextureData],
    color_set: ColorSet,
    render_context: RenderContext,
    surface: cairo.ImageSurface,
    progress_context: Optional[ProgressContext] = None,
) -> None:
    """
    Renders texture and vertices to a cairo context.

    Args:
        ctx: The cairo context to render to.
        vertex_data: The vertex data to render.
        texture_data: Optional texture data to render.
        color_set: The color set for rendering.
        render_context: The RenderContext containing rendering parameters.
        surface: The cairo surface for flushing.
        progress_context: Optional ProgressContext for progress reporting.
    """

    def _set_progress(progress: float, message: str = ""):
        if progress_context:
            progress_context.set_progress(progress)
            progress_context.set_message(message)

    if texture_data:
        _draw_texture(ctx, texture_data, color_set)
        surface.flush()
        _set_progress(0.3, _("Rendering vertices..."))

    matrix_xx = ctx.get_matrix().xx
    line_width_mm = 1.0 / matrix_xx if matrix_xx > 0 else 1.0

    for progress, message in _draw_vertices_progressive(
        ctx,
        vertex_data,
        color_set,
        render_context.show_travel_moves,
        line_width_mm,
        surface,
        num_batches=3,
    ):
        _set_progress(0.3 + 0.7 * progress, message)


def compute_workpiece_view(
    artifact: WorkPieceArtifact,
    render_context: RenderContext,
    generation_id: int,
    progress_context: Optional[ProgressContext] = None,
) -> Optional[WorkPieceViewArtifact]:
    """
    Computes a WorkPieceViewArtifact from a WorkPieceArtifact.

    This is the core logic function that renders a WorkPieceArtifact to a
    bitmap for display on the 2D canvas.

    Args:
        artifact: The WorkPieceArtifact to render.
        render_context: The RenderContext containing rendering parameters.
        generation_id: The generation ID for staleness checking.
        progress_context: Optional ProgressContext for progress reporting.

    Returns:
        A WorkPieceViewArtifact containing the rendered bitmap, or None if
        no content to render.
    """

    def _set_progress(progress: float, message: str = ""):
        if progress_context:
            progress_context.set_progress(progress)
            progress_context.set_message(message)

    _set_progress(0.0, _("Preparing 2D preview..."))

    vertex_data, texture_data = _encode_vertex_and_texture_data(
        artifact, render_context
    )

    bbox = _get_content_bbox(
        vertex_data, texture_data, render_context.show_travel_moves
    )
    if not bbox or bbox[2] <= 1e-9 or bbox[3] <= 1e-9:
        logger.warning(f"No content to render (bbox={bbox}). Returning None.")
        return None

    dimensions = calculate_render_dimensions(bbox, render_context)
    if dimensions is None:
        return None

    width_px, height_px, effective_ppm_x, effective_ppm_y = dimensions

    bitmap = np.zeros(shape=(height_px, width_px, 4), dtype=np.uint8)
    surface = cairo.ImageSurface.create_for_data(
        memoryview(bitmap), cairo.FORMAT_ARGB32, width_px, height_px
    )
    ctx = cairo.Context(surface)

    x_mm, y_mm, w_mm, h_mm = bbox
    margin = render_context.margin_px

    ctx.translate(margin, height_px - margin)
    ctx.scale(effective_ppm_x, -effective_ppm_y)
    ctx.translate(-x_mm, -y_mm)

    color_set = ColorSet.from_dict(render_context.color_set_dict)

    _set_progress(0.1, _("Rendering texture..."))

    line_width_mm = 1.0 / effective_ppm_x if effective_ppm_x > 0 else 1.0

    if texture_data:
        _draw_texture(ctx, texture_data, color_set)
        surface.flush()
        _set_progress(0.3, _("Rendering vertices..."))

    for progress, message in _draw_vertices_progressive(
        ctx,
        vertex_data,
        color_set,
        render_context.show_travel_moves,
        line_width_mm,
        surface,
        num_batches=3,
    ):
        _set_progress(0.3 + 0.7 * progress, message)

    _set_progress(1.0, _("Rendering complete."))

    return WorkPieceViewArtifact(
        bitmap_data=bitmap,
        bbox_mm=bbox,
        workpiece_size_mm=artifact.generation_size,
        generation_id=generation_id,
    )


def compute_workpiece_view_to_buffer(
    artifact: WorkPieceArtifact,
    render_context: RenderContext,
    bitmap: np.ndarray,
    progress_context: Optional[ProgressContext] = None,
) -> Optional[Tuple[float, float, float, float]]:
    """
    Renders a WorkPieceArtifact directly into a pre-allocated bitmap buffer.

    This function renders into the provided buffer and yields progress updates
    after each batch, allowing for progressive rendering.

    Args:
        artifact: The WorkPieceArtifact to render.
        render_context: The RenderContext containing rendering parameters.
        bitmap: Pre-allocated numpy array to render into.
        progress_context: Optional ProgressContext for progress reporting.

    Returns:
        The bounding box (x, y, width, height) in mm, or None if
        no content to render.
    """

    def _set_progress(progress: float, message: str = ""):
        if progress_context:
            progress_context.set_progress(progress)
            progress_context.set_message(message)

    _set_progress(0.0, "Preparing 2D preview...")

    vertex_data, texture_data = _encode_vertex_and_texture_data(
        artifact, render_context
    )

    bbox = _get_content_bbox(
        vertex_data, texture_data, render_context.show_travel_moves
    )
    if not bbox or bbox[2] <= 1e-9 or bbox[3] <= 1e-9:
        logger.warning(f"No content to render (bbox={bbox}). Returning None.")
        return None

    dimensions = calculate_render_dimensions(bbox, render_context)
    if dimensions is None:
        return None

    width_px, height_px, effective_ppm_x, effective_ppm_y = dimensions

    height_px, width_px = bitmap.shape[:2]

    surface = cairo.ImageSurface.create_for_data(
        memoryview(bitmap), cairo.FORMAT_ARGB32, width_px, height_px
    )
    ctx = cairo.Context(surface)

    x_mm, y_mm, w_mm, h_mm = bbox
    margin = render_context.margin_px

    ctx.translate(margin, height_px - margin)
    ctx.scale(effective_ppm_x, -effective_ppm_y)
    ctx.translate(-x_mm, -y_mm)

    color_set = ColorSet.from_dict(render_context.color_set_dict)

    _set_progress(0.1, "Rendering texture...")

    line_width_mm = 1.0 / effective_ppm_x if effective_ppm_x > 0 else 1.0

    if texture_data:
        _draw_texture(ctx, texture_data, color_set)
        surface.flush()
        _set_progress(0.3, "Rendering vertices...")

    for progress, message in _draw_vertices_progressive(
        ctx,
        vertex_data,
        color_set,
        render_context.show_travel_moves,
        line_width_mm,
        surface,
        num_batches=3,
    ):
        _set_progress(0.3 + 0.7 * progress, message)

    _set_progress(1.0, "Rendering complete.")

    return bbox


def _draw_vertices(
    ctx: cairo.Context,
    vertex_data: VertexData,
    color_set: ColorSet,
    show_travel: bool,
    line_width_mm: float,
):
    """Draws all vertex data onto the provided cairo context."""
    ctx.save()
    ctx.set_line_width(line_width_mm)
    ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

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

    if vertex_data.powered_vertices.size > 0:
        powered_v = vertex_data.powered_vertices.reshape(-1, 2, 3)
        powered_c = vertex_data.powered_colors
        cut_lut = color_set.get_lut("cut")

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


def _draw_travel_vertices(
    ctx: cairo.Context,
    vertex_data: VertexData,
    color_set: ColorSet,
) -> None:
    """
    Draws travel vertices to the cairo context.

    Args:
        ctx: The cairo context to draw to.
        vertex_data: The vertex data containing travel vertices.
        color_set: The color set for rendering.
    """
    if vertex_data.travel_vertices.size == 0:
        return

    travel_v = vertex_data.travel_vertices.reshape(-1, 2, 3)
    ctx.set_source_rgba(*color_set.get_rgba("travel"))
    for start, end in travel_v:
        ctx.move_to(start[0], start[1])
        ctx.line_to(end[0], end[1])
    ctx.stroke()


def _draw_zero_power_vertices(
    ctx: cairo.Context,
    vertex_data: VertexData,
    color_set: ColorSet,
) -> None:
    """
    Draws zero-power vertices to the cairo context.

    Args:
        ctx: The cairo context to draw to.
        vertex_data: The vertex data containing zero-power vertices.
        color_set: The color set for rendering.
    """
    if vertex_data.zero_power_vertices.size == 0:
        return

    zero_v = vertex_data.zero_power_vertices.reshape(-1, 2, 3)
    ctx.set_source_rgba(*color_set.get_rgba("zero_power"))
    for start, end in zero_v:
        ctx.move_to(start[0], start[1])
        ctx.line_to(end[0], end[1])
    ctx.stroke()


def _prepare_powered_vertices_for_batching(
    vertex_data: VertexData,
    color_set: ColorSet,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepares powered vertices for batched rendering.

    Args:
        vertex_data: The vertex data containing powered vertices.
        color_set: The color set for rendering.

    Returns:
        A tuple of (powered_vertices, unique_colors, inverse_indices).
    """
    powered_v = vertex_data.powered_vertices.reshape(-1, 2, 3)
    powered_c = vertex_data.powered_colors
    cut_lut = color_set.get_lut("cut")

    power_indices = (powered_c[::2, 0] * 255.0).astype(np.uint8)
    themed_colors_per_segment = cut_lut[power_indices]
    unique_colors, inverse_indices = np.unique(
        themed_colors_per_segment, axis=0, return_inverse=True
    )

    return powered_v, unique_colors, inverse_indices


def _draw_powered_vertices_batch(
    ctx: cairo.Context,
    powered_v: np.ndarray,
    unique_colors: np.ndarray,
    inverse_indices: np.ndarray,
    start_seg: int,
    end_seg: int,
) -> None:
    """
    Draws a batch of powered vertices to the cairo context.

    Args:
        ctx: The cairo context to draw to.
        powered_v: The powered vertices array.
        unique_colors: Array of unique colors.
        inverse_indices: Indices mapping segments to colors.
        start_seg: Starting segment index.
        end_seg: Ending segment index.
    """
    for seg_idx in range(start_seg, end_seg):
        start, end = powered_v[seg_idx]
        color_idx = inverse_indices[seg_idx]
        color = unique_colors[color_idx]
        ctx.set_source_rgba(*color)
        ctx.move_to(start[0], start[1])
        ctx.line_to(end[0], end[1])
        ctx.stroke()


def _draw_vertices_progressive(
    ctx: cairo.Context,
    vertex_data: VertexData,
    color_set: ColorSet,
    show_travel: bool,
    line_width_mm: float,
    surface,
    num_batches: int = 3,
) -> Iterator[Tuple[float, str]]:
    """
    Draws vertex data in multiple batches for progressive rendering.

    Yields progress updates after each batch.

    Yields:
        Tuples of (progress: float, message: str).
    """
    ctx.save()
    ctx.set_line_width(line_width_mm)
    ctx.set_line_cap(cairo.LINE_CAP_SQUARE)

    batch_num = 0
    total_batches = num_batches

    if show_travel:
        _draw_travel_vertices(ctx, vertex_data, color_set)
        _draw_zero_power_vertices(ctx, vertex_data, color_set)

        surface.flush()
        batch_num += 1
        progress = batch_num / total_batches
        yield progress, "Rendered travel moves"

    if vertex_data.powered_vertices.size > 0:
        powered_v, unique_colors, inverse_indices = (
            _prepare_powered_vertices_for_batching(vertex_data, color_set)
        )

        remaining_batches = total_batches - batch_num
        if remaining_batches < 1:
            remaining_batches = 1

        total_segments = len(powered_v)
        segments_per_batch = max(1, total_segments // remaining_batches)

        for batch_idx in range(remaining_batches):
            start_seg = batch_idx * segments_per_batch
            end_seg = (
                start_seg + segments_per_batch
                if batch_idx < remaining_batches - 1
                else total_segments
            )

            if start_seg >= total_segments:
                continue

            _draw_powered_vertices_batch(
                ctx,
                powered_v,
                unique_colors,
                inverse_indices,
                start_seg,
                end_seg,
            )

            surface.flush()
            batch_num += 1
            progress = batch_num / total_batches
            yield progress, f"Rendered powered batch {batch_idx + 1}"

    ctx.restore()


def _draw_texture(
    ctx: cairo.Context,
    texture_data: TextureData,
    color_set: ColorSet,
):
    """Draws themed texture data onto the provided cairo context."""
    power_data = texture_data.power_texture_data
    if power_data.size == 0:
        return

    engrave_lut = color_set.get_lut("engrave")
    rgba_texture = engrave_lut[power_data]
    rgba_texture[power_data == 0, 3] = 0.0

    h, w = rgba_texture.shape[:2]
    alpha_ch = rgba_texture[..., 3, np.newaxis]
    rgb_ch = rgba_texture[..., :3]
    bgra_texture = np.empty((h, w, 4), dtype=np.uint8)
    premultiplied_rgb = rgb_ch * alpha_ch * 255
    premultiplied_rgb_int = np.clip(np.round(premultiplied_rgb), 0, 255)
    premultiplied_rgb_int = premultiplied_rgb_int.astype(np.uint8)

    bgra_texture[..., 0] = premultiplied_rgb_int[..., 2]
    bgra_texture[..., 1] = premultiplied_rgb_int[..., 1]
    bgra_texture[..., 2] = premultiplied_rgb_int[..., 0]
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

    ctx.translate(pos_mm[0], pos_mm[1] + dim_mm[1])
    ctx.scale(1, -1)
    ctx.scale(dim_mm[0] / w, dim_mm[1] / h)

    ctx.set_source_surface(texture_surface, 0, 0)
    ctx.get_source().set_filter(cairo.FILTER_GOOD)
    ctx.paint()
    ctx.restore()


def render_chunk_to_buffer(
    artifact: WorkPieceArtifact,
    render_context: RenderContext,
    bitmap: np.ndarray,
    view_bbox_mm: Tuple[float, float, float, float],
) -> bool:
    """
    Renders a chunk WorkPieceArtifact directly into a pre-allocated bitmap.

    This function renders a chunk into the provided buffer, used for
    progressive chunk rendering where chunks are incrementally added
    to a view artifact.

    Args:
        artifact: The WorkPieceArtifact chunk to render.
        render_context: The RenderContext containing rendering parameters.
        bitmap: Pre-allocated numpy array to render into.
        view_bbox_mm: The bounding box (x, y, width, height) in mm of the
            view artifact being rendered to.

    Returns:
        True if rendering succeeded, False otherwise.
    """
    vertex_data, texture_data = _encode_vertex_and_texture_data(
        artifact, render_context
    )

    ctx, line_width_mm = _setup_cairo_context(
        bitmap, view_bbox_mm, render_context
    )

    color_set = ColorSet.from_dict(render_context.color_set_dict)

    if texture_data:
        _draw_texture(ctx, texture_data, color_set)
        surface = ctx.get_target()
        surface.flush()

    _draw_vertices(
        ctx,
        vertex_data,
        color_set,
        render_context.show_travel_moves,
        line_width_mm,
    )

    surface = ctx.get_target()
    surface.flush()
    return True


def _get_content_bbox(
    vertex_data: Optional[VertexData],
    texture_data: Optional[TextureData],
    show_travel: bool,
) -> Optional[Tuple[float, float, float, float]]:
    """Calculate the union bounding box of all visual content."""
    all_vertices = []
    has_content = False

    if vertex_data:
        v_data = vertex_data
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

    if texture_data:
        tex = texture_data
        t_x1, t_y1 = tex.position_mm
        t_x2, t_y2 = t_x1 + tex.dimensions_mm[0], t_y1 + tex.dimensions_mm[1]
        v_x1, v_x2 = min(v_x1, t_x1), max(v_x2, t_x2)
        v_y1, v_y2 = min(v_y1, t_y1), max(v_y2, t_y2)
        has_content = True

    if not has_content:
        return None

    return (v_x1, v_y1, v_x2 - v_x1, v_y2 - v_y1)
