from __future__ import annotations

import logging
import math
from typing import Optional, Tuple

import numpy as np
from raygeo.geo.types import Rect
from raygeo.ops.convert import ViewSpec
from raygeo.ops.convert.view import render_ops as raygeo_render_ops
from raygeo.ops.convert.view import render_ops_into as raygeo_render_ops_into

from ...core.color import ColorSet
from ...core.config import OpsColorMode
from ..artifact import WorkPieceArtifact
from ..artifact.workpiece_view import (
    RenderContext,
)
from ..encoder.vertexencoder import VertexEncoder

logger = logging.getLogger(__name__)

CAIRO_MAX_DIMENSION = 8192
MAX_TOTAL_PIXELS = CAIRO_MAX_DIMENSION * CAIRO_MAX_DIMENSION


def _resolve_color_set(
    render_context: RenderContext,
    laser_uid: Optional[str] = None,
    layer_uid: Optional[str] = None,
) -> ColorSet:
    """
    Resolve the appropriate ColorSet based on the ops color mode.

    When ops_color_mode is OpsColorMode.LAYER, uses layer_color_sets
    keyed by layer_uid.  Otherwise falls back to laser-specific or
    default colors.
    """
    if render_context.ops_color_mode == OpsColorMode.LAYER and layer_uid:
        if layer_uid in render_context.layer_color_sets:
            logger.debug(
                f"_resolve_color_set: using layer color for "
                f"layer_uid={layer_uid}"
            )
            return ColorSet.from_dict(
                render_context.layer_color_sets[layer_uid]
            )
        logger.warning(
            f"_resolve_color_set: layer_uid={layer_uid} "
            f"not in layer_color_sets, falling back"
        )
    if laser_uid and laser_uid in render_context.laser_color_sets:
        return ColorSet.from_dict(render_context.laser_color_sets[laser_uid])
    return ColorSet.from_dict(render_context.color_set_dict)


# ──────────────────────────────────────────────────────────────────
# Bounding box
# ──────────────────────────────────────────────────────────────────


def _get_content_bbox(
    artifact: WorkPieceArtifact,
    show_travel: bool,
) -> Optional[Rect]:
    """Calculate the union bounding box of all visual content."""
    encoder = VertexEncoder()
    vertex_data = encoder.encode(artifact.ops)

    all_vertices = []
    has_content = False

    if vertex_data.powered_vertices.size > 0:
        all_vertices.append(vertex_data.powered_vertices)
    if show_travel:
        if vertex_data.travel_vertices.size > 0:
            all_vertices.append(vertex_data.travel_vertices)
        if vertex_data.zero_power_vertices.size > 0:
            all_vertices.append(vertex_data.zero_power_vertices)

    v_stack = np.vstack(all_vertices) if all_vertices else None
    if v_stack is not None:
        v_x1, v_y1, _ = np.min(v_stack, axis=0)
        v_x2, v_y2, _ = np.max(v_stack, axis=0)
        has_content = True
    else:
        v_x1, v_y1 = math.inf, math.inf
        v_x2, v_y2 = -math.inf, -math.inf

    if not artifact.is_scalable:
        t_x1, t_y1 = 0.0, 0.0
        t_x2 = artifact.generation_size[0]
        t_y2 = artifact.generation_size[1]
        v_x1 = min(v_x1, t_x1)
        v_x2 = max(v_x2, t_x2)
        v_y1 = min(v_y1, t_y1)
        v_y2 = max(v_y2, t_y2)
        has_content = True

    if not has_content:
        return None

    return (v_x1, v_y1, v_x2 - v_x1, v_y2 - v_y1)


def calculate_render_dimensions(
    bbox: Rect,
    render_context: RenderContext,
) -> Optional[Tuple[int, int, float, float]]:
    """
    Calculates pixel dimensions and effective pixels-per-mm for rendering.

    The caller is responsible for including any desired padding in
    *bbox* before calling.  No implicit margin is added.

    Args:
        bbox: The content bounding box (x, y, width, height) in mm.
        render_context: The RenderContext containing rendering parameters.

    Returns:
        ``(width_px, height_px, effective_ppm_x, effective_ppm_y)``
        or ``None`` if dimensions are invalid.
    """
    _, _, w_mm, h_mm = bbox
    ppm_x, ppm_y = render_context.pixels_per_mm

    width_px = min(int(round(w_mm * ppm_x)), CAIRO_MAX_DIMENSION)
    height_px = min(int(round(h_mm * ppm_y)), CAIRO_MAX_DIMENSION)

    if width_px * height_px > MAX_TOTAL_PIXELS:
        scale = (MAX_TOTAL_PIXELS / (width_px * height_px)) ** 0.5
        width_px = max(1, int(width_px * scale))
        height_px = max(1, int(height_px * scale))

    if width_px <= 0 or height_px <= 0:
        return None

    eff_ppm_x = width_px / w_mm if w_mm > 0 else ppm_x
    eff_ppm_y = height_px / h_mm if h_mm > 0 else ppm_y

    return width_px, height_px, eff_ppm_x, eff_ppm_y


# ──────────────────────────────────────────────────────────────────
# ViewSpec construction
# ──────────────────────────────────────────────────────────────────


def _make_view_spec(
    render_context: RenderContext,
    color_set: ColorSet,
    render_bbox_mm: Tuple[float, float, float, float],
) -> ViewSpec:
    """Build a raygeo ViewSpec from the render context and colour set."""
    return ViewSpec(
        pixels_per_mm=render_context.pixels_per_mm,
        render_bbox=render_bbox_mm,
        show_travel_moves=render_context.show_travel_moves,
        cut_color=color_set.get_argb32("cut"),
        travel_color=color_set.get_argb32("travel"),
        zero_power_color=color_set.get_argb32("zero_power"),
        cut_lut=color_set.get_lut_argb32("cut").tolist(),
        engrave_lut=color_set.get_lut_argb32("engrave").tolist(),
        max_dimension_px=CAIRO_MAX_DIMENSION,
        max_total_pixels=MAX_TOTAL_PIXELS,
    )


def _expand_bbox_by_px(
    bbox: Rect,
    ppm: Tuple[float, float],
    margin_px: int,
) -> Tuple[float, float, float, float]:
    """Expand a ``(x, y, w, h)`` bbox by ``margin_px`` on each side,
    returning ``(min_x, min_y, max_x, max_y)``."""
    x, y, w, h = bbox
    ppm_x, ppm_y = ppm
    mx = margin_px / ppm_x if ppm_x > 0 else 0
    my = margin_px / ppm_y if ppm_y > 0 else 0
    return (x - mx, y - my, x + w + mx, y + h + my)


# ──────────────────────────────────────────────────────────────────
# Public entry points
# ──────────────────────────────────────────────────────────────────


def render_workpiece_view_in_process(
    artifact: WorkPieceArtifact,
    render_context: RenderContext,
    laser_uid: Optional[str] = None,
    layer_uid: Optional[str] = None,
) -> Optional[Tuple[np.ndarray, Rect, Tuple[float, float]]]:
    """
    Render a WorkPieceArtifact into a view bitmap in-process.

    Calls ``raygeo.render_ops`` directly and returns
    ``(bitmap, bbox_mm, workpiece_size_mm)`` — no shared memory,
    no artifact store.

    Args:
        artifact: The WorkPieceArtifact to render.
        render_context: The RenderContext containing rendering parameters.
        laser_uid: Optional laser UID for color lookup.
        layer_uid: Optional layer UID for color lookup.

    Returns:
        ``(bitmap, bbox_mm, workpiece_size_mm)`` or ``None`` when
        there is no content to render.
    """
    bbox = _get_content_bbox(artifact, render_context.show_travel_moves)
    if not bbox or bbox[2] <= 1e-9 or bbox[3] <= 1e-9:
        return None

    color_set = _resolve_color_set(render_context, laser_uid, layer_uid)

    # Expand the content bbox by margin_px on each side so strokes
    # at the edge are not clipped.  The expanded bbox is what raygeo
    # renders — no implicit margin inside raygeo.
    render_bbox_mm = _expand_bbox_by_px(
        bbox, render_context.pixels_per_mm, render_context.margin_px
    )

    spec = _make_view_spec(render_context, color_set, render_bbox_mm)
    result = raygeo_render_ops(artifact.ops, spec)
    if result is None:
        return None

    return (
        np.asarray(result.bitmap, dtype=np.uint8),
        bbox,
        artifact.generation_size,
    )


# ──────────────────────────────────────────────────────────────────
# Chunk rendering (progressive)
# ──────────────────────────────────────────────────────────────────


def stitch_chunk_to_bitmap(
    chunk_artifact: WorkPieceArtifact,
    render_context: RenderContext,
    view_bitmap: np.ndarray,
    view_bbox_mm: Rect,
    laser_uid: Optional[str] = None,
    layer_uid: Optional[str] = None,
) -> bool:
    """
    Stitch a chunk artifact into a view bitmap in-process.

    Renders the chunk's texture and vertex strokes directly into
    *view_bitmap* via ``raygeo.render_ops_into``, using *view_bbox_mm*
    as the coordinate-space area the bitmap covers.

    Args:
        chunk_artifact: The WorkPieceArtifact chunk to render.
        render_context: The RenderContext containing rendering parameters.
        view_bitmap: Pre-allocated numpy array to render into.
        view_bbox_mm: The bounding box (x, y, width, height) in mm of
            the view artifact being rendered to.
        laser_uid: Optional laser UID for color lookup.
        layer_uid: Optional layer UID for color lookup.

    Returns:
        True if rendering succeeded, False otherwise.
    """
    color_set = _resolve_color_set(render_context, laser_uid, layer_uid)

    # render_ops_into ignores pixels_per_mm and render_bbox from the
    # spec — it derives ppm from the bitmap dimensions and view_bbox.
    # We still pass the spec for colours, LUTs, and show_travel.
    spec = _make_view_spec(
        render_context,
        color_set,
        (0.0, 0.0, 0.0, 0.0),  # unused by render_ops_into
    )

    min_x, min_y, w_mm, h_mm = view_bbox_mm
    view_bbox = (min_x, min_y, min_x + w_mm, min_y + h_mm)

    return raygeo_render_ops_into(
        chunk_artifact.ops, spec, view_bitmap, view_bbox
    )
