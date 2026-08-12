"""
Scene compiler: thin wrapper that delegates vertex compilation to
raygeo's Rust ``compile_scene_3d`` and handles texture generation.
"""

import logging
from collections.abc import Callable

import numpy as np
from raygeo.compressed_array import CompressedArray
from raygeo.image import rasterize_scanlines
from raygeo.ops import LayerInfo, Ops

from .compiled_scene import (
    CompiledSceneArtifact,
    ScanlineOverlayLayer,
    TextureLayer,
    VertexLayer,
)
from .cylinder_compiler import generate_cylinder_vertices
from .render_config import RenderConfig3D

logger = logging.getLogger(__name__)

MAX_TEXTURE_DIMENSION = 8192
PX_PER_MM = 50.0

# Fallback laser dot width (mm) when no laser head info is available,
# matching the minimum sane spot size used elsewhere in the codebase.
DEFAULT_DOT_WIDTH_MM = 0.1


# ── Texture generation ─────────────────────────


def _rasterize_scanlines(
    ops: Ops,
    bbox: tuple[float, float, float, float],
    dot_width_mm: float,
) -> tuple[CompressedArray, int, int, float] | None:
    x0, y0, w_mm, h_mm = bbox
    if w_mm <= 0 or h_mm <= 0:
        return None

    px_per_mm = PX_PER_MM
    width_px = round(w_mm * px_per_mm)
    height_px = round(h_mm * px_per_mm)

    if width_px > MAX_TEXTURE_DIMENSION or height_px > MAX_TEXTURE_DIMENSION:
        scale = min(
            MAX_TEXTURE_DIMENSION / width_px,
            MAX_TEXTURE_DIMENSION / height_px,
        )
        px_per_mm *= scale
        width_px = round(w_mm * px_per_mm)
        height_px = round(h_mm * px_per_mm)

    if width_px <= 0 or height_px <= 0:
        return None

    dot_width_px = dot_width_mm * px_per_mm
    radius_px = max(0, int((dot_width_px - 1) / 2))

    buffer = rasterize_scanlines(
        ops,
        width_px,
        height_px,
        (px_per_mm, px_per_mm),
        origin_mm=(x0, y0),
        radius_px=radius_px,
    )
    if not isinstance(buffer, CompressedArray):
        return None

    return buffer, width_px, height_px, px_per_mm


def _generate_texture_layers(
    ops: Ops,
    layer_infos: list[LayerInfo],
    config: RenderConfig3D,
) -> list[TextureLayer]:
    texture_layers: list[TextureLayer] = []
    dot_widths = config.laser_dot_widths_mm or {}

    for li in layer_infos:
        if not li.has_scanlines:
            continue

        layer_ops = ops.extract_range(li.cmd_start, li.cmd_end)

        is_rot = li.is_rotary

        if is_rot:
            layer_ops = layer_ops.bake_visual_positions()

        bbox = layer_ops.scanline_bbox()
        if bbox is None:
            continue

        dot_width_mm = dot_widths.get(li.scanline_laser)
        if dot_width_mm is None:
            dot_width_mm = DEFAULT_DOT_WIDTH_MM
        raster_result = _rasterize_scanlines(layer_ops, bbox, dot_width_mm)
        if raster_result is None:
            continue

        tex_buf, w_px, h_px, _actual_ppm = raster_result
        x0, y0, bw, bh = bbox

        diameter = li.diameter

        if is_rot and diameter > 0:
            tex_transform = np.eye(4, dtype=np.float32)
        else:
            tex_transform = config.world_to_visual

        model = np.eye(4, dtype=np.float32)
        model[0, 0] = bw
        model[1, 1] = bh
        model[0, 3] = x0
        model[1, 3] = y0
        final_model = (tex_transform @ model).astype(np.float32)

        cyl_verts = None
        if is_rot and diameter > 0:
            cyl_verts = generate_cylinder_vertices(
                grid_matrix=final_model,
                diameter=diameter,
            )

        texture_layers.append(
            TextureLayer(
                power_texture=tex_buf,
                width_px=w_px,
                height_px=h_px,
                model_matrix=final_model,
                cylinder_vertices=cyl_verts,
                rotary_diameter=diameter,
                rotary_enabled=is_rot,
                activation_cmd_idx=li.activation_cmd_idx,
                laser_uid=li.scanline_laser,
            )
        )

    return texture_layers


# ── Spec building ─────────────────────────────────────────────────


def _build_scene_spec(config: RenderConfig3D) -> tuple[list, dict]:
    w2v = config.world_to_visual.astype(np.float32).tolist()
    layer_configs = {}
    if config.layer_configs:
        for uid, lc in config.layer_configs.items():
            layer_configs[uid] = {
                "rotary_enabled": lc.rotary_enabled,
                "rotary_diameter": lc.rotary_diameter,
                "axis_position": lc.axis_position,
                "reverse": lc.reverse,
            }
    return w2v, layer_configs


# ── Output wrapping ──────────────────────────────────────────────


def _wrap_compiled_scene(
    raw,
    ops: Ops,
    config: RenderConfig3D,
    generation_id: int = 0,
) -> CompiledSceneArtifact:
    vertex_layers: list[VertexLayer] = []
    overlay_layers: list[ScanlineOverlayLayer] = []

    for g in raw.groups:
        vertex_layers.append(
            VertexLayer(
                powered_verts=g.powered_verts,
                powered_attrib=g.powered_attrib,
                travel_verts=g.travel_verts,
                zero_power_verts=g.zero_power_verts,
                powered_cmd_offsets=g.powered_cmd_offsets,
                travel_cmd_offsets=g.travel_cmd_offsets,
                is_rotary=g.is_rotary,
            )
        )
        overlay_layers.append(
            ScanlineOverlayLayer(
                positions=g.overlay_positions,
                overlay_attrib=g.overlay_attrib,
                cmd_offsets=g.overlay_cmd_offsets,
                is_rotary=g.is_rotary,
            )
        )

    layer_infos = raw.layer_infos
    texture_layers = _generate_texture_layers(ops, layer_infos, config)

    return CompiledSceneArtifact(
        generation_id=generation_id,
        vertex_layers=vertex_layers,
        texture_layers=texture_layers,
        overlay_layers=overlay_layers,
        laser_uid_order=raw.laser_uid_order,
    )


# ── Public API ───────────────────────────────────────────────────


def compile_scene(
    ops: Ops,
    config: RenderConfig3D,
    cancel_check: Callable[[], bool] | None = None,
    generation_id: int = 0,
) -> CompiledSceneArtifact:
    if cancel_check is not None and cancel_check():
        raise RuntimeError("Cancelled")

    w2v, layer_configs = _build_scene_spec(config)
    raw = ops.compile_scene_3d(w2v, layer_configs)
    artifact = _wrap_compiled_scene(
        raw, ops, config, generation_id=generation_id
    )

    return artifact
