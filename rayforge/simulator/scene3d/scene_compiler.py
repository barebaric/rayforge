"""
Scene compiler: thin wrapper that delegates vertex compilation to
raygeo's Rust ``compile_scene_3d`` and handles texture generation.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
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


# ── Texture generation ─────────────────────────


def _rasterize_scanlines(
    ops: Ops,
    bbox: tuple[float, float, float, float],
) -> Optional[tuple[np.ndarray, int, int, float]]:
    x0, y0, w_mm, h_mm = bbox
    if w_mm <= 0 or h_mm <= 0:
        return None

    px_per_mm = PX_PER_MM
    width_px = int(round(w_mm * px_per_mm))
    height_px = int(round(h_mm * px_per_mm))

    if width_px > MAX_TEXTURE_DIMENSION or height_px > MAX_TEXTURE_DIMENSION:
        scale = min(
            MAX_TEXTURE_DIMENSION / width_px,
            MAX_TEXTURE_DIMENSION / height_px,
        )
        px_per_mm *= scale
        width_px = int(round(w_mm * px_per_mm))
        height_px = int(round(h_mm * px_per_mm))

    if width_px <= 0 or height_px <= 0:
        return None

    buffer = rasterize_scanlines(
        ops,
        width_px,
        height_px,
        (px_per_mm, px_per_mm),
        origin_mm=(x0, y0),
    )
    if not np.any(buffer):
        return None

    dilated = np.zeros_like(buffer)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            shifted = np.roll(np.roll(buffer, dy, axis=0), dx, axis=1)
            np.maximum(dilated, shifted, out=dilated)
    buffer = dilated

    return buffer, width_px, height_px, px_per_mm


def _generate_texture_layers(
    ops: Ops,
    layer_infos: list[LayerInfo],
    config: RenderConfig3D,
) -> list[TextureLayer]:
    texture_layers: list[TextureLayer] = []

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

        raster_result = _rasterize_scanlines(layer_ops, bbox)
        if raster_result is None:
            continue

        tex_buf, w_px, h_px, actual_ppm = raster_result
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
    cancel_check: Optional[Callable[[], bool]] = None,
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
