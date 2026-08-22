"""
Scene compiler: thin wrapper that delegates vertex compilation to
raygeo's Rust ``compile_scene_3d`` and packs the stock meshes.

Engraving is visualized by the burned-in stock only; the Ops lines and
the scanline ring buffer carry the toolpath/trail. No LUT engrave
underlay quads are produced.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
from raygeo.ops import Ops

from .compiled_scene import (
    CompiledSceneArtifact,
    ScanlineOverlayLayer,
    VertexLayer,
)
from .render_config import RenderConfig3D
from .stock_compiler import compile_stock_layers

logger = logging.getLogger(__name__)


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

    stock_w2v = (
        config.stock_world_to_visual
        if config.stock_world_to_visual is not None
        else config.world_to_visual
    )
    stock_layers = compile_stock_layers(config.stock_specs or [], stock_w2v)

    return CompiledSceneArtifact(
        generation_id=generation_id,
        vertex_layers=vertex_layers,
        overlay_layers=overlay_layers,
        laser_uid_order=raw.laser_uid_order,
        stock_layers=stock_layers,
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
    artifact = _wrap_compiled_scene(raw, config, generation_id=generation_id)

    return artifact
