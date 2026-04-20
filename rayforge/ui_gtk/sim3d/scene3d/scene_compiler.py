"""
Scene compiler: pure function that compiles assembled job Ops into
GPU-ready vertex data.

Walks the full assembled job ops (with JobStart/End, LayerStart/End
markers) to produce vertex buffers and per-command offset arrays that
are indexed 1:1 with the player's command index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ....core.geo.arc import linearize_arc
from ....core.geo.bezier import linearize_bezier_segment
from ....core.ops import Ops
from ....core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    LayerEndCommand,
    LayerStartCommand,
    LineToCommand,
    MoveToCommand,
    MovingCommand,
    ScanLinePowerCommand,
    SetLaserCommand,
    SetPowerCommand,
)
from ....pipeline.encoder.vertexencoder import transform_to_cylinder
from .compiled_scene import (
    CompiledSceneArtifact,
    ScanlineOverlayLayer,
    TextureLayer,
    VertexLayer,
)
from ....pipeline.encoder.scanline_rasterizer import (
    MAX_TEXTURE_DIMENSION,
    rasterize_scanlines as _rasterize_scanlines_shared,
)
from .cylinder_compiler import generate_cylinder_vertices
from .render_config import RenderConfig3D
from .rotary_coords import (
    bake_visual_positions as _bake_visual_positions,
    mu_to_visual as _mu_to_visual,
    reconstruct_mu_arc as _reconstruct_mu_arc,
    reconstruct_mu_bezier as _reconstruct_mu_bezier,
    reconstruct_mu_pos as _reconstruct_mu_pos,
    visual_end as _visual_end,
)

logger = logging.getLogger(__name__)

Z_OFFSET_NON_POWERED = 0.01


def _transform_verts(verts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if verts.size == 0:
        return verts
    pts = verts.reshape(-1, 3)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    transformed = (transform @ hom.T).T
    return transformed[:, :3].astype(np.float32)


def _remap_offsets(
    offsets: List[int],
    pair_expansions: List[Tuple[int, "np.ndarray"]],
) -> List[int]:
    if not pair_expansions:
        return offsets
    result = []
    for pre_count in offsets:
        mapped = pre_count
        for seg_start, cum_subs in pair_expansions:
            num_input_pairs = len(cum_subs) - 1
            num_input_verts = num_input_pairs * 2
            if pre_count <= seg_start:
                continue
            if pre_count >= seg_start + num_input_verts:
                mapped += int(cum_subs[-1]) * 2 - num_input_verts
            else:
                vert_offset = pre_count - seg_start
                pair_idx = min(vert_offset // 2, num_input_pairs)
                extra_verts = vert_offset % 2
                mapped = seg_start + int(cum_subs[pair_idx]) * 2 + extra_verts
                break
        result.append(mapped)
    return result


def _apply_cylinder(
    verts: np.ndarray,
    diameter: float,
    colors: Optional[np.ndarray] = None,
    degrees_input: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    if verts.size == 0 or diameter <= 0:
        return verts, colors, np.array([0], dtype=np.int32)
    result_verts, result_colors, cum_subs = transform_to_cylinder(
        verts.reshape(-1, 3),
        diameter,
        colors,
        degrees_input=degrees_input,
    )
    return result_verts, result_colors, cum_subs


def _to_arr(raw: List[float], cols: int) -> np.ndarray:
    if not raw:
        return np.empty((0, cols), dtype=np.float32)
    return np.array(raw, dtype=np.float32).reshape(-1, cols)


def _to_flat(raw: np.ndarray) -> np.ndarray:
    if raw.size == 0:
        return np.empty((0,), dtype=np.float32)
    return raw.ravel().astype(np.float32)


def _extract_zero_power_segments(
    cmd: ScanLinePowerCommand,
    start_pos: Tuple[float, float, float],
    zero_power_v: List[float],
    end_pos: Optional[Tuple[float, float, float]] = None,
) -> None:
    if cmd.end is None:
        return
    num_steps = len(cmd.power_values)
    if num_steps == 0:
        return

    p_start = np.array(start_pos, dtype=np.float32)
    end = end_pos if end_pos is not None else cmd.end
    p_end = np.array(end, dtype=np.float32)
    line_vec = p_end - p_start

    is_zero = np.frombuffer(cmd.power_values, dtype=np.uint8) == 0
    padded = np.concatenate(([False], is_zero, [False]))
    diffs = np.diff(padded.astype(int))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    for s, e in zip(starts, ends):
        t_start = s / num_steps
        t_end = e / num_steps
        chunk_start = p_start + t_start * line_vec
        chunk_end = p_start + t_end * line_vec
        zero_power_v.extend(chunk_start)
        zero_power_v.extend(chunk_end)


def _encode_overlay_segments(
    cmd: ScanLinePowerCommand,
    start_pos: Tuple[float, float, float],
    ov_pos: List[float],
    ov_pow: List[float],
    ov_lid: List[int],
    laser_index: int = 0,
    end_pos: Optional[Tuple[float, float, float]] = None,
) -> int:
    if cmd.end is None:
        return 0
    num_steps = len(cmd.power_values)
    if num_steps == 0:
        return 0

    sx, sy, sz = start_pos
    end = end_pos if end_pos is not None else cmd.end
    ex, ey, ez = end
    dx = (ex - sx) / num_steps
    dy = (ey - sy) / num_steps
    dz = (ez - sz) / num_steps

    vertex_count = 0
    prev_power_on = False
    seg_start_x = 0.0
    seg_start_y = 0.0
    seg_start_z = 0.0
    seg_power = 0.0

    for i in range(num_steps):
        power_byte = cmd.power_values[i]
        power_on = power_byte > 0

        if power_on and not prev_power_on:
            seg_start_x = sx + i * dx
            seg_start_y = sy + i * dy
            seg_start_z = sz + i * dz
            seg_power = power_byte / 255.0
        elif not power_on and prev_power_on:
            seg_end_x = sx + i * dx
            seg_end_y = sy + i * dy
            seg_end_z = sz + i * dz
            ov_pos.extend(
                [
                    seg_start_x,
                    seg_start_y,
                    seg_start_z,
                    seg_end_x,
                    seg_end_y,
                    seg_end_z,
                ]
            )
            ov_pow.append(seg_power)
            ov_pow.append(seg_power)
            ov_lid.append(laser_index)
            ov_lid.append(laser_index)
            vertex_count += 2

        prev_power_on = power_byte > 0

    if prev_power_on:
        ov_pos.extend([seg_start_x, seg_start_y, seg_start_z, ex, ey, ez])
        ov_pow.append(seg_power)
        ov_pow.append(seg_power)
        ov_lid.append(laser_index)
        ov_lid.append(laser_index)
        vertex_count += 2

    return vertex_count


PX_PER_MM = 50.0


def _scanline_bbox(ops: Ops) -> Optional[Tuple[float, float, float, float]]:
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    has_scanlines = False
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for cmd in ops.commands:
        if isinstance(cmd, MoveToCommand):
            if cmd.end is not None:
                current_pos = cmd.end
        elif isinstance(cmd, ScanLinePowerCommand):
            if cmd.end is None:
                continue
            has_scanlines = True
            sx, sy = current_pos[0], current_pos[1]
            ex, ey = cmd.end[0], cmd.end[1]
            min_x = min(min_x, sx, ex)
            min_y = min(min_y, sy, ey)
            max_x = max(max_x, sx, ex)
            max_y = max(max_y, sy, ey)
            current_pos = cmd.end

    if not has_scanlines:
        return None
    return (min_x, min_y, max_x - min_x, max_y - min_y)


def _rasterize_scanlines(
    ops: Ops,
    bbox: Tuple[float, float, float, float],
) -> Optional[Tuple[np.ndarray, int, int, float]]:
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

    buffer = np.zeros((height_px, width_px), dtype=np.uint8)
    has_content = _rasterize_scanlines_shared(
        ops,
        buffer,
        width_px,
        height_px,
        origin_mm=(x0, y0),
        px_per_mm=px_per_mm,
    )
    if not has_content:
        return None

    dilated = np.zeros_like(buffer)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            shifted = np.roll(np.roll(buffer, dy, axis=0), dx, axis=1)
            np.maximum(dilated, shifted, out=dilated)
    buffer = dilated

    return buffer, width_px, height_px, px_per_mm


@dataclass
class _WalkState:
    current_power: float = 0.0
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_pos_vis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_initial: bool = True
    current_laser_uid: str = ""
    current_laser_index: int = 0
    laser_uid_order: List[str] = field(default_factory=list)
    is_rotary: bool = False
    rotary_diameter: float = 0.0
    has_mapped_data: bool = False
    layer_infos: List[dict] = field(default_factory=list)
    current_layer_start: Optional[int] = None
    current_layer_has_scanlines: bool = False
    current_layer_scanline_laser: str = ""


class _LayerAccumulator:
    """Accumulates vertex data for one rendering treatment group."""

    __slots__ = (
        "pv",
        "pvv",
        "pvl",
        "tv",
        "zpv",
        "ov_pos",
        "ov_pow",
        "ov_lid",
        "pv_cum",
        "tv_cum",
        "ov_cum",
        "pv_off",
        "tv_off",
        "ov_off",
        "is_rotary",
        "rotary_segments",
        "current_rotary_seg",
        "axis_position",
        "reverse",
        "diameter",
        "axis_position_3d",
        "cylinder_dir",
    )

    def __init__(self, total_cmds: int, is_rotary: bool):
        self.pv: List[float] = []
        self.pvv: List[float] = []
        self.pvl: List[int] = []
        self.tv: List[float] = []
        self.zpv: List[float] = []
        self.ov_pos: List[float] = []
        self.ov_pow: List[float] = []
        self.ov_lid: List[int] = []
        self.pv_cum = 0
        self.tv_cum = 0
        self.ov_cum = 0
        self.pv_off = [0] * (total_cmds + 1)
        self.tv_off = [0] * (total_cmds + 1)
        self.ov_off = [0] * (total_cmds + 1)
        self.is_rotary = is_rotary
        self.rotary_segments: List[dict] = []
        self.current_rotary_seg: Optional[dict] = None
        self.axis_position: float = 0.0
        self.reverse: bool = False
        self.diameter: float = 0.0
        self.axis_position_3d: Optional[Tuple[float, ...]] = None
        self.cylinder_dir: Optional[Tuple[float, ...]] = None

    def record_offset(self, cmd_idx: int):
        self.pv_off[cmd_idx + 1] = self.pv_cum
        self.tv_off[cmd_idx + 1] = self.tv_cum
        self.ov_off[cmd_idx + 1] = self.ov_cum

    def begin_rotary_segment(self, cmd_idx: int, degrees_input: bool = False):
        self.current_rotary_seg = {
            "pv_start": len(self.pv) // 3,
            "tv_start": len(self.tv) // 3,
            "zpv_vtx_start": len(self.zpv) // 3,
            "ov_start": len(self.ov_pos) // 3,
            "cmd_start": cmd_idx + 1,
            "diameter": 0.0,
            "degrees_input": degrees_input,
        }

    def end_rotary_segment(self, diameter: float, cmd_idx: int):
        seg = self.current_rotary_seg
        assert seg is not None
        seg["pv_end"] = len(self.pv) // 3
        seg["tv_end"] = len(self.tv) // 3
        seg["zpv_vtx_end"] = len(self.zpv) // 3
        seg["ov_end"] = len(self.ov_pos) // 3
        seg["cmd_end"] = cmd_idx
        seg["diameter"] = diameter
        self.rotary_segments.append(seg)
        self.current_rotary_seg = None

    def has_content(self) -> bool:
        return bool(self.pv or self.tv or self.zpv or self.ov_pos)

    def finalize(
        self,
        transform: np.ndarray,
    ):
        pv_arr = _to_arr(self.pv, 3)
        pvv_arr = _to_arr(self.pvv, 1)
        pvl_arr = np.array(self.pvl, dtype=np.float32).reshape(-1, 1)
        tv_arr = _to_arr(self.tv, 3)
        zpv_arr = _to_arr(self.zpv, 3)
        ov_pos_arr = _to_arr(self.ov_pos, 3)
        ov_pow_arr = _to_arr(self.ov_pow, 1)
        ov_lid_arr = np.array(self.ov_lid, dtype=np.float32).reshape(-1, 1)

        pv_arr = _transform_verts(pv_arr, transform)
        tv_arr = _transform_verts(tv_arr, transform)
        zpv_arr = _transform_verts(zpv_arr, transform)
        ov_pos_arr = _transform_verts(ov_pos_arr, transform)

        if self.is_rotary and self.rotary_segments:
            (
                pv_arr,
                pvv_arr,
                pvl_arr,
                tv_arr,
                zpv_arr,
                ov_pos_arr,
                ov_pow_arr,
                ov_lid_arr,
                pv_expansion,
            ) = _apply_cylinder_wrapping(
                pv_arr,
                pvv_arr,
                pvl_arr,
                tv_arr,
                zpv_arr,
                ov_pos_arr,
                ov_pow_arr,
                ov_lid_arr,
                self.rotary_segments,
                self.pv_cum,
                self.tv_cum,
                self.ov_cum,
            )
            if pv_expansion:
                self.pv_off = _remap_offsets(self.pv_off, pv_expansion)

        if tv_arr.size > 0:
            tv_arr[:, 2] += Z_OFFSET_NON_POWERED
        if zpv_arr.size > 0:
            zpv_arr[:, 2] += Z_OFFSET_NON_POWERED

        return (
            pv_arr,
            pvv_arr,
            pvl_arr,
            tv_arr,
            zpv_arr,
            ov_pos_arr,
            ov_pow_arr,
            ov_lid_arr,
        )


def _pad_powers_to_vec4(pvv: np.ndarray) -> np.ndarray:
    if pvv.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    flat = pvv.ravel()
    n = flat.shape[0]
    result = np.zeros((n, 4), dtype=np.float32)
    result[:, 0] = flat
    result[:, 3] = 1.0
    return result


def _extract_powers_from_vec4(pvv4: np.ndarray) -> np.ndarray:
    if pvv4.size == 0:
        return np.empty((0, 1), dtype=np.float32)
    return pvv4[:, 0:1].copy()


def _pad_power_laser_to_vec4(pvv: np.ndarray, pvl: np.ndarray) -> np.ndarray:
    if pvv.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    power = pvv.ravel()
    laser = pvl.ravel()
    n = power.shape[0]
    result = np.zeros((n, 4), dtype=np.float32)
    result[:, 0] = power
    result[:, 1] = laser
    result[:, 3] = 1.0
    return result


def _extract_laser_from_vec4(pvv4: np.ndarray) -> np.ndarray:
    if pvv4.size == 0:
        return np.empty((0, 1), dtype=np.float32)
    return pvv4[:, 1:2].copy()


def _apply_cylinder_wrapping(
    pv_arr,
    pvv_arr,
    pvl_arr,
    tv_arr,
    zpv_arr,
    ov_pos_arr,
    ov_pow_arr,
    ov_lid_arr,
    rotary_segments,
    pv_cum,
    tv_cum,
    ov_cum,
):
    _exp_pv: List[np.ndarray] = []
    _exp_pvv: List[np.ndarray] = []
    _exp_tv: List[np.ndarray] = []
    _exp_zpv: List[np.ndarray] = []
    _exp_ov_pos: List[np.ndarray] = []
    _exp_ov_pow: List[np.ndarray] = []
    pv_expansion: List[Tuple[int, np.ndarray]] = []

    pvv4 = _pad_power_laser_to_vec4(pvv_arr, pvl_arr)
    ov_pow4 = _pad_power_laser_to_vec4(ov_pow_arr, ov_lid_arr)

    for seg in rotary_segments:
        d = seg["diameter"]
        deg_in = seg.get("degrees_input", False)
        if d <= 0:
            continue

        pv_s = seg["pv_start"]
        pv_e = seg.get("pv_end", pv_cum)
        if pv_e > pv_s:
            pv_w, pvv4_w, cum_subs = _apply_cylinder(
                pv_arr[pv_s:pv_e],
                d,
                pvv4[pv_s:pv_e],
                degrees_input=deg_in,
            )
            assert pvv4_w is not None
            _exp_pv.append(pv_w)
            _exp_pvv.append(pvv4_w)
            pv_expansion.append((pv_s, cum_subs))

        tv_s = seg["tv_start"]
        tv_e = seg.get("tv_end", tv_cum)
        if tv_e > tv_s:
            tv_w, _, _ = _apply_cylinder(
                tv_arr[tv_s:tv_e],
                d,
                degrees_input=deg_in,
            )
            _exp_tv.append(tv_w)

        zpv_s = seg["zpv_vtx_start"]
        zpv_e = seg.get("zpv_vtx_end", len(zpv_arr))
        if zpv_e > zpv_s:
            zpv_w, _, _ = _apply_cylinder(
                zpv_arr[zpv_s:zpv_e],
                d,
                degrees_input=deg_in,
            )
            _exp_zpv.append(zpv_w)

        ov_s = seg["ov_start"]
        ov_e = seg.get("ov_end", ov_cum)
        if ov_e > ov_s:
            ov_pos_w, ov_pow4_w, _ = _apply_cylinder(
                ov_pos_arr[ov_s:ov_e],
                d,
                ov_pow4[ov_s:ov_e],
                degrees_input=deg_in,
            )
            assert ov_pow4_w is not None
            _exp_ov_pos.append(ov_pos_w)
            _exp_ov_pow.append(ov_pow4_w)

    if _exp_pv:
        pv_arr = np.concatenate(_exp_pv, axis=0)
        pvv4 = np.concatenate(_exp_pvv, axis=0)
    if _exp_tv:
        tv_arr = np.concatenate(_exp_tv, axis=0)
    if _exp_zpv:
        zpv_arr = np.concatenate(_exp_zpv, axis=0)
    if _exp_ov_pos:
        ov_pos_arr = np.concatenate(_exp_ov_pos, axis=0)
        ov_pow4 = np.concatenate(_exp_ov_pow, axis=0)

    return (
        pv_arr,
        _extract_powers_from_vec4(pvv4),
        _extract_laser_from_vec4(pvv4),
        tv_arr,
        zpv_arr,
        ov_pos_arr,
        _extract_powers_from_vec4(ov_pow4),
        _extract_laser_from_vec4(ov_pow4),
        pv_expansion,
    )


def _finalize_layers(
    accumulators: dict[bool, _LayerAccumulator],
    config: RenderConfig3D,
) -> Tuple[List[VertexLayer], List[ScanlineOverlayLayer]]:
    flat_transform = config.world_to_visual

    vertex_layers: List[VertexLayer] = []
    overlay_layers: List[ScanlineOverlayLayer] = []

    for acc in accumulators.values():
        if not acc.has_content():
            continue
        if acc.is_rotary:
            transform = np.eye(4, dtype=np.float32)
        else:
            transform = flat_transform
        (
            pv_arr,
            pvv_arr,
            pvl_arr,
            tv_arr,
            zpv_arr,
            ov_pos_arr,
            ov_pow_arr,
            ov_lid_arr,
        ) = acc.finalize(transform)

        vertex_layers.append(
            VertexLayer(
                powered_verts=_to_flat(pv_arr),
                power_values=_to_flat(pvv_arr),
                laser_indices=_to_flat(pvl_arr),
                travel_verts=_to_flat(tv_arr),
                zero_power_verts=_to_flat(zpv_arr),
                powered_cmd_offsets=acc.pv_off,
                travel_cmd_offsets=acc.tv_off,
                is_rotary=acc.is_rotary,
            )
        )

        overlay_layers.append(
            ScanlineOverlayLayer(
                positions=_to_flat(ov_pos_arr),
                power_values=_to_flat(ov_pow_arr),
                laser_indices=_to_flat(ov_lid_arr),
                cmd_offsets=acc.ov_off,
                is_rotary=acc.is_rotary,
            )
        )

    return vertex_layers, overlay_layers


def _generate_texture_layers(
    ops: Ops,
    layer_infos: List[dict],
    config: RenderConfig3D,
) -> List[TextureLayer]:
    texture_layers: List[TextureLayer] = []

    for li in layer_infos:
        if not li["has_scanlines"]:
            continue

        layer_ops = Ops()
        layer_ops.replace_all(
            list(ops.commands[li["cmd_start"] : li["cmd_end"]])
        )

        is_rot = li["is_rotary"]

        if is_rot:
            layer_ops = _bake_visual_positions(layer_ops)

        bbox = _scanline_bbox(layer_ops)
        if bbox is None:
            continue

        raster_result = _rasterize_scanlines(layer_ops, bbox)
        if raster_result is None:
            continue

        tex_buf, w_px, h_px, actual_ppm = raster_result
        x0, y0, bw, bh = bbox

        diameter = li["diameter"]

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
                activation_cmd_idx=li["activation_cmd_idx"],
                laser_uid=li.get("scanline_laser", ""),
            )
        )

    return texture_layers


def _handle_layer_start(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd_idx: int,
    cmd: LayerStartCommand,
    config: RenderConfig3D,
    accumulators: dict[bool, _LayerAccumulator],
) -> _LayerAccumulator:
    layer_uid = cmd.layer_uid
    layer_cfg = None
    if config.layer_configs and layer_uid in config.layer_configs:
        layer_cfg = config.layer_configs[layer_uid]
    st.is_rotary = layer_cfg.rotary_enabled if layer_cfg else False
    st.rotary_diameter = layer_cfg.rotary_diameter if layer_cfg else 0.0
    acc = accumulators[st.is_rotary]

    acc.axis_position = layer_cfg.axis_position if layer_cfg else 0.0
    acc.reverse = layer_cfg.reverse if layer_cfg else False
    acc.diameter = st.rotary_diameter
    acc.axis_position_3d = layer_cfg.axis_position_3d if layer_cfg else None
    acc.cylinder_dir = layer_cfg.cylinder_dir if layer_cfg else None
    st.has_mapped_data = st.is_rotary

    if st.is_rotary and st.rotary_diameter > 0:
        acc.begin_rotary_segment(cmd_idx, degrees_input=st.has_mapped_data)

    st.current_layer_start = cmd_idx + 1
    st.current_layer_has_scanlines = False
    st.current_layer_scanline_laser = ""

    return acc


def _handle_layer_end(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd_idx: int,
) -> None:
    if st.current_layer_start is not None:
        st.layer_infos.append(
            {
                "cmd_start": st.current_layer_start,
                "cmd_end": cmd_idx,
                "is_rotary": st.is_rotary,
                "diameter": st.rotary_diameter,
                "has_scanlines": st.current_layer_has_scanlines,
                "scanline_laser": st.current_layer_scanline_laser,
                "activation_cmd_idx": cmd_idx,
                "axis_position": acc.axis_position,
                "reverse": acc.reverse,
            }
        )
    if acc.current_rotary_seg is not None:
        acc.end_rotary_segment(st.rotary_diameter, cmd_idx)
    st.current_layer_start = None


def _handle_set_laser(
    st: _WalkState,
    cmd: SetLaserCommand,
) -> None:
    st.current_laser_uid = cmd.laser_uid
    if cmd.laser_uid not in st.laser_uid_order:
        st.laser_uid_order.append(cmd.laser_uid)
    st.current_laser_index = st.laser_uid_order.index(cmd.laser_uid)


def _update_positions(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd: MovingCommand,
) -> None:
    st.current_pos_vis = _visual_end(cmd)
    if st.has_mapped_data and st.is_rotary:
        st.current_pos = _reconstruct_mu_pos(
            cmd,
            acc.diameter,
            acc.reverse,
        )
    else:
        st.current_pos = cmd.end
    st.is_initial = False


def _handle_move_to(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd: MoveToCommand,
) -> None:
    vis_end = _visual_end(cmd)
    if not st.is_initial:
        acc.tv.extend(st.current_pos_vis)
        acc.tv.extend(vis_end)
        acc.tv_cum += 2
    _update_positions(st, acc, cmd)


def _handle_line_to(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd: LineToCommand,
) -> None:
    vis_end = _visual_end(cmd)
    if st.current_power > 0.0:
        acc.pv.extend(st.current_pos_vis)
        acc.pv.extend(vis_end)
        acc.pvv.append(st.current_power)
        acc.pvv.append(st.current_power)
        acc.pvl.append(st.current_laser_index)
        acc.pvl.append(st.current_laser_index)
        acc.pv_cum += 2
    else:
        acc.zpv.extend(st.current_pos_vis)
        acc.zpv.extend(vis_end)
    _update_positions(st, acc, cmd)


def _handle_arc_to(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd: ArcToCommand,
) -> None:
    if st.has_mapped_data and st.is_rotary:
        mu_cmd = _reconstruct_mu_arc(
            cmd,
            acc.diameter,
            acc.reverse,
        )
        segments = linearize_arc(mu_cmd, st.current_pos)
        vis_segs = []
        for seg_start, seg_end in segments:
            vis_start = _mu_to_visual(
                seg_start,
                acc.diameter,
                acc.reverse,
            )
            vis_end_pt = _mu_to_visual(
                seg_end,
                acc.diameter,
                acc.reverse,
            )
            vis_segs.append((vis_start, vis_end_pt))
    else:
        segments = linearize_arc(cmd, st.current_pos)
        vis_segs = segments
    if st.current_power > 0.0:
        n_segs = len(vis_segs)
        for seg_start, seg_end in vis_segs:
            acc.pv.extend(seg_start)
            acc.pv.extend(seg_end)
            acc.pvv.append(st.current_power)
            acc.pvv.append(st.current_power)
            acc.pvl.append(st.current_laser_index)
            acc.pvl.append(st.current_laser_index)
        acc.pv_cum += n_segs * 2
    else:
        for seg_start, seg_end in vis_segs:
            acc.zpv.extend(seg_start)
            acc.zpv.extend(seg_end)
    _update_positions(st, acc, cmd)


def _handle_bezier_to(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd: BezierToCommand,
) -> None:
    if st.has_mapped_data and st.is_rotary:
        mu_cmd = _reconstruct_mu_bezier(
            cmd,
            acc.diameter,
            acc.reverse,
        )
        polyline = linearize_bezier_segment(
            st.current_pos,
            mu_cmd.control1,
            mu_cmd.control2,
            mu_cmd.end,
        )
        vis_poly = [
            _mu_to_visual(
                pt,
                acc.diameter,
                acc.reverse,
            )
            for pt in polyline
        ]
    else:
        polyline = linearize_bezier_segment(
            st.current_pos,
            cmd.control1,
            cmd.control2,
            cmd.end,
        )
        vis_poly = list(polyline)
    if st.current_power > 0.0:
        for j in range(len(vis_poly) - 1):
            acc.pv.extend(vis_poly[j])
            acc.pv.extend(vis_poly[j + 1])
            acc.pvv.append(st.current_power)
            acc.pvv.append(st.current_power)
            acc.pvl.append(st.current_laser_index)
            acc.pvl.append(st.current_laser_index)
        acc.pv_cum += (len(vis_poly) - 1) * 2
    else:
        for j in range(len(vis_poly) - 1):
            acc.zpv.extend(vis_poly[j])
            acc.zpv.extend(vis_poly[j + 1])
    _update_positions(st, acc, cmd)


def _handle_scanline(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd: ScanLinePowerCommand,
) -> None:
    if cmd.end is not None:
        vis_end = _visual_end(cmd)
        _extract_zero_power_segments(
            cmd,
            st.current_pos_vis,
            acc.zpv,
            end_pos=vis_end,
        )
        if not st.is_initial:
            n = _encode_overlay_segments(
                cmd,
                st.current_pos_vis,
                acc.ov_pos,
                acc.ov_pow,
                acc.ov_lid,
                laser_index=st.current_laser_index,
                end_pos=vis_end,
            )
            acc.ov_cum += n
        _update_positions(st, acc, cmd)
    st.current_layer_has_scanlines = True
    if not st.current_layer_scanline_laser:
        st.current_layer_scanline_laser = st.current_laser_uid


def compile_scene(
    ops: Ops,
    config: RenderConfig3D,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> CompiledSceneArtifact:
    total_cmds = len(ops.commands)

    accumulators: dict[bool, _LayerAccumulator] = {
        False: _LayerAccumulator(total_cmds, is_rotary=False),
        True: _LayerAccumulator(total_cmds, is_rotary=True),
    }

    st = _WalkState()

    for i, cmd in enumerate(ops.commands):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")

        acc = accumulators[st.is_rotary]

        if isinstance(cmd, LayerStartCommand):
            acc = _handle_layer_start(
                st,
                acc,
                i,
                cmd,
                config,
                accumulators,
            )
        elif isinstance(cmd, LayerEndCommand):
            _handle_layer_end(st, acc, i)
        elif isinstance(cmd, SetLaserCommand):
            _handle_set_laser(st, cmd)
        elif isinstance(cmd, SetPowerCommand):
            st.current_power = cmd.power
        elif isinstance(cmd, MoveToCommand):
            _handle_move_to(st, acc, cmd)
        elif isinstance(cmd, LineToCommand):
            _handle_line_to(st, acc, cmd)
        elif isinstance(cmd, ArcToCommand):
            _handle_arc_to(st, acc, cmd)
        elif isinstance(cmd, BezierToCommand):
            _handle_bezier_to(st, acc, cmd)
        elif isinstance(cmd, ScanLinePowerCommand):
            _handle_scanline(st, acc, cmd)

        for a in accumulators.values():
            a.record_offset(i)

    vertex_layers, overlay_layers = _finalize_layers(accumulators, config)

    texture_layers = _generate_texture_layers(ops, st.layer_infos, config)

    return CompiledSceneArtifact(
        generation_id=0,
        vertex_layers=vertex_layers,
        texture_layers=texture_layers,
        overlay_layers=overlay_layers,
        laser_uid_order=st.laser_uid_order,
    )
