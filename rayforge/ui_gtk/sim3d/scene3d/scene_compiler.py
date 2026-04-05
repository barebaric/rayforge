"""
Scene compiler: pure function that compiles assembled job Ops into
GPU-ready vertex data.

Walks the full assembled job ops (with JobStart/End, LayerStart/End
markers) to produce vertex buffers and per-command offset arrays that
are indexed 1:1 with the player's command index.
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Tuple

import numpy as np

from ....core.geo.linearize import linearize_arc
from ....core.ops import Ops
from ....core.ops.commands import (
    ArcToCommand,
    LayerEndCommand,
    LayerStartCommand,
    LineToCommand,
    MoveToCommand,
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

logger = logging.getLogger(__name__)

Z_OFFSET_NON_POWERED = 0.01


def _get_lut(
    config: RenderConfig3D,
    laser_uid: str,
    lut_type: str,
) -> Optional[np.ndarray]:
    laser_luts = config.laser_color_luts.get(laser_uid, {})
    lut_bytes = laser_luts.get(lut_type)
    if lut_bytes is None:
        if lut_type == "cut":
            lut_bytes = config.default_color_lut_cut
        elif lut_type == "engrave":
            lut_bytes = config.default_color_lut_engrave
    if lut_bytes is None:
        return None
    return np.frombuffer(lut_bytes, dtype=np.float32).reshape(256, 4).copy()


def _transform_verts(verts: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if verts.size == 0:
        return verts
    pts = verts.reshape(-1, 3)
    hom = np.hstack([pts, np.ones((pts.shape[0], 1), dtype=pts.dtype)])
    transformed = (transform @ hom.T).T
    return transformed[:, :3].astype(np.float32)


def _apply_cylinder(
    verts: np.ndarray,
    diameter: float,
    colors: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if verts.size == 0 or diameter <= 0:
        return verts, colors
    return transform_to_cylinder(verts.reshape(-1, 3), diameter, colors)


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
) -> None:
    if cmd.end is None:
        return
    num_steps = len(cmd.power_values)
    if num_steps == 0:
        return

    p_start = np.array(start_pos, dtype=np.float32)
    p_end = np.array(cmd.end, dtype=np.float32)
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
    engrave_lut: Optional[np.ndarray],
    ov_pos: List[float],
    ov_col: List[float],
) -> int:
    if cmd.end is None:
        return 0
    num_steps = len(cmd.power_values)
    if num_steps == 0:
        return 0

    sx, sy, sz = start_pos
    ex, ey, ez = cmd.end
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
            if engrave_lut is not None:
                idx = min(255, int(seg_power * 255))
                c = engrave_lut[idx]
                ov_col.extend(c)
                ov_col.extend(c)
            else:
                v = seg_power
                ov_col.extend([v, v, v, 1.0])
                ov_col.extend([v, v, v, 1.0])
            vertex_count += 2

        prev_power_on = power_on

    if prev_power_on:
        ov_pos.extend([seg_start_x, seg_start_y, seg_start_z, ex, ey, ez])
        if engrave_lut is not None:
            idx = min(255, int(seg_power * 255))
            c = engrave_lut[idx]
            ov_col.extend(c)
            ov_col.extend(c)
        else:
            v = seg_power
            ov_col.extend([v, v, v, 1.0])
            ov_col.extend([v, v, v, 1.0])
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

    return buffer, width_px, height_px, px_per_mm


class _LayerAccumulator:
    """Accumulates vertex data for one rendering treatment group."""

    __slots__ = (
        "pv",
        "pc",
        "tv",
        "zpv",
        "ov_pos",
        "ov_col",
        "pv_cum",
        "tv_cum",
        "ov_cum",
        "pv_off",
        "tv_off",
        "ov_off",
        "is_rotary",
        "rotary_segments",
        "current_rotary_seg",
    )

    def __init__(self, total_cmds: int, is_rotary: bool):
        self.pv: List[float] = []
        self.pc: List[float] = []
        self.tv: List[float] = []
        self.zpv: List[float] = []
        self.ov_pos: List[float] = []
        self.ov_col: List[float] = []
        self.pv_cum = 0
        self.tv_cum = 0
        self.ov_cum = 0
        self.pv_off = [0] * (total_cmds + 1)
        self.tv_off = [0] * (total_cmds + 1)
        self.ov_off = [0] * (total_cmds + 1)
        self.is_rotary = is_rotary
        self.rotary_segments: List[dict] = []
        self.current_rotary_seg: Optional[dict] = None

    def record_offset(self, cmd_idx: int):
        self.pv_off[cmd_idx + 1] = self.pv_cum
        self.tv_off[cmd_idx + 1] = self.tv_cum
        self.ov_off[cmd_idx + 1] = self.ov_cum

    def begin_rotary_segment(self, cmd_idx: int):
        self.current_rotary_seg = {
            "pv_start": len(self.pv) // 3,
            "tv_start": len(self.tv) // 3,
            "zpv_vtx_start": len(self.zpv) // 3,
            "ov_start": len(self.ov_pos) // 3,
            "cmd_start": cmd_idx + 1,
            "diameter": 0.0,
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
        zero_rgba: np.ndarray,
        transform: np.ndarray,
    ):
        pv_arr = _to_arr(self.pv, 3)
        pc_arr = _to_arr(self.pc, 4)
        tv_arr = _to_arr(self.tv, 3)
        zpv_arr = _to_arr(self.zpv, 3)
        zpc_count = len(self.zpv) // 3
        zpc_arr = (
            np.tile(zero_rgba, (zpc_count, 1))
            if zpc_count > 0
            else np.empty((0, 4), dtype=np.float32)
        )
        ov_pos_arr = _to_arr(self.ov_pos, 3)
        ov_col_arr = _to_arr(self.ov_col, 4)

        pv_arr = _transform_verts(pv_arr, transform)
        tv_arr = _transform_verts(tv_arr, transform)
        zpv_arr = _transform_verts(zpv_arr, transform)
        ov_pos_arr = _transform_verts(ov_pos_arr, transform)

        if self.is_rotary and self.rotary_segments:
            (
                pv_arr,
                pc_arr,
                tv_arr,
                zpv_arr,
                zpc_arr,
                ov_pos_arr,
                ov_col_arr,
            ) = _apply_cylinder_wrapping(
                pv_arr,
                pc_arr,
                tv_arr,
                zpv_arr,
                zpc_arr,
                ov_pos_arr,
                ov_col_arr,
                self.rotary_segments,
                self.pv_cum,
                self.tv_cum,
                self.ov_cum,
            )

        if tv_arr.size > 0:
            tv_arr[:, 2] += Z_OFFSET_NON_POWERED
        if zpv_arr.size > 0:
            zpv_arr[:, 2] += Z_OFFSET_NON_POWERED

        return (
            pv_arr,
            pc_arr,
            tv_arr,
            zpv_arr,
            zpc_arr,
            ov_pos_arr,
            ov_col_arr,
        )


def _apply_cylinder_wrapping(
    pv_arr,
    pc_arr,
    tv_arr,
    zpv_arr,
    zpc_arr,
    ov_pos_arr,
    ov_col_arr,
    rotary_segments,
    pv_cum,
    tv_cum,
    ov_cum,
):
    _exp_pv: List[np.ndarray] = []
    _exp_pc: List[np.ndarray] = []
    _exp_tv: List[np.ndarray] = []
    _exp_zpv: List[np.ndarray] = []
    _exp_zpc: List[np.ndarray] = []
    _exp_ov_pos: List[np.ndarray] = []
    _exp_ov_col: List[np.ndarray] = []

    for seg in rotary_segments:
        d = seg["diameter"]
        if d <= 0:
            continue

        pv_s = seg["pv_start"]
        pv_e = seg.get("pv_end", pv_cum)
        if pv_e > pv_s:
            pv_w, pc_w = _apply_cylinder(
                pv_arr[pv_s:pv_e], d, pc_arr[pv_s:pv_e]
            )
            assert pc_w is not None
            _exp_pv.append(pv_w)
            _exp_pc.append(pc_w)

        tv_s = seg["tv_start"]
        tv_e = seg.get("tv_end", tv_cum)
        if tv_e > tv_s:
            tv_w, _ = _apply_cylinder(tv_arr[tv_s:tv_e], d)
            _exp_tv.append(tv_w)

        zpv_s = seg["zpv_vtx_start"]
        zpv_e = seg.get("zpv_vtx_end", len(zpv_arr))
        if zpv_e > zpv_s:
            zpv_w, zpc_w = _apply_cylinder(
                zpv_arr[zpv_s:zpv_e], d, zpc_arr[zpv_s:zpv_e]
            )
            assert zpc_w is not None
            _exp_zpv.append(zpv_w)
            _exp_zpc.append(zpc_w)

        ov_s = seg["ov_start"]
        ov_e = seg.get("ov_end", ov_cum)
        if ov_e > ov_s:
            ov_pos_w, ov_col_w = _apply_cylinder(
                ov_pos_arr[ov_s:ov_e], d, ov_col_arr[ov_s:ov_e]
            )
            assert ov_col_w is not None
            _exp_ov_pos.append(ov_pos_w)
            _exp_ov_col.append(ov_col_w)

    if _exp_pv:
        pv_arr = np.concatenate(_exp_pv, axis=0)
        pc_arr = np.concatenate(_exp_pc, axis=0)
    if _exp_tv:
        tv_arr = np.concatenate(_exp_tv, axis=0)
    if _exp_zpv:
        zpv_arr = np.concatenate(_exp_zpv, axis=0)
        zpc_arr = np.concatenate(_exp_zpc, axis=0)
    if _exp_ov_pos:
        ov_pos_arr = np.concatenate(_exp_ov_pos, axis=0)
        ov_col_arr = np.concatenate(_exp_ov_col, axis=0)

    return (
        pv_arr,
        pc_arr,
        tv_arr,
        zpv_arr,
        zpc_arr,
        ov_pos_arr,
        ov_col_arr,
    )


def compile_scene(
    ops: Ops,
    config: RenderConfig3D,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> CompiledSceneArtifact:
    total_cmds = len(ops.commands)
    zero_rgba = np.array(config.zero_power_rgba, dtype=np.float32)

    accumulators: dict[bool, _LayerAccumulator] = {
        False: _LayerAccumulator(total_cmds, is_rotary=False),
        True: _LayerAccumulator(total_cmds, is_rotary=True),
    }

    texture_layers: List[TextureLayer] = []

    current_power = 0.0
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_initial = True
    current_laser_uid = ""
    is_rotary = False
    rotary_diameter = 0.0

    layer_infos: List[dict] = []
    current_layer_start = None
    current_layer_has_scanlines = False
    current_layer_scanline_laser = ""
    _default_cut = _get_lut(config, "", "cut")
    if _default_cut is None:
        _default_cut = np.zeros((256, 4), dtype=np.float32)
    current_cut_lut: np.ndarray = _default_cut

    _default_engrave = _get_lut(config, "", "engrave")
    if _default_engrave is None:
        _default_engrave = np.zeros((256, 4), dtype=np.float32)
    current_engrave_lut: np.ndarray = _default_engrave

    for i, cmd in enumerate(ops.commands):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")

        acc = accumulators[is_rotary]

        if isinstance(cmd, LayerStartCommand):
            layer_uid = cmd.layer_uid
            layer_cfg = None
            if config.layer_configs and layer_uid in config.layer_configs:
                layer_cfg = config.layer_configs[layer_uid]
            is_rotary = layer_cfg.rotary_enabled if layer_cfg else False
            rotary_diameter = layer_cfg.rotary_diameter if layer_cfg else 0.0
            acc = accumulators[is_rotary]

            if is_rotary and rotary_diameter > 0:
                acc.begin_rotary_segment(i)

            current_layer_start = i + 1
            current_layer_has_scanlines = False
            current_layer_scanline_laser = ""

        elif isinstance(cmd, LayerEndCommand):
            if current_layer_start is not None:
                layer_infos.append(
                    {
                        "cmd_start": current_layer_start,
                        "cmd_end": i,
                        "is_rotary": is_rotary,
                        "diameter": rotary_diameter,
                        "has_scanlines": current_layer_has_scanlines,
                        "scanline_laser": current_layer_scanline_laser,
                        "activation_cmd_idx": (current_layer_start - 1),
                    }
                )
            if acc.current_rotary_seg is not None:
                acc.end_rotary_segment(rotary_diameter, i)
            current_layer_start = None

        elif isinstance(cmd, SetLaserCommand):
            current_laser_uid = cmd.laser_uid
            _cut = _get_lut(config, current_laser_uid, "cut")
            current_cut_lut = (
                _cut
                if _cut is not None
                else np.zeros((256, 4), dtype=np.float32)
            )
            _engrave = _get_lut(config, current_laser_uid, "engrave")
            current_engrave_lut = (
                _engrave
                if _engrave is not None
                else np.zeros((256, 4), dtype=np.float32)
            )

        elif isinstance(cmd, SetPowerCommand):
            current_power = cmd.power

        elif isinstance(cmd, MoveToCommand):
            if not is_initial:
                acc.tv.extend(current_pos)
                acc.tv.extend(cmd.end)
                acc.tv_cum += 2
            current_pos = cmd.end
            is_initial = False

        elif isinstance(cmd, LineToCommand):
            if current_power > 0.0:
                power_byte = min(255, int(current_power * 255.0))
                color = current_cut_lut[power_byte]
                acc.pv.extend(current_pos)
                acc.pv.extend(cmd.end)
                acc.pc.extend(color)
                acc.pc.extend(color)
                acc.pv_cum += 2
            else:
                acc.zpv.extend(current_pos)
                acc.zpv.extend(cmd.end)
            current_pos = cmd.end
            is_initial = False

        elif isinstance(cmd, ArcToCommand):
            segments = linearize_arc(cmd, current_pos)
            if current_power > 0.0:
                power_byte = min(255, int(current_power * 255.0))
                color = current_cut_lut[power_byte]
                n_segs = len(segments)
                for seg_start, seg_end in segments:
                    acc.pv.extend(seg_start)
                    acc.pv.extend(seg_end)
                    acc.pc.extend(color)
                    acc.pc.extend(color)
                acc.pv_cum += n_segs * 2
            else:
                for seg_start, seg_end in segments:
                    acc.zpv.extend(seg_start)
                    acc.zpv.extend(seg_end)
            current_pos = cmd.end
            is_initial = False

        elif isinstance(cmd, ScanLinePowerCommand):
            if cmd.end is not None:
                _extract_zero_power_segments(cmd, current_pos, acc.zpv)
                if not is_initial:
                    n = _encode_overlay_segments(
                        cmd,
                        current_pos,
                        current_engrave_lut,
                        acc.ov_pos,
                        acc.ov_col,
                    )
                    acc.ov_cum += n
                current_pos = cmd.end
                is_initial = False
            current_layer_has_scanlines = True
            if not current_layer_scanline_laser:
                current_layer_scanline_laser = current_laser_uid

        for a in accumulators.values():
            a.record_offset(i)

    # Finalize each accumulator that has content
    vertex_layers: List[VertexLayer] = []
    overlay_layers: List[ScanlineOverlayLayer] = []

    flat_transform = config.world_to_visual
    rot_transform = config.world_to_cyl_local

    for acc in accumulators.values():
        if not acc.has_content():
            continue
        transform = rot_transform if acc.is_rotary else flat_transform
        (
            pv_arr,
            pc_arr,
            tv_arr,
            zpv_arr,
            zpc_arr,
            ov_pos_arr,
            ov_col_arr,
        ) = acc.finalize(zero_rgba, transform)

        vertex_layers.append(
            VertexLayer(
                powered_verts=_to_flat(pv_arr),
                powered_colors=_to_flat(pc_arr),
                travel_verts=_to_flat(tv_arr),
                zero_power_verts=_to_flat(zpv_arr),
                zero_power_colors=_to_flat(zpc_arr),
                powered_cmd_offsets=acc.pv_off,
                travel_cmd_offsets=acc.tv_off,
                is_rotary=acc.is_rotary,
            )
        )

        overlay_layers.append(
            ScanlineOverlayLayer(
                positions=_to_flat(ov_pos_arr),
                colors=_to_flat(ov_col_arr),
                cmd_offsets=acc.ov_off,
                is_rotary=acc.is_rotary,
            )
        )

    # Texture generation: rasterize scanlines per layer
    for li in layer_infos:
        if not li["has_scanlines"]:
            continue

        layer_ops = Ops()
        layer_ops.commands = list(
            ops.commands[li["cmd_start"] : li["cmd_end"]]
        )

        bbox = _scanline_bbox(layer_ops)
        if bbox is None:
            continue

        raster_result = _rasterize_scanlines(layer_ops, bbox)
        if raster_result is None:
            continue

        tex_buf, w_px, h_px, actual_ppm = raster_result
        x0, y0, bw, bh = bbox

        is_rot = li["is_rotary"]
        diameter = li["diameter"]
        transform = (
            config.world_to_cyl_local if is_rot else config.world_to_visual
        )

        model = np.eye(4, dtype=np.float32)
        model[0, 0] = bw
        model[1, 1] = bh
        model[0, 3] = x0
        model[1, 3] = y0
        final_model = (transform @ model).astype(np.float32)

        scan_lut = _get_lut(config, li["scanline_laser"], "engrave")

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
                color_lut=(scan_lut.copy() if scan_lut is not None else None),
                cylinder_vertices=cyl_verts,
                rotary_diameter=diameter,
                rotary_enabled=is_rot,
                activation_cmd_idx=li["activation_cmd_idx"],
            )
        )

    return CompiledSceneArtifact(
        generation_id=0,
        vertex_layers=vertex_layers,
        texture_layers=texture_layers,
        overlay_layers=overlay_layers,
    )
