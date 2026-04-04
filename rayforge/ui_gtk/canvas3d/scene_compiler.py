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

from ...core.geo.linearize import linearize_arc
from ...core.ops import Ops
from ...core.ops.commands import (
    ArcToCommand,
    LayerEndCommand,
    LayerStartCommand,
    LineToCommand,
    MoveToCommand,
    ScanLinePowerCommand,
    SetLaserCommand,
    SetPowerCommand,
)
from ...pipeline.encoder.vertexencoder import transform_to_cylinder
from .compiled_scene import (
    CompiledSceneArtifact,
    ScanlineOverlayLayer,
    TextureLayer,
    VertexLayer,
)
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


MAX_TEXTURE_DIMENSION = 8192
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


def _bresenham_line(
    buffer: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    power: int,
):
    h, w = buffer.shape
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        if 0 <= x < w and 0 <= y < h:
            buffer[y, x] = max(buffer[y, x], power)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


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
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    for cmd in ops.commands:
        if isinstance(cmd, MoveToCommand):
            if cmd.end is not None:
                current_pos = cmd.end
        elif isinstance(cmd, ScanLinePowerCommand):
            if cmd.end is None:
                continue
            end_mm = cmd.end
            num_steps = len(cmd.power_values)
            if num_steps == 0:
                current_pos = end_mm
                continue

            sx = (current_pos[0] - x0) * px_per_mm
            sy = height_px - (current_pos[1] - y0) * px_per_mm
            ex = (end_mm[0] - x0) * px_per_mm
            ey = height_px - (end_mm[1] - y0) * px_per_mm

            power_array = np.frombuffer(cmd.power_values, dtype=np.uint8)

            for i in range(num_steps):
                t_start = i / num_steps
                t_end = (i + 1) / num_steps
                psx = int(round(sx + t_start * (ex - sx)))
                psy = int(round(sy + t_start * (ey - sy)))
                pex = int(round(sx + t_end * (ex - sx)))
                pey = int(round(sy + t_end * (ey - sy)))

                power = power_array[i]
                if power == 0:
                    continue
                _bresenham_line(buffer, psx, psy, pex, pey, int(power))

            current_pos = end_mm

    if buffer.max() == 0:
        return None

    return buffer, width_px, height_px, px_per_mm


def compile_scene(
    ops: Ops,
    config: RenderConfig3D,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> CompiledSceneArtifact:
    pv_f: List[float] = []
    pc_f: List[float] = []
    tv_f: List[float] = []
    zpv_f: List[float] = []
    ov_pos_f: List[float] = []
    ov_col_f: List[float] = []

    pv_r: List[float] = []
    pc_r: List[float] = []
    tv_r: List[float] = []
    zpv_r: List[float] = []
    ov_pos_r: List[float] = []
    ov_col_r: List[float] = []

    texture_layers: List[TextureLayer] = []
    zero_rgba = np.array(config.zero_power_rgba, dtype=np.float32)

    total_cmds = len(ops.commands)
    flat_pv_off = [0] * (total_cmds + 1)
    flat_tv_off = [0] * (total_cmds + 1)
    flat_ov_off = [0] * (total_cmds + 1)
    rot_pv_off = [0] * (total_cmds + 1)
    rot_tv_off = [0] * (total_cmds + 1)
    rot_ov_off = [0] * (total_cmds + 1)

    flat_pv_cum = 0
    flat_tv_cum = 0
    flat_ov_cum = 0
    rot_pv_cum = 0
    rot_tv_cum = 0
    rot_ov_cum = 0

    current_power = 0.0
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_initial = True
    current_laser_uid = ""
    is_rotary = False
    rotary_diameter = 0.0

    rotary_segments: List[dict] = []
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
    current_rotary_seg: Optional[dict] = None

    for i, cmd in enumerate(ops.commands):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")

        if isinstance(cmd, LayerStartCommand):
            layer_uid = cmd.layer_uid
            layer_cfg = None
            if config.layer_configs and layer_uid in config.layer_configs:
                layer_cfg = config.layer_configs[layer_uid]
            is_rotary = layer_cfg.rotary_enabled if layer_cfg else False
            rotary_diameter = layer_cfg.rotary_diameter if layer_cfg else 0.0

            if is_rotary and rotary_diameter > 0:
                current_rotary_seg = {
                    "pv_start": len(pv_r) // 3,
                    "tv_start": len(tv_r) // 3,
                    "zpv_vtx_start": len(zpv_r) // 3,
                    "ov_start": len(ov_pos_r) // 3,
                    "diameter": rotary_diameter,
                    "cmd_start": i + 1,
                }

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
            if current_rotary_seg is not None:
                current_rotary_seg["pv_end"] = len(pv_r) // 3
                current_rotary_seg["tv_end"] = len(tv_r) // 3
                current_rotary_seg["zpv_vtx_end"] = len(zpv_r) // 3
                current_rotary_seg["ov_end"] = len(ov_pos_r) // 3
                current_rotary_seg["cmd_end"] = i
                rotary_segments.append(current_rotary_seg)
                current_rotary_seg = None
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
                if is_rotary:
                    tv_r.extend(current_pos)
                    tv_r.extend(cmd.end)
                    rot_tv_cum += 2
                else:
                    tv_f.extend(current_pos)
                    tv_f.extend(cmd.end)
                    flat_tv_cum += 2
            current_pos = cmd.end
            is_initial = False

        elif isinstance(cmd, LineToCommand):
            if current_power > 0.0:
                power_byte = min(255, int(current_power * 255.0))
                color = current_cut_lut[power_byte]
                if is_rotary:
                    pv_r.extend(current_pos)
                    pv_r.extend(cmd.end)
                    pc_r.extend(color)
                    pc_r.extend(color)
                    rot_pv_cum += 2
                else:
                    pv_f.extend(current_pos)
                    pv_f.extend(cmd.end)
                    pc_f.extend(color)
                    pc_f.extend(color)
                    flat_pv_cum += 2
            else:
                if is_rotary:
                    zpv_r.extend(current_pos)
                    zpv_r.extend(cmd.end)
                else:
                    zpv_f.extend(current_pos)
                    zpv_f.extend(cmd.end)
            current_pos = cmd.end
            is_initial = False

        elif isinstance(cmd, ArcToCommand):
            segments = linearize_arc(cmd, current_pos)
            if current_power > 0.0:
                power_byte = min(255, int(current_power * 255.0))
                color = current_cut_lut[power_byte]
                n_segs = len(segments)
                for seg_start, seg_end in segments:
                    if is_rotary:
                        pv_r.extend(seg_start)
                        pv_r.extend(seg_end)
                        pc_r.extend(color)
                        pc_r.extend(color)
                    else:
                        pv_f.extend(seg_start)
                        pv_f.extend(seg_end)
                        pc_f.extend(color)
                        pc_f.extend(color)
                if is_rotary:
                    rot_pv_cum += n_segs * 2
                else:
                    flat_pv_cum += n_segs * 2
            else:
                for seg_start, seg_end in segments:
                    if is_rotary:
                        zpv_r.extend(seg_start)
                        zpv_r.extend(seg_end)
                    else:
                        zpv_f.extend(seg_start)
                        zpv_f.extend(seg_end)
            current_pos = cmd.end
            is_initial = False

        elif isinstance(cmd, ScanLinePowerCommand):
            if cmd.end is not None:
                _zpv = zpv_r if is_rotary else zpv_f
                _extract_zero_power_segments(cmd, current_pos, _zpv)
                if not is_initial:
                    _ov_pos = ov_pos_r if is_rotary else ov_pos_f
                    _ov_col = ov_col_r if is_rotary else ov_col_f
                    n = _encode_overlay_segments(
                        cmd,
                        current_pos,
                        current_engrave_lut,
                        _ov_pos,
                        _ov_col,
                    )
                    if is_rotary:
                        rot_ov_cum += n
                    else:
                        flat_ov_cum += n
                current_pos = cmd.end
                is_initial = False
            current_layer_has_scanlines = True
            if not current_layer_scanline_laser:
                current_layer_scanline_laser = current_laser_uid

        flat_pv_off[i + 1] = flat_pv_cum
        flat_tv_off[i + 1] = flat_tv_cum
        flat_ov_off[i + 1] = flat_ov_cum
        rot_pv_off[i + 1] = rot_pv_cum
        rot_tv_off[i + 1] = rot_tv_cum
        rot_ov_off[i + 1] = rot_ov_cum

    # Convert raw lists to numpy arrays
    pv_f_arr = _to_arr(pv_f, 3)
    pc_f_arr = _to_arr(pc_f, 4)
    tv_f_arr = _to_arr(tv_f, 3)
    zpv_f_arr = _to_arr(zpv_f, 3)
    zpc_f_count = len(zpv_f) // 3
    zpc_f_arr = (
        np.tile(zero_rgba, (zpc_f_count, 1))
        if zpc_f_count > 0
        else np.empty((0, 4), dtype=np.float32)
    )
    ov_pos_f_arr = _to_arr(ov_pos_f, 3)
    ov_col_f_arr = _to_arr(ov_col_f, 4)

    pv_r_arr = _to_arr(pv_r, 3)
    pc_r_arr = _to_arr(pc_r, 4)
    tv_r_arr = _to_arr(tv_r, 3)
    zpv_r_arr = _to_arr(zpv_r, 3)
    zpc_r_count = len(zpv_r) // 3
    zpc_r_arr = (
        np.tile(zero_rgba, (zpc_r_count, 1))
        if zpc_r_count > 0
        else np.empty((0, 4), dtype=np.float32)
    )
    ov_pos_r_arr = _to_arr(ov_pos_r, 3)
    ov_col_r_arr = _to_arr(ov_col_r, 4)

    # Apply transforms
    flat_transform = config.world_to_visual
    rot_transform = config.world_to_cyl_local

    pv_f_arr = _transform_verts(pv_f_arr, flat_transform)
    tv_f_arr = _transform_verts(tv_f_arr, flat_transform)
    zpv_f_arr = _transform_verts(zpv_f_arr, flat_transform)
    ov_pos_f_arr = _transform_verts(ov_pos_f_arr, flat_transform)

    pv_r_arr = _transform_verts(pv_r_arr, rot_transform)
    tv_r_arr = _transform_verts(tv_r_arr, rot_transform)
    zpv_r_arr = _transform_verts(zpv_r_arr, rot_transform)
    ov_pos_r_arr = _transform_verts(ov_pos_r_arr, rot_transform)

    # Apply cylinder wrapping per rotary layer segment.
    # transform_to_cylinder subdivides line pairs that span a large arc,
    # producing MORE vertices than the input.  We rebuild the rotary
    # arrays by concatenating expanded segments.
    _exp_pv_r: List[np.ndarray] = []
    _exp_pc_r: List[np.ndarray] = []
    _exp_tv_r: List[np.ndarray] = []
    _exp_zpv_r: List[np.ndarray] = []
    _exp_zpc_r: List[np.ndarray] = []
    _exp_ov_pos_r: List[np.ndarray] = []
    _exp_ov_col_r: List[np.ndarray] = []

    for seg in rotary_segments:
        d = seg["diameter"]
        if d <= 0:
            continue

        pv_s = seg["pv_start"]
        pv_e = seg.get("pv_end", rot_pv_cum)
        if pv_e > pv_s:
            pv_w, pc_w = _apply_cylinder(
                pv_r_arr[pv_s:pv_e], d, pc_r_arr[pv_s:pv_e]
            )
            assert pc_w is not None
            _exp_pv_r.append(pv_w)
            _exp_pc_r.append(pc_w)

        tv_s = seg["tv_start"]
        tv_e = seg.get("tv_end", rot_tv_cum)
        if tv_e > tv_s:
            tv_w, _ = _apply_cylinder(tv_r_arr[tv_s:tv_e], d)
            _exp_tv_r.append(tv_w)

        zpv_s = seg["zpv_vtx_start"]
        zpv_e = seg.get("zpv_vtx_end", len(zpv_r_arr))
        if zpv_e > zpv_s:
            zpv_w, zpc_w = _apply_cylinder(
                zpv_r_arr[zpv_s:zpv_e], d, zpc_r_arr[zpv_s:zpv_e]
            )
            assert zpc_w is not None
            _exp_zpv_r.append(zpv_w)
            _exp_zpc_r.append(zpc_w)

        ov_s = seg["ov_start"]
        ov_e = seg.get("ov_end", rot_ov_cum)
        if ov_e > ov_s:
            ov_pos_w, ov_col_w = _apply_cylinder(
                ov_pos_r_arr[ov_s:ov_e], d, ov_col_r_arr[ov_s:ov_e]
            )
            assert ov_col_w is not None
            _exp_ov_pos_r.append(ov_pos_w)
            _exp_ov_col_r.append(ov_col_w)

    if _exp_pv_r:
        pv_r_arr = np.concatenate(_exp_pv_r, axis=0)
        pc_r_arr = np.concatenate(_exp_pc_r, axis=0)
    if _exp_tv_r:
        tv_r_arr = np.concatenate(_exp_tv_r, axis=0)
    if _exp_zpv_r:
        zpv_r_arr = np.concatenate(_exp_zpv_r, axis=0)
        zpc_r_arr = np.concatenate(_exp_zpc_r, axis=0)
    if _exp_ov_pos_r:
        ov_pos_r_arr = np.concatenate(_exp_ov_pos_r, axis=0)
        ov_col_r_arr = np.concatenate(_exp_ov_col_r, axis=0)

    # Z-offset for travel and zero-power vertices
    if tv_f_arr.size > 0:
        tv_f_arr[:, 2] += Z_OFFSET_NON_POWERED
    if zpv_f_arr.size > 0:
        zpv_f_arr[:, 2] += Z_OFFSET_NON_POWERED
    if tv_r_arr.size > 0:
        tv_r_arr[:, 2] += Z_OFFSET_NON_POWERED
    if zpv_r_arr.size > 0:
        zpv_r_arr[:, 2] += Z_OFFSET_NON_POWERED

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

        texture_layers.append(
            TextureLayer(
                power_texture=tex_buf,
                width_px=w_px,
                height_px=h_px,
                model_matrix=final_model,
                color_lut=scan_lut.copy() if scan_lut is not None else None,
                rotary_diameter=diameter,
                rotary_enabled=is_rot,
                activation_cmd_idx=li["activation_cmd_idx"],
            )
        )

    vertex_layers: List[VertexLayer] = [
        VertexLayer(
            powered_verts=_to_flat(pv_f_arr),
            powered_colors=_to_flat(pc_f_arr),
            travel_verts=_to_flat(tv_f_arr),
            zero_power_verts=_to_flat(zpv_f_arr),
            zero_power_colors=_to_flat(zpc_f_arr),
            powered_cmd_offsets=flat_pv_off,
            travel_cmd_offsets=flat_tv_off,
        ),
        VertexLayer(
            powered_verts=_to_flat(pv_r_arr),
            powered_colors=_to_flat(pc_r_arr),
            travel_verts=_to_flat(tv_r_arr),
            zero_power_verts=_to_flat(zpv_r_arr),
            zero_power_colors=_to_flat(zpc_r_arr),
            powered_cmd_offsets=rot_pv_off,
            travel_cmd_offsets=rot_tv_off,
        ),
    ]

    overlay_layers: List[ScanlineOverlayLayer] = [
        ScanlineOverlayLayer(
            positions=_to_flat(ov_pos_f_arr),
            colors=_to_flat(ov_col_f_arr),
            cmd_offsets=flat_ov_off,
        ),
        ScanlineOverlayLayer(
            positions=_to_flat(ov_pos_r_arr),
            colors=_to_flat(ov_col_r_arr),
            cmd_offsets=rot_ov_off,
        ),
    ]

    return CompiledSceneArtifact(
        generation_id=0,
        vertex_layers=vertex_layers,
        texture_layers=texture_layers,
        overlay_layers=overlay_layers,
    )
