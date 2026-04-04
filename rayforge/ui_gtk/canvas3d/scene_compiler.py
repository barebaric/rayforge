"""
Scene compiler: pure function that compiles Ops into GPU-ready vertex data.

Replaces:
- VertexEncoder (pipeline-side vertex encoding)
- prepare_scene_vertices_async (inline color-mapping, transforms,
  cylinder-wrapping)
- build_scanline_overlay + _upload_scanline_overlay (scanline overlay
  encoding and upload)

All three encoding passes are unified into a single pass per step.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from ...core.geo.linearize import linearize_arc
from ...core.ops import Ops
from ...core.ops.commands import (
    ArcToCommand,
    LineToCommand,
    MoveToCommand,
    ScanLinePowerCommand,
    SetPowerCommand,
)
from ...pipeline.encoder.vertexencoder import transform_to_cylinder
from .compiled_scene import (
    CompiledSceneArtifact,
    ScanlineOverlayLayer,
    VertexLayer,
)
from .render_config import RenderConfig3D, StepRenderConfig

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


@dataclass
class _StepResult:
    pv: np.ndarray
    pc: np.ndarray
    tv: np.ndarray
    zpv: np.ndarray
    zpc: np.ndarray
    ov_pos: np.ndarray
    ov_col: np.ndarray
    ov_off: List[int]


def _walk_step_ops(
    ops: Ops,
    cut_lut: np.ndarray,
    engrave_lut: Optional[np.ndarray],
    zero_rgba: np.ndarray,
) -> _StepResult:
    powered_v: List[float] = []
    powered_c: List[float] = []
    travel_v: List[float] = []
    zero_power_v: List[float] = []
    ov_pos: List[float] = []
    ov_col: List[float] = []
    ov_off: List[int] = [0]
    ov_cumulative = 0

    current_power = 0.0
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_initial = True

    for cmd in ops.commands:
        if isinstance(cmd, SetPowerCommand):
            current_power = cmd.power
            ov_off.append(ov_cumulative)
            continue

        if isinstance(cmd, MoveToCommand):
            if not is_initial:
                travel_v.extend(current_pos)
                travel_v.extend(cmd.end)
            current_pos = cmd.end
            is_initial = False
            ov_off.append(ov_cumulative)
            continue

        if isinstance(cmd, LineToCommand):
            if current_power > 0.0:
                power_byte = min(255, int(current_power * 255.0))
                color = cut_lut[power_byte]
                powered_v.extend(current_pos)
                powered_v.extend(cmd.end)
                powered_c.extend(color)
                powered_c.extend(color)
            else:
                zero_power_v.extend(current_pos)
                zero_power_v.extend(cmd.end)
            current_pos = cmd.end
            is_initial = False
            ov_off.append(ov_cumulative)
            continue

        if isinstance(cmd, ArcToCommand):
            segments = linearize_arc(cmd, current_pos)
            if current_power > 0.0:
                power_byte = min(255, int(current_power * 255.0))
                color = cut_lut[power_byte]
                for seg_start, seg_end in segments:
                    powered_v.extend(seg_start)
                    powered_v.extend(seg_end)
                    powered_c.extend(color)
                    powered_c.extend(color)
            else:
                for seg_start, seg_end in segments:
                    zero_power_v.extend(seg_start)
                    zero_power_v.extend(seg_end)
            current_pos = cmd.end
            is_initial = False
            ov_off.append(ov_cumulative)
            continue

        if isinstance(cmd, ScanLinePowerCommand):
            if cmd.end is not None:
                _extract_zero_power_segments(cmd, current_pos, zero_power_v)
                if not is_initial:
                    n = _encode_overlay_segments(
                        cmd,
                        current_pos,
                        engrave_lut,
                        ov_pos,
                        ov_col,
                    )
                    ov_cumulative += n
                current_pos = cmd.end
                is_initial = False
            ov_off.append(ov_cumulative)
            continue

        ov_off.append(ov_cumulative)

    num_zp = len(zero_power_v) // 3
    zpc = (
        np.tile(zero_rgba, (num_zp, 1))
        if num_zp > 0
        else np.empty((0, 4), dtype=np.float32)
    )

    return _StepResult(
        pv=np.array(powered_v, dtype=np.float32).reshape(-1, 3),
        pc=np.array(powered_c, dtype=np.float32).reshape(-1, 4),
        tv=np.array(travel_v, dtype=np.float32).reshape(-1, 3),
        zpv=np.array(zero_power_v, dtype=np.float32).reshape(-1, 3),
        zpc=zpc,
        ov_pos=np.array(ov_pos, dtype=np.float32).reshape(-1, 3),
        ov_col=np.array(ov_col, dtype=np.float32).reshape(-1, 4),
        ov_off=ov_off,
    )


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


def compile_scene(
    ops_list: List[Tuple[Ops, StepRenderConfig]],
    config: RenderConfig3D,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> CompiledSceneArtifact:
    pv_f: List[np.ndarray] = []
    pc_f: List[np.ndarray] = []
    tv_f: List[np.ndarray] = []
    zpv_f: List[np.ndarray] = []
    zpc_f: List[np.ndarray] = []
    ov_pos_f: List[np.ndarray] = []
    ov_col_f: List[np.ndarray] = []
    ov_off_f: List[int] = [0]
    ov_cum_f = 0

    pv_r: List[np.ndarray] = []
    pc_r: List[np.ndarray] = []
    tv_r: List[np.ndarray] = []
    zpv_r: List[np.ndarray] = []
    zpc_r: List[np.ndarray] = []
    ov_pos_r: List[np.ndarray] = []
    ov_col_r: List[np.ndarray] = []
    ov_off_r: List[int] = [0]
    ov_cum_r = 0

    zero_rgba = np.array(config.zero_power_rgba, dtype=np.float32)

    for ops, step_cfg in ops_list:
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")

        is_rot = step_cfg.rotary_enabled
        transform = (
            config.world_to_cyl_local if is_rot else config.world_to_visual
        )
        diameter = step_cfg.rotary_diameter if is_rot else 0.0
        cut_lut = _get_lut(config, step_cfg.laser_uid, "cut")
        assert cut_lut is not None
        engrave_lut = _get_lut(config, step_cfg.laser_uid, "engrave")

        result = _walk_step_ops(ops, cut_lut, engrave_lut, zero_rgba)

        pv = _transform_verts(result.pv, transform)
        tv = _transform_verts(result.tv, transform)
        zpv = _transform_verts(result.zpv, transform)
        pc = result.pc
        zpc = result.zpc
        ov_p = _transform_verts(result.ov_pos, transform)
        ov_c = result.ov_col

        if is_rot and diameter > 0:
            if pv.size > 0:
                pv, pc = _apply_cylinder(pv, diameter, pc)
                assert pc is not None
            if tv.size > 0:
                tv, _ = _apply_cylinder(tv, diameter)
            if zpv.size > 0:
                zpv, zpc = _apply_cylinder(zpv, diameter, zpc)
                assert zpc is not None
            if ov_p.size > 0:
                ov_p, ov_c = _apply_cylinder(ov_p, diameter, ov_c)
                assert ov_c is not None

        if is_rot:
            pv_r.append(pv)
            pc_r.append(pc)
            tv_r.append(tv)
            zpv_r.append(zpv)
            zpc_r.append(zpc)
            if ov_p.size > 0:
                ov_pos_r.append(ov_p)
                ov_col_r.append(ov_c)
            adjusted = [ov_cum_r + x for x in result.ov_off[1:]]
            ov_off_r.extend(adjusted)
            ov_cum_r += result.ov_off[-1]
        else:
            pv_f.append(pv)
            pc_f.append(pc)
            tv_f.append(tv)
            zpv_f.append(zpv)
            zpc_f.append(zpc)
            if ov_p.size > 0:
                ov_pos_f.append(ov_p)
                ov_col_f.append(ov_c)
            adjusted = [ov_cum_f + x for x in result.ov_off[1:]]
            ov_off_f.extend(adjusted)
            ov_cum_f += result.ov_off[-1]

    if tv_f:
        tv_all = np.concatenate(tv_f)
        if tv_all.ndim > 1 and tv_all.size > 0:
            tv_all[:, 2] += Z_OFFSET_NON_POWERED
        tv_f = [tv_all]
    if zpv_f:
        zpv_all = np.concatenate(zpv_f)
        if zpv_all.ndim > 1 and zpv_all.size > 0:
            zpv_all[:, 2] += Z_OFFSET_NON_POWERED
        zpv_f = [zpv_all]

    def _cat(lists):
        if not lists:
            return np.empty((0,), dtype=np.float32)
        parts = [a.ravel() for a in lists if a.size > 0]
        if not parts:
            return np.empty((0,), dtype=np.float32)
        return np.concatenate(parts)

    vertex_layers: List[VertexLayer] = [
        VertexLayer(
            powered_verts=_cat(pv_f),
            powered_colors=_cat(pc_f),
            travel_verts=_cat(tv_f),
            zero_power_verts=_cat(zpv_f),
            zero_power_colors=_cat(zpc_f),
        ),
        VertexLayer(
            powered_verts=_cat(pv_r),
            powered_colors=_cat(pc_r),
            travel_verts=_cat(tv_r),
            zero_power_verts=_cat(zpv_r),
            zero_power_colors=_cat(zpc_r),
        ),
    ]

    overlay_layers: List[ScanlineOverlayLayer] = [
        ScanlineOverlayLayer(
            positions=_cat(ov_pos_f),
            colors=_cat(ov_col_f),
            cmd_offsets=ov_off_f,
        ),
        ScanlineOverlayLayer(
            positions=_cat(ov_pos_r),
            colors=_cat(ov_col_r),
            cmd_offsets=ov_off_r,
        ),
    ]

    return CompiledSceneArtifact(
        generation_id=0,
        vertex_layers=vertex_layers,
        texture_layers=[],
        overlay_layers=overlay_layers,
    )
