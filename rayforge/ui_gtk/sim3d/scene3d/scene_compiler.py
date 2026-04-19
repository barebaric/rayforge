"""
Scene compiler: pure function that compiles assembled job Ops into
GPU-ready vertex data.

Walks the full assembled job ops (with JobStart/End, LayerStart/End
markers) to produce vertex buffers and per-command offset arrays that
are indexed 1:1 with the player's command index.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from ....core.geo.arc import linearize_arc
from ....core.geo.bezier import linearize_bezier_segment
from ....core.ops import Ops
from ....core.ops.axis import Axis
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
from ....machine.kinematic_math import KinematicMath
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


def _degrees_to_mu(
    degrees: float,
    diameter: float,
    gear_ratio: float,
    reverse: bool,
) -> float:
    if diameter <= 0 or gear_ratio <= 0:
        return degrees
    sign = -1.0 if reverse else 1.0
    return degrees * math.pi * diameter / 360.0 / gear_ratio * sign


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


def _get_layer_lut(
    config: RenderConfig3D,
    layer_uid: str,
    lut_type: str,
) -> Optional[np.ndarray]:
    if config.layer_color_luts is None:
        return None
    layer_luts = config.layer_color_luts.get(layer_uid, {})
    lut_bytes = layer_luts.get(lut_type)
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
    engrave_lut: Optional[np.ndarray],
    ov_pos: List[float],
    ov_col: List[float],
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

    dilated = np.zeros_like(buffer)
    for dy in range(-1, 2):
        for dx in range(-1, 2):
            shifted = np.roll(np.roll(buffer, dy, axis=0), dx, axis=1)
            np.maximum(dilated, shifted, out=dilated)
    buffer = dilated

    return buffer, width_px, height_px, px_per_mm


def _find_degrees(cmd: MovingCommand) -> Optional[float]:
    for ax in (Axis.A, Axis.B, Axis.C, Axis.U, Axis.Y):
        val = cmd.extra_axes.get(ax)
        if val is not None:
            return val
    return None


def _visual_end(
    cmd: MovingCommand,
) -> Tuple[float, float, float]:
    degrees = _find_degrees(cmd)
    if degrees is not None:
        pos = list(cmd.end)
        pos[1] = degrees
        return (pos[0], pos[1], pos[2])
    return cmd.end


def _reconstruct_mu_pos(
    cmd: MovingCommand,
    diameter: float,
    gear_ratio: float,
    reverse: bool,
) -> Tuple[float, float, float]:
    degrees = _find_degrees(cmd)
    if degrees is None:
        return cmd.end
    mu_val = _degrees_to_mu(degrees, diameter, gear_ratio, reverse)
    pos = list(cmd.end)
    pos[1] = mu_val
    return (pos[0], pos[1], pos[2])


def _reconstruct_mu_arc(
    cmd: ArcToCommand,
    diameter: float,
    gear_ratio: float,
    reverse: bool,
) -> ArcToCommand:
    degrees = _find_degrees(cmd)
    if degrees is None:
        return cmd
    scale = _degrees_to_mu(1.0, diameter, gear_ratio, reverse)

    pos = list(cmd.end)
    pos[1] = degrees * scale

    offset = list(cmd.center_offset)
    offset[1] = offset[1] * scale

    return ArcToCommand(
        end=(pos[0], pos[1], pos[2]),
        center_offset=(offset[0], offset[1]),
        clockwise=cmd.clockwise,
        extra_axes=dict(cmd.extra_axes),
    )


def _reconstruct_mu_bezier(
    cmd: BezierToCommand,
    diameter: float,
    gear_ratio: float,
    reverse: bool,
) -> BezierToCommand:
    degrees = _find_degrees(cmd)
    if degrees is None:
        return cmd
    scale = _degrees_to_mu(1.0, diameter, gear_ratio, reverse)

    pos = list(cmd.end)
    pos[1] = degrees * scale

    cp1 = list(cmd.control1)
    cp1[1] = cp1[1] * scale

    cp2 = list(cmd.control2)
    cp2[1] = cp2[1] * scale

    return BezierToCommand(
        end=(pos[0], pos[1], pos[2]),
        control1=(cp1[0], cp1[1], cp1[2]),
        control2=(cp2[0], cp2[1], cp2[2]),
        extra_axes=dict(cmd.extra_axes),
    )


def _mu_to_visual(
    pos: Tuple[float, float, float],
    diameter: float,
    gear_ratio: float,
    reverse: bool,
) -> Tuple[float, float, float]:
    degrees = KinematicMath.mu_to_degrees(
        pos[1], diameter, gear_ratio=gear_ratio, reverse=reverse
    )
    result = list(pos)
    result[1] = degrees
    return (result[0], result[1], result[2])


def _bake_visual_positions(
    ops: Ops,
) -> Ops:
    baked = Ops()
    for cmd in ops.commands:
        if isinstance(cmd, MovingCommand):
            degrees = _find_degrees(cmd)
            if degrees is not None:
                pos = list(cmd.end)
                pos[1] = degrees
                new_end = (pos[0], pos[1], pos[2])
                if isinstance(cmd, ScanLinePowerCommand):
                    new_cmd = ScanLinePowerCommand(
                        new_end,
                        cmd.power_values,
                        extra_axes=dict(cmd.extra_axes),
                    )
                elif isinstance(cmd, ArcToCommand):
                    new_cmd = ArcToCommand(
                        new_end,
                        cmd.center_offset,
                        cmd.clockwise,
                        extra_axes=dict(cmd.extra_axes),
                    )
                elif isinstance(cmd, BezierToCommand):
                    new_cmd = BezierToCommand(
                        new_end,
                        cmd.control1,
                        cmd.control2,
                        extra_axes=dict(cmd.extra_axes),
                    )
                else:
                    new_cmd = cmd.__class__(
                        new_end,
                        extra_axes=dict(cmd.extra_axes),
                    )
                new_cmd.state = cmd.state
                baked.add(new_cmd)
            else:
                baked.add(cmd)
        else:
            baked.add(cmd)
    return baked


@dataclass
class _WalkState:
    current_power: float = 0.0
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    current_pos_vis: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_initial: bool = True
    current_laser_uid: str = ""
    is_rotary: bool = False
    rotary_diameter: float = 0.0
    has_mapped_data: bool = False
    layer_infos: List[dict] = field(default_factory=list)
    current_layer_start: Optional[int] = None
    current_layer_has_scanlines: bool = False
    current_layer_scanline_laser: str = ""
    current_cut_lut: np.ndarray = field(
        default_factory=lambda: np.zeros((256, 4), dtype=np.float32)
    )
    current_engrave_lut: np.ndarray = field(
        default_factory=lambda: np.zeros((256, 4), dtype=np.float32)
    )


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
        "axis_position",
        "gear_ratio",
        "reverse",
        "diameter",
        "axis_position_3d",
        "cylinder_dir",
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
        self.axis_position: float = 0.0
        self.gear_ratio: float = 1.0
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
                pv_expansion,
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
            if pv_expansion:
                self.pv_off = _remap_offsets(self.pv_off, pv_expansion)

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
    pv_expansion: List[Tuple[int, np.ndarray]] = []

    for seg in rotary_segments:
        d = seg["diameter"]
        deg_in = seg.get("degrees_input", False)
        if d <= 0:
            continue

        pv_s = seg["pv_start"]
        pv_e = seg.get("pv_end", pv_cum)
        if pv_e > pv_s:
            pv_w, pc_w, cum_subs = _apply_cylinder(
                pv_arr[pv_s:pv_e],
                d,
                pc_arr[pv_s:pv_e],
                degrees_input=deg_in,
            )
            assert pc_w is not None
            _exp_pv.append(pv_w)
            _exp_pc.append(pc_w)
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
            zpv_w, zpc_w, _ = _apply_cylinder(
                zpv_arr[zpv_s:zpv_e],
                d,
                zpc_arr[zpv_s:zpv_e],
                degrees_input=deg_in,
            )
            assert zpc_w is not None
            _exp_zpv.append(zpv_w)
            _exp_zpc.append(zpc_w)

        ov_s = seg["ov_start"]
        ov_e = seg.get("ov_end", ov_cum)
        if ov_e > ov_s:
            ov_pos_w, ov_col_w, _ = _apply_cylinder(
                ov_pos_arr[ov_s:ov_e],
                d,
                ov_col_arr[ov_s:ov_e],
                degrees_input=deg_in,
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
        pv_expansion,
    )


def _finalize_layers(
    accumulators: dict[bool, _LayerAccumulator],
    config: RenderConfig3D,
) -> Tuple[List[VertexLayer], List[ScanlineOverlayLayer]]:
    zero_rgba = np.array(config.zero_power_rgba, dtype=np.float32)
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

    return texture_layers


def _handle_layer_start(
    st: _WalkState,
    acc: _LayerAccumulator,
    cmd_idx: int,
    cmd: LayerStartCommand,
    config: RenderConfig3D,
    use_layer_colors: bool,
    accumulators: dict[bool, _LayerAccumulator],
) -> _LayerAccumulator:
    layer_uid = cmd.layer_uid
    layer_cfg = None
    if config.layer_configs and layer_uid in config.layer_configs:
        layer_cfg = config.layer_configs[layer_uid]
    st.is_rotary = layer_cfg.rotary_enabled if layer_cfg else False
    st.rotary_diameter = (
        layer_cfg.rotary_diameter if layer_cfg else 0.0
    )
    acc = accumulators[st.is_rotary]

    acc.axis_position = layer_cfg.axis_position if layer_cfg else 0.0
    acc.gear_ratio = layer_cfg.gear_ratio if layer_cfg else 1.0
    acc.reverse = layer_cfg.reverse if layer_cfg else False
    acc.diameter = st.rotary_diameter
    acc.axis_position_3d = (
        layer_cfg.axis_position_3d if layer_cfg else None
    )
    acc.cylinder_dir = (
        layer_cfg.cylinder_dir if layer_cfg else None
    )
    st.has_mapped_data = st.is_rotary

    if st.is_rotary and st.rotary_diameter > 0:
        acc.begin_rotary_segment(
            cmd_idx, degrees_input=st.has_mapped_data
        )

    st.current_layer_start = cmd_idx + 1
    st.current_layer_has_scanlines = False
    st.current_layer_scanline_laser = ""

    if use_layer_colors:
        _cut = _get_layer_lut(config, layer_uid, "cut")
        if _cut is not None:
            st.current_cut_lut = _cut
        _engrave = _get_layer_lut(config, layer_uid, "engrave")
        if _engrave is not None:
            st.current_engrave_lut = _engrave

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
                "gear_ratio": acc.gear_ratio,
                "reverse": acc.reverse,
            }
        )
    if acc.current_rotary_seg is not None:
        acc.end_rotary_segment(st.rotary_diameter, cmd_idx)
    st.current_layer_start = None


def _handle_set_laser(
    st: _WalkState,
    cmd: SetLaserCommand,
    config: RenderConfig3D,
    use_layer_colors: bool,
) -> None:
    st.current_laser_uid = cmd.laser_uid
    if not use_layer_colors:
        _cut = _get_lut(config, st.current_laser_uid, "cut")
        st.current_cut_lut = (
            _cut
            if _cut is not None
            else np.zeros((256, 4), dtype=np.float32)
        )
        _engrave = _get_lut(
            config, st.current_laser_uid, "engrave"
        )
        st.current_engrave_lut = (
            _engrave
            if _engrave is not None
            else np.zeros((256, 4), dtype=np.float32)
        )


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
            acc.gear_ratio,
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
        power_byte = min(255, int(st.current_power * 255.0))
        color = st.current_cut_lut[power_byte]
        acc.pv.extend(st.current_pos_vis)
        acc.pv.extend(vis_end)
        acc.pc.extend(color)
        acc.pc.extend(color)
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
            acc.gear_ratio,
            acc.reverse,
        )
        segments = linearize_arc(mu_cmd, st.current_pos)
        vis_segs = []
        for seg_start, seg_end in segments:
            vis_start = _mu_to_visual(
                seg_start,
                acc.diameter,
                acc.gear_ratio,
                acc.reverse,
            )
            vis_end_pt = _mu_to_visual(
                seg_end,
                acc.diameter,
                acc.gear_ratio,
                acc.reverse,
            )
            vis_segs.append((vis_start, vis_end_pt))
    else:
        segments = linearize_arc(cmd, st.current_pos)
        vis_segs = segments
    if st.current_power > 0.0:
        power_byte = min(255, int(st.current_power * 255.0))
        color = st.current_cut_lut[power_byte]
        n_segs = len(vis_segs)
        for seg_start, seg_end in vis_segs:
            acc.pv.extend(seg_start)
            acc.pv.extend(seg_end)
            acc.pc.extend(color)
            acc.pc.extend(color)
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
            acc.gear_ratio,
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
                acc.gear_ratio,
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
        power_byte = min(255, int(st.current_power * 255.0))
        color = st.current_cut_lut[power_byte]
        for j in range(len(vis_poly) - 1):
            acc.pv.extend(vis_poly[j])
            acc.pv.extend(vis_poly[j + 1])
            acc.pc.extend(color)
            acc.pc.extend(color)
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
                st.current_engrave_lut,
                acc.ov_pos,
                acc.ov_col,
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

    default_cut = _get_lut(config, "", "cut")
    if default_cut is None:
        default_cut = np.zeros((256, 4), dtype=np.float32)
    default_engrave = _get_lut(config, "", "engrave")
    if default_engrave is None:
        default_engrave = np.zeros((256, 4), dtype=np.float32)

    st = _WalkState(
        current_cut_lut=default_cut,
        current_engrave_lut=default_engrave,
    )

    use_layer_colors = (
        config.ops_color_mode == "layer"
        if isinstance(config.ops_color_mode, str)
        else config.ops_color_mode.value == "layer"
    )

    for i, cmd in enumerate(ops.commands):
        if cancel_check is not None and cancel_check():
            raise RuntimeError("Cancelled")

        acc = accumulators[st.is_rotary]

        if isinstance(cmd, LayerStartCommand):
            acc = _handle_layer_start(
                st, acc, i, cmd, config, use_layer_colors,
                accumulators,
            )
        elif isinstance(cmd, LayerEndCommand):
            _handle_layer_end(st, acc, i)
        elif isinstance(cmd, SetLaserCommand):
            _handle_set_laser(st, cmd, config, use_layer_colors)
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

    vertex_layers, overlay_layers = _finalize_layers(
        accumulators, config
    )

    texture_layers = _generate_texture_layers(
        ops, st.layer_infos, config
    )

    return CompiledSceneArtifact(
        generation_id=0,
        vertex_layers=vertex_layers,
        texture_layers=texture_layers,
        overlay_layers=overlay_layers,
    )
