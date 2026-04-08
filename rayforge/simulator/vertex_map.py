from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np

from ..core.geo.arc import linearize_arc
from ..core.ops import Ops
from ..core.ops.commands import (
    ArcToCommand,
    Command,
    LineToCommand,
    MoveToCommand,
    ScanLinePowerCommand,
    SetPowerCommand,
)


@dataclass
class SceneVertexMap:
    command_vertex_offset: List[int] = field(default_factory=list)
    total_powered_vertices: int = 0
    command_travel_vertex_offset: List[int] = field(default_factory=list)
    total_travel_vertices: int = 0


def build_vertex_map(ops: Ops) -> SceneVertexMap:
    """
    Walks the global Ops and counts powered and travel vertices per
    command, producing cumulative offset arrays.

    command_vertex_offset[i] = powered vertex count for commands
    0..i-1. Length is len(ops.commands) + 1.
    """
    offsets: List[int] = []
    travel_offsets: List[int] = []
    cumulative = 0
    travel_cumulative = 0
    current_power = 0.0
    current_pos = (0.0, 0.0, 0.0)
    is_initial_position = True

    for cmd in ops.commands:
        offsets.append(cumulative)
        travel_offsets.append(travel_cumulative)
        powered_count = _count_powered_vertices(
            cmd, current_power, current_pos, is_initial_position
        )
        travel_count = _count_travel_vertices(cmd, is_initial_position)
        cumulative += powered_count
        travel_cumulative += travel_count
        if isinstance(cmd, SetPowerCommand):
            current_power = cmd.power
        elif isinstance(cmd, MoveToCommand):
            current_pos = cmd.end
            is_initial_position = False
        elif isinstance(cmd, (LineToCommand, ArcToCommand)):
            current_pos = cmd.end
            is_initial_position = False
        elif isinstance(cmd, ScanLinePowerCommand):
            if cmd.end is not None:
                current_pos = cmd.end
                is_initial_position = False

    offsets.append(cumulative)
    travel_offsets.append(travel_cumulative)
    return SceneVertexMap(
        command_vertex_offset=offsets,
        total_powered_vertices=cumulative,
        command_travel_vertex_offset=travel_offsets,
        total_travel_vertices=travel_cumulative,
    )


def _count_powered_vertices(
    cmd: Command,
    current_power: float,
    current_pos: tuple,
    is_initial_position: bool,
) -> int:
    """Returns the number of powered vertices produced by a command."""
    if isinstance(cmd, SetPowerCommand):
        return 0

    if isinstance(cmd, MoveToCommand):
        return 0

    if isinstance(cmd, LineToCommand):
        if current_power > 0.0:
            return 2
        return 0

    if isinstance(cmd, ArcToCommand):
        if current_power > 0.0:
            segments = linearize_arc(cmd, current_pos)
            return len(segments) * 2
        return 0

    if isinstance(cmd, ScanLinePowerCommand):
        return 0

    return 0


def _count_travel_vertices(cmd: Command, is_initial_position: bool) -> int:
    """Returns the number of travel vertices produced by a command."""
    if isinstance(cmd, MoveToCommand) and not is_initial_position:
        return 2
    return 0


@dataclass
class ScanlineOverlay:
    """
    Encoded powered pixel-segments for all ScanLinePowerCommands,
    ready for GPU upload.

    positions: flat float32 array, shape (N*3,) — x,y,z per vertex.
    colors:    flat float32 array, shape (N*4,) — rgba per vertex.
    cmd_vertex_offset: cumulative overlay-vertex count; same layout as
        SceneVertexMap.command_vertex_offset —
        cmd_vertex_offset[i] = overlay vertices for scanline commands
        0..global_cmd_index[i]-1.
        Length == len(global_ops.commands) + 1.
    total_overlay_vertices: total number of vertices in positions/colors.
    """

    positions: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    colors: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float32)
    )
    cmd_vertex_offset: List[int] = field(default_factory=list)
    total_overlay_vertices: int = 0


def _encode_scanline_segments(
    cmd: ScanLinePowerCommand,
    start_pos: Tuple[float, float, float],
) -> Tuple[List[float], List[float], int]:
    """
    Encode powered pixel-segments of a single ScanLinePowerCommand.

    Returns (positions_flat, colors_flat, vertex_count).
    Positions are in world space (same coordinate system as the ops).
    Colors encode power as normalized float (0..1) in the red channel.
    The caller must apply a color LUT to produce final RGBA colors.
    """
    if cmd.end is None:
        return [], [], 0

    num_steps = len(cmd.power_values)
    if num_steps == 0:
        return [], [], 0

    sx, sy, sz = start_pos
    ex, ey, ez = cmd.end
    dx = (ex - sx) / num_steps
    dy = (ey - sy) / num_steps
    dz = (ez - sz) / num_steps

    positions: List[float] = []
    colors: List[float] = []
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
            positions.extend(
                [
                    seg_start_x,
                    seg_start_y,
                    seg_start_z,
                    seg_end_x,
                    seg_end_y,
                    seg_end_z,
                ]
            )
            colors.extend([seg_power, seg_power, seg_power, 1.0] * 2)
            vertex_count += 2

        prev_power_on = power_on

    if prev_power_on:
        positions.extend([seg_start_x, seg_start_y, seg_start_z, ex, ey, ez])
        colors.extend([seg_power, seg_power, seg_power, 1.0] * 2)
        vertex_count += 2

    return positions, colors, vertex_count


def build_scanline_overlay(ops: Ops) -> ScanlineOverlay:
    """
    Walk the global Ops and encode powered segments of every
    ScanLinePowerCommand into a vertex overlay.

    Also produces cmd_vertex_offset — a per-command-index cumulative
    vertex count (same shape as SceneVertexMap.command_vertex_offset)
    so the caller can map the playhead command index to an overlay
    vertex count.
    """
    all_pos: List[float] = []
    all_col: List[float] = []
    cmd_offsets: List[int] = []
    cumulative = 0
    current_pos = (0.0, 0.0, 0.0)
    is_initial_position = True

    for cmd in ops.commands:
        cmd_offsets.append(cumulative)
        if isinstance(cmd, ScanLinePowerCommand) and cmd.end is not None:
            if not is_initial_position:
                p, c, n = _encode_scanline_segments(cmd, current_pos)
                all_pos.extend(p)
                all_col.extend(c)
                cumulative += n

        if isinstance(cmd, MoveToCommand):
            current_pos = cmd.end
            is_initial_position = False
        elif isinstance(cmd, (LineToCommand, ArcToCommand)):
            current_pos = cmd.end
            is_initial_position = False
        elif isinstance(cmd, ScanLinePowerCommand):
            if cmd.end is not None:
                current_pos = cmd.end
                is_initial_position = False

    cmd_offsets.append(cumulative)

    return ScanlineOverlay(
        positions=np.array(all_pos, dtype=np.float32),
        colors=np.array(all_col, dtype=np.float32),
        cmd_vertex_offset=cmd_offsets,
        total_overlay_vertices=cumulative,
    )
