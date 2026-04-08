from __future__ import annotations

import math
from copy import deepcopy
from typing import List, Optional, Tuple

import numpy as np

from ..geo import clipping as geo_clipping
from ..geo.constants import (
    CMD_TYPE_ARC,
    CMD_TYPE_LINE,
    COL_CW,
    COL_I,
    COL_J,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
)
from ..geo.fitting import fit_points_to_primitives
from ..geo.arc import is_arc_fully_inside_regions
from ..geo.types import Point3D, Polygon
from .commands import (
    ArcToCommand,
    LineToCommand,
    MoveToCommand,
    MovingCommand,
    ScanLinePowerCommand,
)
from .container import Ops


def clip_ops_to_regions(
    ops: Ops,
    regions: List[List[Tuple[float, float]]],
    tolerance: float = 0.3,
) -> Ops:
    valid_regions: List[Polygon] = [r for r in regions if len(r) >= 3]
    if not valid_regions:
        return ops

    new_ops = Ops()
    last_point: Point3D = (0.0, 0.0, 0.0)
    pen_pos: Optional[Point3D] = None

    first_move_idx = next(
        (
            i
            for i, cmd in enumerate(ops._commands)
            if isinstance(cmd, MovingCommand)
        ),
        len(ops._commands),
    )
    for i in range(first_move_idx):
        new_ops.add(deepcopy(ops._commands[i]))

    for cmd in ops._commands[first_move_idx:]:
        if not isinstance(cmd, MovingCommand) or cmd.end is None:
            if not new_ops._commands or new_ops._commands[-1] is not cmd:
                new_ops.add(deepcopy(cmd))
            continue

        if isinstance(cmd, MoveToCommand):
            last_point = cmd.end
            pen_pos = None
            continue

        if isinstance(cmd, ScanLinePowerCommand):
            pen_pos = _clip_scanline(
                new_ops,
                cmd,
                last_point,
                pen_pos,
                valid_regions,
            )
            last_point = cmd.end
            continue

        if isinstance(cmd, ArcToCommand):
            if is_arc_fully_inside_regions(
                (last_point[0], last_point[1]),
                (cmd.end[0], cmd.end[1]),
                cmd.center_offset,
                cmd.clockwise,
                valid_regions,
            ):
                if pen_pos is None or math.dist(pen_pos, last_point) > 1e-6:
                    new_ops.move_to(*last_point)
                new_ops.add(deepcopy(cmd))
                pen_pos = cmd.end
                last_point = cmd.end
                continue

            pen_pos = _clip_and_refit_arc(
                new_ops,
                cmd,
                last_point,
                pen_pos,
                valid_regions,
                tolerance,
            )
            last_point = cmd.end
            continue

        linearized_commands = cmd.linearize(last_point)
        p_seg_start = last_point
        for l_cmd in linearized_commands:
            if l_cmd.end is None:
                continue
            p_seg_end = l_cmd.end
            kept_segments = geo_clipping.clip_line_segment_to_regions(
                p_seg_start,
                p_seg_end,
                valid_regions,
            )
            for sub_p1, sub_p2 in kept_segments:
                if pen_pos is None or math.dist(pen_pos, sub_p1) > 1e-6:
                    new_ops.move_to(*sub_p1)
                new_ops.line_to(*sub_p2)
                pen_pos = sub_p2
            p_seg_start = p_seg_end

        last_point = cmd.end

    ops._commands = new_ops._commands
    ops._invalidate_time_cache()
    if new_ops._commands:
        for cmd_rev in reversed(new_ops._commands):
            if isinstance(cmd_rev, MoveToCommand):
                ops.last_move_to = cmd_rev.end
                break
    return ops


def _clip_scanline(
    new_ops: Ops,
    cmd: ScanLinePowerCommand,
    last_point: Point3D,
    pen_pos: Optional[Point3D],
    valid_regions: List[Polygon],
) -> Optional[Point3D]:
    kept_segments = geo_clipping.clip_line_segment_to_regions(
        last_point, cmd.end, valid_regions
    )
    num_values = len(cmd.power_values)
    p_start_orig = np.array(last_point)
    p_end_orig = np.array(cmd.end)
    vec_orig = p_end_orig - p_start_orig
    len_sq = np.dot(vec_orig, vec_orig)

    for new_start, new_end in kept_segments:
        if len_sq > 1e-9:
            t_start = (
                np.dot(np.array(new_start) - p_start_orig, vec_orig) / len_sq
            )
            t_end = np.dot(np.array(new_end) - p_start_orig, vec_orig) / len_sq
        else:
            t_start, t_end = 0.0, 1.0

        idx_start = int(num_values * t_start)
        idx_end = int(num_values * t_end)
        new_power = cmd.power_values[idx_start:idx_end]

        if new_power:
            if pen_pos is None or math.dist(pen_pos, new_start) > 1e-6:
                new_ops.move_to(*new_start)
            new_ops.add(ScanLinePowerCommand(new_end, new_power))
            pen_pos = new_end
    return pen_pos


def _clip_and_refit_arc(
    new_ops: Ops,
    cmd: ArcToCommand,
    last_point: Point3D,
    pen_pos: Optional[Point3D],
    valid_regions: List[Polygon],
    tolerance: float,
) -> Optional[Point3D]:
    arc_state = cmd.state
    linearized_commands = cmd.linearize(last_point)
    kept_pairs = []
    p_seg_start = last_point
    for l_cmd in linearized_commands:
        if l_cmd.end is None:
            continue
        p_seg_end = l_cmd.end
        segs = geo_clipping.clip_line_segment_to_regions(
            p_seg_start,
            p_seg_end,
            valid_regions,
        )
        kept_pairs.extend(segs)
        p_seg_start = p_seg_end

    chains = []
    for p1, p2 in kept_pairs:
        if chains and math.dist(chains[-1][-1], p1) <= 1e-6:
            chains[-1].append(p2)
        else:
            chains.append([p1, p2])

    for chain in chains:
        primitives = fit_points_to_primitives(chain, tolerance)
        if not primitives:
            continue
        if pen_pos is None or math.dist(pen_pos, chain[0]) > 1e-6:
            new_ops.move_to(*chain[0])
        for prim_row in primitives:
            ct = prim_row[COL_TYPE]
            end = (
                prim_row[COL_X],
                prim_row[COL_Y],
                prim_row[COL_Z],
            )
            if ct == CMD_TYPE_LINE:
                new_cmd = LineToCommand(end)
            elif ct == CMD_TYPE_ARC:
                co = (prim_row[COL_I], prim_row[COL_J])
                cw = bool(prim_row[COL_CW])
                new_cmd = ArcToCommand(end, co, cw)
            else:
                continue
            new_cmd.state = arc_state
            new_ops.add(new_cmd)
        pen_pos = chain[-1]

    return pen_pos
