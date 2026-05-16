from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np

from raygeo import (
    Point3D,
    Polygon,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    CMD_TYPE_LINE,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
    COL_CW,
    COL_I,
    COL_J,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    clip_line_segment_with_polygons,
    is_arc_inside_polygons,
    is_bezier_inside_polygons,
    fit_points_with_primitives,
)
from .container import Ops, MachineState
from .enums import CommandType, CommandCategory


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
            for i in range(ops.len())
            if ops.category(i) == CommandCategory.MOVING
        ),
        ops.len(),
    )
    for i in range(first_move_idx):
        new_ops.copy_command_from(ops, i)

    for i in range(first_move_idx, ops.len()):
        ct = ops.command_type(i)
        cat = ops.category(i)

        if cat != CommandCategory.MOVING or ops.endpoint(i) is None:
            new_ops.copy_command_from(ops, i)
            continue

        end = ops.endpoint(i)

        if ct == CommandType.MOVE_TO:
            last_point = end
            pen_pos = None
            continue

        if ct == CommandType.SCAN_LINE:
            pen_pos = _clip_scanline(
                new_ops,
                ops,
                i,
                last_point,
                pen_pos,
                valid_regions,
            )
            last_point = end
            continue

        if ct == CommandType.ARC_TO:
            arc_i, arc_j, arc_cw = ops.arc_params(i)
            if is_arc_inside_polygons(
                (last_point[0], last_point[1]),
                (end[0], end[1]),
                (arc_i, arc_j),
                arc_cw,
                valid_regions,
            ):
                if pen_pos is None or math.dist(pen_pos, last_point) > 1e-6:
                    new_ops.move_to(*last_point)
                new_ops.copy_command_from(ops, i)
                pen_pos = end
                last_point = end
                continue

            pen_pos = _clip_and_refit_arc(
                new_ops,
                ops,
                i,
                last_point,
                pen_pos,
                valid_regions,
                tolerance,
            )
            last_point = end
            continue

        if ct == CommandType.BEZIER_TO:
            c1, c2 = ops.bezier_params(i)
            start_2d = (last_point[0], last_point[1])
            end_2d = (end[0], end[1])
            c1_2d = (c1[0], c1[1])
            c2_2d = (c2[0], c2[1])
            if is_bezier_inside_polygons(
                start_2d, c1_2d, c2_2d, end_2d, valid_regions
            ):
                if pen_pos is None or math.dist(pen_pos, last_point) > 1e-6:
                    new_ops.move_to(*last_point)
                new_ops.copy_command_from(ops, i)
                pen_pos = end
                last_point = end
                continue

            pen_pos = _clip_and_refit_bezier(
                new_ops,
                ops,
                i,
                last_point,
                pen_pos,
                valid_regions,
                tolerance,
            )
            last_point = end
            continue

        linearized = ops.linearize(i, last_point)
        p_seg_start = last_point
        for j in range(linearized.len()):
            p_seg_end = linearized.endpoint(j)
            if p_seg_end is None:
                continue
            kept_segments = clip_line_segment_with_polygons(
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

        last_point = end

    ops.replace_with(new_ops)
    if new_ops.len() > 0:
        for j in range(new_ops.len() - 1, -1, -1):
            if new_ops.command_type(j) == CommandType.MOVE_TO:
                ops.last_move_to = new_ops.endpoint(j)
                break
    return ops


def _get_machine_state(ops: Ops, idx: int) -> Optional[MachineState]:
    try:
        return ops.preloaded_state(idx)
    except ValueError:
        pass

    power = 0.0
    air_assist = False
    cut_speed = None
    travel_speed = None
    active_laser_uid = None
    frequency = None
    pulse_width = None
    found_any = False

    for j in range(idx - 1, -1, -1):
        if ops.category(j) != CommandCategory.STATE:
            continue
        found_any = True
        ct = ops.command_type(j)
        if ct == CommandType.SET_POWER:
            power = ops.power(j)
        elif ct == CommandType.SET_CUT_SPEED:
            cut_speed = int(ops.speed(j))
        elif ct == CommandType.SET_TRAVEL_SPEED:
            travel_speed = int(ops.speed(j))
        elif ct == CommandType.ENABLE_AIR_ASSIST:
            air_assist = True
        elif ct == CommandType.DISABLE_AIR_ASSIST:
            air_assist = False
        elif ct == CommandType.SET_LASER:
            active_laser_uid = ops.laser_uid(j)
        elif ct == CommandType.SET_FREQUENCY:
            frequency = ops.frequency(j)
        elif ct == CommandType.SET_PULSE_WIDTH:
            pulse_width = ops.pulse_width(j)

    if found_any:
        return MachineState(
            power=power,
            air_assist=air_assist,
            cut_speed=cut_speed,
            travel_speed=travel_speed,
            active_laser_uid=active_laser_uid,
            frequency=frequency,
            pulse_width=pulse_width,
        )
    return None


def _clip_scanline(
    new_ops: Ops,
    ops: Ops,
    idx: int,
    last_point: Point3D,
    pen_pos: Optional[Point3D],
    valid_regions: List[Polygon],
) -> Optional[Point3D]:
    end = ops.endpoint(idx)
    power_values = ops.scanline_data(idx)
    kept_segments = clip_line_segment_with_polygons(
        last_point, end, valid_regions
    )
    num_values = len(power_values)
    p_start_orig = np.array(last_point)
    p_end_orig = np.array(end)
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
        new_power = bytes(power_values[idx_start:idx_end])

        if new_power:
            if pen_pos is None or math.dist(pen_pos, new_start) > 1e-6:
                new_ops.move_to(*new_start)
            new_ops.scan_to(
                new_end[0],
                new_end[1],
                new_end[2],
                power_values=bytearray(new_power),
            )
            pen_pos = new_end
    return pen_pos


def _clip_and_refit_arc(
    new_ops: Ops,
    ops: Ops,
    idx: int,
    last_point: Point3D,
    pen_pos: Optional[Point3D],
    valid_regions: List[Polygon],
    tolerance: float,
) -> Optional[Point3D]:
    arc_state = _get_machine_state(ops, idx)
    linearized = ops.linearize(idx, last_point)
    kept_pairs = []
    p_seg_start = last_point
    for j in range(linearized.len()):
        p_seg_end = linearized.endpoint(j)
        if p_seg_end is None:
            continue
        segs = clip_line_segment_with_polygons(
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
        primitives = fit_points_with_primitives(chain, tolerance)
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
                new_ops.line_to(*end)
            elif ct == CMD_TYPE_ARC:
                co_i = prim_row[COL_I]
                co_j = prim_row[COL_J]
                cw = bool(prim_row[COL_CW])
                new_ops.arc_to(end[0], end[1], co_i, co_j, cw, end[2])
            elif ct == CMD_TYPE_BEZIER:
                c1 = (prim_row[COL_C1X], prim_row[COL_C1Y], end[2])
                c2 = (prim_row[COL_C2X], prim_row[COL_C2Y], end[2])
                new_ops.bezier_to(c1, c2, end)
            else:
                continue
            if arc_state is not None:
                new_ops.set_state_at(new_ops.len() - 1, arc_state)
        pen_pos = chain[-1]

    return pen_pos


def _clip_and_refit_bezier(
    new_ops: Ops,
    ops: Ops,
    idx: int,
    last_point: Point3D,
    pen_pos: Optional[Point3D],
    valid_regions: List[Polygon],
    tolerance: float,
) -> Optional[Point3D]:
    bezier_state = _get_machine_state(ops, idx)
    linearized = ops.linearize(idx, last_point)
    kept_pairs = []
    p_seg_start = last_point
    for j in range(linearized.len()):
        p_seg_end = linearized.endpoint(j)
        if p_seg_end is None:
            continue
        segs = clip_line_segment_with_polygons(
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
        primitives = fit_points_with_primitives(chain, tolerance)
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
                new_ops.line_to(*end)
            elif ct == CMD_TYPE_ARC:
                co_i = prim_row[COL_I]
                co_j = prim_row[COL_J]
                cw = bool(prim_row[COL_CW])
                new_ops.arc_to(end[0], end[1], co_i, co_j, cw, end[2])
            elif ct == CMD_TYPE_BEZIER:
                c1 = (prim_row[COL_C1X], prim_row[COL_C1Y], end[2])
                c2 = (prim_row[COL_C2X], prim_row[COL_C2Y], end[2])
                new_ops.bezier_to(c1, c2, end)
            else:
                continue
            if bezier_state is not None:
                new_ops.set_state_at(new_ops.len() - 1, bezier_state)
        pen_pos = chain[-1]

    return pen_pos
