"""
Rotary coordinate conversion helpers for the scene compiler.

Handles the degrees <-> mu (machine units) conversions needed
when rendering rotary axis data onto a cylinder surface.
"""

import math
from typing import Optional, Tuple

from ...core.ops import Ops, CommandType, CommandCategory
from ...core.ops.axis import Axis
from ...machine.kinematic_math import KinematicMath


def degrees_to_mu(
    degrees: float,
    diameter: float,
    reverse: bool,
) -> float:
    if diameter <= 0:
        return degrees
    sign = -1.0 if reverse else 1.0
    return degrees * math.pi * diameter / 360.0 * sign


def _find_degrees_from_axes(
    extra_axes: Optional[dict],
) -> Optional[float]:
    if extra_axes is None:
        return None
    for ax in (Axis.A, Axis.B, Axis.C, Axis.U, Axis.Y):
        val = extra_axes.get(ax)
        if val is not None:
            return val
    return None


def find_degrees(ops: Ops, idx: int) -> Optional[float]:
    return _find_degrees_from_axes(ops.extra_axes(idx))


def visual_end(ops: Ops, idx: int) -> Tuple[float, float, float]:
    end = ops.endpoint(idx)
    degrees = find_degrees(ops, idx)
    if degrees is not None:
        return (end[0], degrees, end[2])
    return end


def reconstruct_mu_pos(
    ops: Ops,
    idx: int,
    diameter: float,
    reverse: bool,
) -> Tuple[float, float, float]:
    end = ops.endpoint(idx)
    degrees = find_degrees(ops, idx)
    if degrees is None:
        return end
    mu_val = degrees_to_mu(degrees, diameter, reverse=reverse)
    return (end[0], mu_val, end[2])


def reconstruct_mu_arc(
    ops: Ops,
    idx: int,
    diameter: float,
    reverse: bool,
) -> Tuple[Tuple[float, float, float], float, float, bool]:
    """Reconstructs arc parameters in mu coordinates.

    Returns (end, i, j, clockwise) for building an arc_row.
    """
    end = ops.endpoint(idx)
    i_val, j_val, cw = ops.arc_params(idx)
    degrees = find_degrees(ops, idx)
    if degrees is None:
        return end, i_val, j_val, cw
    scale = degrees_to_mu(1.0, diameter, reverse=reverse)
    mu_end = (end[0], degrees * scale, end[2])
    mu_j = j_val * scale
    return mu_end, i_val, mu_j, cw


def reconstruct_mu_bezier(
    ops: Ops,
    idx: int,
    diameter: float,
    reverse: bool,
) -> Tuple[
    Tuple[float, float, float],
    Tuple[float, float, float],
    Tuple[float, float, float],
]:
    """Reconstructs bezier parameters in mu coordinates.

    Returns (end, control1, control2).
    """
    end = ops.endpoint(idx)
    c1, c2 = ops.bezier_params(idx)
    degrees = find_degrees(ops, idx)
    if degrees is None:
        return end, c1, c2
    scale = degrees_to_mu(1.0, diameter, reverse=reverse)
    mu_end = (end[0], degrees * scale, end[2])
    mu_c1 = (c1[0], c1[1] * scale, c1[2])
    mu_c2 = (c2[0], c2[1] * scale, c2[2])
    return mu_end, mu_c1, mu_c2


def mu_to_visual(
    pos: Tuple[float, float, float],
    diameter: float,
    reverse: bool,
) -> Tuple[float, float, float]:
    degrees = KinematicMath.mu_to_degrees(pos[1], diameter, reverse=reverse)
    result = list(pos)
    result[1] = degrees
    return (result[0], result[1], result[2])


def bake_visual_positions(
    ops: Ops,
) -> Ops:
    baked = Ops()
    for i in range(ops.len()):
        ct = ops.command_type(i)
        cat = ops.category(i)

        if cat != CommandCategory.MOVING:
            baked.copy_command_from(ops, i)
            continue

        end = ops.endpoint(i)
        ea = ops.extra_axes(i)
        degrees = _find_degrees_from_axes(ea)
        if degrees is not None:
            new_end = (end[0], degrees, end[2])
            if ct == CommandType.SCAN_LINE:
                baked.scan_to(
                    new_end[0],
                    new_end[1],
                    new_end[2],
                    power_values=bytearray(ops.scanline_data(i)),
                    extra=ea if ea else None,
                )
            elif ct == CommandType.ARC_TO:
                i_val, j_val, cw = ops.arc_params(i)
                baked.arc_to(
                    new_end[0],
                    new_end[1],
                    i_val,
                    j_val,
                    cw,
                    new_end[2],
                    extra=ea if ea else None,
                )
            elif ct == CommandType.BEZIER_TO:
                c1, c2 = ops.bezier_params(i)
                baked.bezier_to(c1, c2, new_end, extra=ea if ea else None)
            elif ct == CommandType.MOVE_TO:
                baked.move_to(
                    new_end[0],
                    new_end[1],
                    new_end[2],
                    extra=ea if ea else None,
                )
            else:
                baked.copy_command_from(ops, i)
        else:
            baked.copy_command_from(ops, i)
    return baked
