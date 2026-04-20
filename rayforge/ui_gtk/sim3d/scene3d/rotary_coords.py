"""
Rotary coordinate conversion helpers for the scene compiler.

Handles the degrees <-> mu (machine units) conversions needed
when rendering rotary axis data onto a cylinder surface.
"""

import math
from typing import Optional, Tuple

from ....core.ops import Ops
from ....core.ops.axis import Axis
from ....core.ops.commands import (
    ArcToCommand,
    BezierToCommand,
    MovingCommand,
    ScanLinePowerCommand,
)
from ....machine.kinematic_math import KinematicMath


def degrees_to_mu(
    degrees: float,
    diameter: float,
    reverse: bool,
) -> float:
    if diameter <= 0:
        return degrees
    sign = -1.0 if reverse else 1.0
    return degrees * math.pi * diameter / 360.0 * sign


def find_degrees(cmd: MovingCommand) -> Optional[float]:
    for ax in (Axis.A, Axis.B, Axis.C, Axis.U, Axis.Y):
        val = cmd.extra_axes.get(ax)
        if val is not None:
            return val
    return None


def visual_end(
    cmd: MovingCommand,
) -> Tuple[float, float, float]:
    degrees = find_degrees(cmd)
    if degrees is not None:
        pos = list(cmd.end)
        pos[1] = degrees
        return (pos[0], pos[1], pos[2])
    return cmd.end


def reconstruct_mu_pos(
    cmd: MovingCommand,
    diameter: float,
    reverse: bool,
) -> Tuple[float, float, float]:
    degrees = find_degrees(cmd)
    if degrees is None:
        return cmd.end
    mu_val = degrees_to_mu(degrees, diameter, reverse=reverse)
    pos = list(cmd.end)
    pos[1] = mu_val
    return (pos[0], pos[1], pos[2])


def reconstruct_mu_arc(
    cmd: ArcToCommand,
    diameter: float,
    reverse: bool,
) -> ArcToCommand:
    degrees = find_degrees(cmd)
    if degrees is None:
        return cmd
    scale = degrees_to_mu(1.0, diameter, reverse=reverse)

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


def reconstruct_mu_bezier(
    cmd: BezierToCommand,
    diameter: float,
    reverse: bool,
) -> BezierToCommand:
    degrees = find_degrees(cmd)
    if degrees is None:
        return cmd
    scale = degrees_to_mu(1.0, diameter, reverse=reverse)

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


def mu_to_visual(
    pos: Tuple[float, float, float],
    diameter: float,
    reverse: bool,
) -> Tuple[float, float, float]:
    degrees = KinematicMath.mu_to_degrees(
        pos[1], diameter, reverse=reverse
    )
    result = list(pos)
    result[1] = degrees
    return (result[0], result[1], result[2])


def bake_visual_positions(
    ops: Ops,
) -> Ops:
    baked = Ops()
    for cmd in ops.commands:
        if isinstance(cmd, MovingCommand):
            degrees = find_degrees(cmd)
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
