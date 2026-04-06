from __future__ import annotations

import numpy as np
from typing import Tuple, Union

from ...core.ops import MoveToCommand, ScanLinePowerCommand

MAX_TEXTURE_DIMENSION = 8192


def bresenham_line(
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


def rasterize_scanlines(
    ops,
    buffer: np.ndarray,
    width_px: int,
    height_px: int,
    origin_mm: Tuple[float, float] = (0.0, 0.0),
    px_per_mm: Union[float, Tuple[float, float]] = 50.0,
) -> bool:
    if isinstance(px_per_mm, (list, tuple)):
        px_per_mm_x, px_per_mm_y = px_per_mm
    else:
        px_per_mm_x = px_per_mm_y = px_per_mm
    x_origin, y_origin = origin_mm
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

            sx = (current_pos[0] - x_origin) * px_per_mm_x
            sy = height_px - (current_pos[1] - y_origin) * px_per_mm_y
            ex = (end_mm[0] - x_origin) * px_per_mm_x
            ey = height_px - (end_mm[1] - y_origin) * px_per_mm_y

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
                bresenham_line(buffer, psx, psy, pex, pey, int(power))

            current_pos = end_mm

    return buffer.max() > 0
