from __future__ import annotations

from typing import Tuple, Union

import numpy as np
from raygeo.ops.types import CommandType

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


def _rasterize_horizontal(
    buffer: np.ndarray,
    iy: int,
    height_px: int,
    width_px: int,
    x: float,
    dx: float,
    pwr,
    num_steps: int,
) -> bool:
    """Fast path for horizontal scanlines (dy == 0).

    All pixels share the same row, so we fill contiguous x-ranges
    directly without per-pixel bresenham overhead.
    """
    has_content = False
    if not (0 <= iy < height_px):
        return False
    row = buffer[iy]
    for i in range(num_steps):
        p = pwr[i]
        if p > 0:
            x0 = round(x)
            x1 = round(x + dx)
            if x0 > x1:
                x0, x1 = x1, x0
            if x0 < 0:
                x0 = 0
            if x1 >= width_px:
                x1 = width_px - 1
            for xi in range(x0, x1 + 1):
                if p > row[xi]:
                    row[xi] = p
            has_content = True
        x += dx
    return has_content


def _rasterize_diagonal(
    buffer: np.ndarray,
    width_px: int,
    height_px: int,
    x: float,
    y: float,
    dx: float,
    dy: float,
    pwr,
    num_steps: int,
) -> bool:
    """General path for non-horizontal scanlines.

    Uses bresenham for multi-pixel segments and direct writes for
    single-pixel segments.
    """
    has_content = False
    for i in range(num_steps):
        p = pwr[i]
        if p > 0:
            psx = round(x)
            psy = round(y)
            pex = round(x + dx)
            pey = round(y + dy)
            if psx == pex and psy == pey:
                if 0 <= psx < width_px and 0 <= psy < height_px:
                    if p > buffer[psy, psx]:
                        buffer[psy, psx] = p
                        has_content = True
            else:
                bresenham_line(buffer, psx, psy, pex, pey, p)
                has_content = True
        x += dx
        y += dy
    return has_content


def rasterize_scanlines(
    ops,
    buffer: np.ndarray,
    width_px: int,
    height_px: int,
    origin_mm: Tuple[float, float] = (0.0, 0.0),
    px_per_mm: Union[float, Tuple[float, float]] = 50.0,
) -> bool:
    """Rasterize ScanLinePowerCommands into a 2D power-map buffer.

    Iterates all scanline commands in *ops*, converts their mm
    coordinates to pixel space using *px_per_mm*, and writes each
    non-zero power value into *buffer* (uint8 2-D array).  Where
    multiple scanlines overlap the same pixel the maximum power
    wins.

    Two fast paths avoid unnecessary per-step overhead:

    * **Horizontal** (dy == 0): fills a contiguous x-range per step
      directly into the target row, skipping bresenham entirely.
    * **Single-pixel**: when a step's start and end round to the
      same pixel, writes directly instead of calling bresenham.

    Zero-power steps are skipped early to avoid ``round()`` cost on
    the majority of pixels (typically ~88 % are zero in a raster
    engraving pass).

    Coordinates are accumulated additively (``x += dx``) rather than
    recomputed from scratch each step, eliminating per-step
    multiplication and division.

    Returns True if any non-zero power was written.
    """
    if isinstance(px_per_mm, (list, tuple)):
        px_per_mm_x, px_per_mm_y = px_per_mm
    else:
        px_per_mm_x = px_per_mm_y = px_per_mm
    x_origin, y_origin = origin_mm
    current_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    has_content = False

    for i in range(ops.len()):
        ct = ops.command_type(i)
        if ct == CommandType.MOVE_TO:
            end = ops.endpoint(i)
            if end is not None:
                current_pos = end
            continue

        if ct != CommandType.SCAN_LINE:
            continue

        end = ops.endpoint(i)
        if end is None:
            continue

        end_mm = end
        power_values = ops.scanline_data(i)
        num_steps = len(power_values)
        if num_steps == 0:
            current_pos = end_mm
            continue

        # Convert mm → pixel coords (y is inverted: bottom-up mm
        # → top-down pixels).
        sx = (current_pos[0] - x_origin) * px_per_mm_x
        sy = height_px - (current_pos[1] - y_origin) * px_per_mm_y
        ex = (end_mm[0] - x_origin) * px_per_mm_x
        ey = height_px - (end_mm[1] - y_origin) * px_per_mm_y

        # Per-step delta in pixel space (used for additive
        # accumulation instead of per-step multiplication).
        dx = (ex - sx) / num_steps
        dy = (ey - sy) / num_steps

        if dy == 0.0 and dx != 0.0:
            has_content |= _rasterize_horizontal(
                buffer,
                round(sy),
                height_px,
                width_px,
                sx,
                dx,
                power_values,
                num_steps,
            )
        elif dx != 0.0 or dy != 0.0:
            has_content |= _rasterize_diagonal(
                buffer,
                width_px,
                height_px,
                sx,
                sy,
                dx,
                dy,
                power_values,
                num_steps,
            )

        current_pos = end_mm

    return has_content
