from __future__ import annotations
import logging
import numpy as np
from typing import TYPE_CHECKING, Tuple

from .base import OpsEncoder
from ...core.ops import MoveToCommand, ScanLinePowerCommand

if TYPE_CHECKING:
    from ...core.ops import Ops

logger = logging.getLogger(__name__)


class TextureEncoder(OpsEncoder):
    """
    Encodes Ops into a 2D numpy array representing a texture of power levels.
    """

    def encode(
        self,
        ops: "Ops",
        width_px: int,
        height_px: int,
        px_per_mm: Tuple[float, float],
    ) -> np.ndarray:
        """
        Converts ScanLinePowerCommands within an Ops object into a 2D power
        map (texture).

        Args:
            ops: The Ops object to encode.
            width_px: The width of the target texture in pixels.
            height_px: The height of the target texture in pixels.
            px_per_mm: The (x, y) resolution in pixels per millimeter.

        Returns:
            A 2D numpy array (uint8) representing the power texture.
        """
        if width_px <= 0 or height_px <= 0:
            return np.array([], dtype=np.uint8)

        px_per_mm_x, px_per_mm_y = px_per_mm
        buffer = np.zeros((height_px, width_px), dtype=np.uint8)
        current_pos_mm = (0.0, 0.0, 0.0)

        for cmd in ops:
            if isinstance(cmd, MoveToCommand):
                if cmd.end:
                    current_pos_mm = cmd.end
            elif isinstance(cmd, ScanLinePowerCommand):
                start_mm = current_pos_mm
                if cmd.end is None:
                    continue
                end_mm = cmd.end
                power_values = cmd.power_values
                num_steps = len(power_values)

                if num_steps == 0:
                    current_pos_mm = end_mm
                    continue

                start_px = (
                    int(round(start_mm[0] * px_per_mm_x)),
                    int(round(height_px - (start_mm[1] * px_per_mm_y))),
                )
                end_px = (
                    int(round(end_mm[0] * px_per_mm_x)),
                    int(round(height_px - (end_mm[1] * px_per_mm_y))),
                )

                power_array = np.frombuffer(power_values, dtype=np.uint8)

                for i in range(num_steps):
                    t_start = i / num_steps
                    t_end = (i + 1) / num_steps

                    seg_start_x = int(
                        round(
                            start_px[0] + t_start * (end_px[0] - start_px[0])
                        )
                    )
                    seg_start_y = int(
                        round(
                            start_px[1] + t_start * (end_px[1] - start_px[1])
                        )
                    )
                    seg_end_x = int(
                        round(start_px[0] + t_end * (end_px[0] - start_px[0]))
                    )
                    seg_end_y = int(
                        round(start_px[1] + t_end * (end_px[1] - start_px[1]))
                    )

                    power = power_array[i]
                    if power == 0:
                        continue

                    self._draw_line(
                        buffer,
                        seg_start_x,
                        seg_start_y,
                        seg_end_x,
                        seg_end_y,
                        power,
                    )

                current_pos_mm = end_mm
        return buffer

    def _draw_line(
        self,
        buffer: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        power: int,
    ):
        """
        Draw a line segment using Bresenham's algorithm.

        Args:
            buffer: The texture buffer to draw into.
            x0, y0: Start pixel coordinates.
            x1, y1: End pixel coordinates.
            power: The power value to set for pixels along the line.
        """
        height_px, width_px = buffer.shape

        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            if 0 <= x < width_px and 0 <= y < height_px:
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
