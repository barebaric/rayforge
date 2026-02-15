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

                # Convert start/end from mm (Y-up) to pixel (Y-down) space
                start_px_vec = np.array(
                    [
                        start_mm[0] * px_per_mm_x,
                        height_px - (start_mm[1] * px_per_mm_y),
                    ]
                )
                end_px_vec = np.array(
                    [
                        end_mm[0] * px_per_mm_x,
                        height_px - (end_mm[1] * px_per_mm_y),
                    ]
                )

                # Generate integer pixel coordinates for each step along the
                # line
                x_coords = np.linspace(
                    start_px_vec[0], end_px_vec[0], num_steps, dtype=np.int32
                )
                y_coords = np.linspace(
                    start_px_vec[1], end_px_vec[1], num_steps, dtype=np.int32
                )

                # Create a mask to filter out any coordinates that fall outside
                # the buffer dimensions due to rounding or other artifacts.
                valid_mask = (
                    (x_coords >= 0)
                    & (x_coords < width_px)
                    & (y_coords >= 0)
                    & (y_coords < height_px)
                )

                valid_x = x_coords[valid_mask]
                valid_y = y_coords[valid_mask]
                # Ensure power_values is a numpy array for boolean indexing
                valid_power = np.frombuffer(power_values, dtype=np.uint8)[
                    valid_mask
                ]

                # Use advanced indexing to efficiently set pixel values
                # Additive approach: take maximum power at each pixel
                buffer[valid_y, valid_x] = np.maximum(
                    buffer[valid_y, valid_x], valid_power
                )

                current_pos_mm = end_mm
        return buffer
