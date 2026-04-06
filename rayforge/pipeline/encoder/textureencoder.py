from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING, Tuple

from .base import OpsEncoder
from .scanline_rasterizer import rasterize_scanlines

if TYPE_CHECKING:
    from ...core.ops import Ops


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

        buffer = np.zeros((height_px, width_px), dtype=np.uint8)
        rasterize_scanlines(
            ops,
            buffer,
            width_px,
            height_px,
            origin_mm=(0.0, 0.0),
            px_per_mm=px_per_mm,
        )
        return buffer
