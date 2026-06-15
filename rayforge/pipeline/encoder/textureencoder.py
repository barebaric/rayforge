from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import numpy as np
from raygeo.image import rasterize_scanlines

from .base import OpsEncoder

if TYPE_CHECKING:
    from raygeo.ops import Ops


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

        return rasterize_scanlines(
            ops,
            width_px,
            height_px,
            px_per_mm,
            origin_mm=(0.0, 0.0),
        )
