from __future__ import annotations

import numpy as np
from raygeo.ops import Ops

from ..artifact.base import VertexData
from .base import OpsEncoder


class VertexEncoder(OpsEncoder):
    """
    Encodes Ops objects into vertex arrays for GPU-friendly rendering.

    This encoder converts machine operations into pre-computed vertex arrays
    with associated color data, making rendering more efficient.
    """

    def __init__(self):
        # Create a standard, non-themed ColorSet for grayscale power mapping
        self._grayscale_lut = self._create_grayscale_lut()

    def _create_grayscale_lut(self) -> np.ndarray:
        """Creates a 256x4 grayscale lookup table for power levels."""
        lut = np.zeros((256, 4), dtype=np.float32)
        # Map power 0-255 to grayscale 0-1 for RGBA
        gray_values = np.arange(256, dtype=np.float32) / 255.0
        lut[:, 0] = gray_values  # R
        lut[:, 1] = gray_values  # G
        lut[:, 2] = gray_values  # B
        lut[:, 3] = 1.0  # A
        return lut

    def encode(
        self,
        ops: Ops,
    ) -> VertexData:
        """
        Converts Ops into vertex arrays for different path types.

        Args:
            ops: The Ops object to encode

        Returns:
            A VertexData object containing the computed vertex arrays.
        """
        (
            powered_vertices,
            powered_colors,
            travel_vertices,
            zero_power_vertices,
        ) = ops.to_vertex_arrays()

        return VertexData(
            powered_vertices=powered_vertices,
            powered_colors=powered_colors,
            travel_vertices=travel_vertices,
            zero_power_vertices=zero_power_vertices,
        )
