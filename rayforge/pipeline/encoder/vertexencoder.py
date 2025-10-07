from __future__ import annotations
from typing import List, Tuple
import numpy as np
from ...core.ops import Ops
from ...core.ops.commands import (
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
    ScanLinePowerCommand,
)
from ...core.geo.linearize import linearize_arc
from .base import OpsEncoder
from ..artifact.base import VertexData


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

    def encode(self, ops: Ops) -> VertexData:
        """
        Converts Ops into vertex arrays for different path types.

        Args:
            ops: The Ops object to encode

        Returns:
            A VertexData object containing the computed vertex arrays.
        """
        powered_v: List[float] = []
        powered_c: List[float] = []
        travel_v: List[float] = []
        zero_power_v: List[float] = []

        # Track current state
        current_power = 0.0
        current_pos = (0.0, 0.0, 0.0)

        for cmd in ops.commands:
            if isinstance(cmd, SetPowerCommand):
                current_power = cmd.power
                continue

            # Use a match statement for clarity and direct handling
            match cmd:
                case MoveToCommand():
                    start_pos, end_pos = current_pos, cmd.end
                    travel_v.extend(start_pos)
                    travel_v.extend(end_pos)
                    current_pos = end_pos

                case LineToCommand():
                    start_pos, end_pos = current_pos, cmd.end
                    if current_power > 0.0:
                        power_byte = min(255, int(current_power * 255.0))
                        color = self._grayscale_lut[power_byte]
                        powered_v.extend(start_pos)
                        powered_v.extend(end_pos)
                        powered_c.extend(color)
                        powered_c.extend(color)
                    else:
                        zero_power_v.extend(start_pos)
                        zero_power_v.extend(end_pos)
                    current_pos = end_pos

                case ArcToCommand():
                    start_pos = current_pos
                    segments = linearize_arc(cmd, start_pos)
                    if current_power > 0.0:
                        power_byte = min(255, int(current_power * 255.0))
                        color = self._grayscale_lut[power_byte]
                        for seg_start, seg_end in segments:
                            powered_v.extend(seg_start)
                            powered_v.extend(seg_end)
                            powered_c.extend(color)
                            powered_c.extend(color)
                    else:
                        for seg_start, seg_end in segments:
                            zero_power_v.extend(seg_start)
                            zero_power_v.extend(seg_end)
                    current_pos = cmd.end

                case ScanLinePowerCommand():
                    if cmd.end is not None:
                        self._handle_scanline(
                            cmd,
                            current_pos,
                            zero_power_v,
                        )
                        current_pos = cmd.end

        # Convert lists to numpy arrays and return a VertexData object
        return VertexData(
            powered_vertices=np.array(powered_v, dtype=np.float32).reshape(
                -1, 3
            ),
            powered_colors=np.array(powered_c, dtype=np.float32).reshape(
                -1, 4
            ),
            travel_vertices=np.array(travel_v, dtype=np.float32).reshape(
                -1, 3
            ),
            zero_power_vertices=np.array(
                zero_power_v, dtype=np.float32
            ).reshape(-1, 3),
        )

    def _handle_scanline(
        self,
        cmd: ScanLinePowerCommand,
        start_pos: Tuple[float, float, float],
        zero_power_v: List[float],
    ):
        """
        Processes a ScanLine, adding ONLY zero-power segments to vertices.
        """
        if cmd.end is None:
            return

        p_start_vec = np.array(start_pos, dtype=np.float32)
        p_end_vec = np.array(cmd.end, dtype=np.float32)
        line_vec = p_end_vec - p_start_vec
        num_steps = len(cmd.power_values)
        if num_steps == 0:
            return

        chunk_start_idx = 0
        is_zero_chunk = cmd.power_values[0] == 0

        for i in range(1, num_steps):
            is_current_zero = cmd.power_values[i] == 0
            if is_current_zero != is_zero_chunk:
                # End of a chunk. Process it.
                self._process_scanline_chunk(
                    p_start_vec,
                    line_vec,
                    num_steps,
                    chunk_start_idx,
                    i,
                    is_zero_chunk,
                    zero_power_v,
                )
                chunk_start_idx = i
                is_zero_chunk = is_current_zero

        # Process the final chunk
        self._process_scanline_chunk(
            p_start_vec,
            line_vec,
            num_steps,
            chunk_start_idx,
            num_steps,
            is_zero_chunk,
            zero_power_v,
        )

    def _process_scanline_chunk(
        self,
        p_start_vec,
        line_vec,
        total_steps,
        start_idx,
        end_idx,
        is_zero_chunk,
        zero_power_v,
    ):
        """Processes a single chunk of a scanline."""
        if start_idx >= end_idx:
            return

        if is_zero_chunk:
            t_start = start_idx / total_steps
            t_end = end_idx / total_steps
            chunk_start_pt = p_start_vec + t_start * line_vec
            chunk_end_pt = p_start_vec + t_end * line_vec
            zero_power_v.extend(chunk_start_pt)
            zero_power_v.extend(chunk_end_pt)
        else:
            # For powered chunks, we do nothing. The visualization for these
            # comes from the RasterArtifactRenderer's textured quad.
            pass
