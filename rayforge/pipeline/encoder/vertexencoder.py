from __future__ import annotations

from typing import List

import numpy as np
from raygeo.geo.shape.arc import linearize_arc
from raygeo.geo.shape.bezier import linearize_bezier_segment
from raygeo.geo.types import Point3D
from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType

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
        powered_v: List[float] = []
        powered_c: List[float] = []
        travel_v: List[float] = []
        zero_power_v: List[float] = []

        # Track current state
        current_power = 0.0
        current_pos = (0.0, 0.0, 0.0)
        is_initial_position = True

        for i in range(ops.len()):
            ct = ops.command_type(i)

            if ct == CommandType.SET_POWER:
                current_power = ops.power(i)
                continue

            cat = ops.category(i)
            if cat != CommandCategory.MOVING:
                continue

            end = ops.endpoint(i)

            if ct == CommandType.MOVE_TO:
                start_pos, end_pos = current_pos, end
                # Skip the initial travel move from machine origin to first
                # cut point - this is positioning, not actual workpiece
                # travel
                if not is_initial_position:
                    travel_v.extend(start_pos)
                    travel_v.extend(end_pos)
                current_pos = end_pos
                is_initial_position = False

            elif ct == CommandType.LINE_TO:
                start_pos, end_pos = current_pos, end
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
                is_initial_position = False

            elif ct == CommandType.ARC_TO:
                start_pos = current_pos
                i_val, j_val, cw = ops.arc_params(i)
                arc_row = [
                    3,
                    end[0],
                    end[1],
                    end[2],
                    i_val,
                    j_val,
                    1.0 if cw else 0.0,
                    0.0,
                ]
                segments = linearize_arc(arc_row, start_pos)
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
                current_pos = end
                is_initial_position = False

            elif ct == CommandType.BEZIER_TO:
                start_pos = current_pos
                c1, c2 = ops.bezier_params(i)
                polyline = linearize_bezier_segment(start_pos, c1, c2, end)
                if current_power > 0.0:
                    power_byte = min(255, int(current_power * 255.0))
                    color = self._grayscale_lut[power_byte]
                    for j in range(len(polyline) - 1):
                        powered_v.extend(polyline[j])
                        powered_v.extend(polyline[j + 1])
                        powered_c.extend(color)
                        powered_c.extend(color)
                else:
                    for j in range(len(polyline) - 1):
                        zero_power_v.extend(polyline[j])
                        zero_power_v.extend(polyline[j + 1])
                current_pos = end
                is_initial_position = False

            elif ct == CommandType.SCAN_LINE:
                scanline_mv = ops.scanline_data(i)
                self._handle_scanline_indexed(
                    end, scanline_mv, current_pos, zero_power_v
                )
                current_pos = end
                is_initial_position = False

        powered_verts = np.array(powered_v, dtype=np.float32).reshape(-1, 3)
        travel_verts = np.array(travel_v, dtype=np.float32).reshape(-1, 3)
        zero_power_verts = np.array(zero_power_v, dtype=np.float32).reshape(
            -1, 3
        )

        return VertexData(
            powered_vertices=powered_verts,
            powered_colors=np.array(powered_c, dtype=np.float32).reshape(
                -1, 4
            ),
            travel_vertices=travel_verts,
            zero_power_vertices=zero_power_verts,
        )

    def _handle_scanline_indexed(
        self,
        end: Point3D,
        power_mv: memoryview | bytes,
        start_pos: Point3D,
        zero_power_v: List[float],
    ):
        """
        Processes a ScanLine, adding ONLY zero-power segments to vertices.
        Powered segments are ignored, as their visualization comes from the
        texture generated by the TextureEncoder.
        """
        num_steps = len(power_mv)
        if num_steps == 0:
            return

        sx, sy, sz = start_pos[0], start_pos[1], start_pos[2]
        ex, ey, ez = end[0], end[1], end[2]
        dx = ex - sx
        dy = ey - sy
        dz = ez - sz
        inv_n = 1.0 / num_steps
        run_start = -1

        for i in range(num_steps):
            if power_mv[i] == 0:
                if run_start < 0:
                    run_start = i
            else:
                if run_start >= 0:
                    t0 = run_start * inv_n
                    t1 = i * inv_n
                    zero_power_v.append(sx + t0 * dx)
                    zero_power_v.append(sy + t0 * dy)
                    zero_power_v.append(sz + t0 * dz)
                    zero_power_v.append(sx + t1 * dx)
                    zero_power_v.append(sy + t1 * dy)
                    zero_power_v.append(sz + t1 * dz)
                    run_start = -1

        if run_start >= 0:
            t0 = run_start * inv_n
            zero_power_v.append(sx + t0 * dx)
            zero_power_v.append(sy + t0 * dy)
            zero_power_v.append(sz + t0 * dz)
            zero_power_v.append(ex)
            zero_power_v.append(ey)
            zero_power_v.append(ez)
