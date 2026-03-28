from __future__ import annotations
import math
from typing import List
import numpy as np
from ...core.geo import Point3D
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


def transform_to_cylinder(verts: np.ndarray, diameter: float) -> np.ndarray:
    """
    Transform flat XY vertices to cylindrical coordinates.

    X remains unchanged (along cylinder axis).
    Y maps to rotation angle around cylinder.
    Z is computed from the angle.

    Line segments are subdivided as needed to follow the cylinder surface
    instead of cutting through the interior.

    Args:
        verts: Array of shape (N, 3) with X, Y, Z coordinates.
               Vertices are in pairs (line segments for GL_LINES).
        diameter: Cylinder diameter in mm.

    Returns:
        Transformed vertices array. May contain more vertices than input
        if segments were split.
    """
    if verts.size == 0 or diameter <= 0:
        return verts

    radius = diameter / 2.0
    circumference = diameter * math.pi
    max_angle_per_segment = math.radians(15)

    def cyl_point(x, y):
        theta = (y / circumference) * 2.0 * math.pi
        return [x, radius * math.sin(theta), radius * math.cos(theta)]

    def normalize_angle_diff(delta):
        while delta > math.pi:
            delta -= 2 * math.pi
        while delta < -math.pi:
            delta += 2 * math.pi
        return delta

    result_verts = []

    for i in range(0, verts.shape[0], 2):
        if i + 1 >= verts.shape[0]:
            break

        x1, y1, _ = verts[i]
        x2, y2, _ = verts[i + 1]

        theta1 = (y1 / circumference) * 2.0 * math.pi
        theta2 = (y2 / circumference) * 2.0 * math.pi

        delta_theta = abs(normalize_angle_diff(theta2 - theta1))

        num_subdivisions = max(
            1, int(math.ceil(delta_theta / max_angle_per_segment))
        )

        prev_x, prev_y = x1, y1
        for j in range(1, num_subdivisions + 1):
            t = j / num_subdivisions
            curr_x = x1 + t * (x2 - x1)
            curr_y = y1 + t * (y2 - y1)

            result_verts.append(cyl_point(prev_x, prev_y))
            result_verts.append(cyl_point(curr_x, curr_y))

            prev_x, prev_y = curr_x, curr_y

    return np.array(result_verts, dtype=np.float32)


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

        for cmd in ops.commands:
            if isinstance(cmd, SetPowerCommand):
                current_power = cmd.power
                continue

            # Use a match statement for clarity and direct handling
            match cmd:
                case MoveToCommand():
                    start_pos, end_pos = current_pos, cmd.end
                    # Skip the initial travel move from machine origin to first
                    # cut point - this is positioning, not actual workpiece
                    # travel
                    if not is_initial_position:
                        travel_v.extend(start_pos)
                        travel_v.extend(end_pos)
                    current_pos = end_pos
                    is_initial_position = False

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
                    is_initial_position = False

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
                    is_initial_position = False

                case ScanLinePowerCommand():
                    if cmd.end is not None:
                        self._handle_scanline(
                            cmd,
                            current_pos,
                            zero_power_v,
                        )
                        current_pos = cmd.end
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

    def _handle_scanline(
        self,
        cmd: ScanLinePowerCommand,
        start_pos: Point3D,
        zero_power_v: List[float],
    ):
        """
        Processes a ScanLine, adding ONLY zero-power segments to vertices.
        Powered segments are ignored, as their visualization comes from the
        texture generated by the TextureEncoder.
        """
        if cmd.end is None:
            return

        p_start_vec = np.array(start_pos, dtype=np.float32)
        p_end_vec = np.array(cmd.end, dtype=np.float32)
        line_vec = p_end_vec - p_start_vec
        num_steps = len(cmd.power_values)
        if num_steps == 0:
            return

        is_zero_power = np.frombuffer(cmd.power_values, dtype=np.uint8) == 0

        # Find contiguous chunks of zero-power pixels
        padded = np.concatenate(([False], is_zero_power, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        for start_idx, end_idx in zip(starts, ends):
            # A chunk of zero-power pixels was found. Add it to vertices.
            t_start = start_idx / num_steps
            t_end = end_idx / num_steps
            chunk_start_pt = p_start_vec + t_start * line_vec
            chunk_end_pt = p_start_vec + t_end * line_vec
            zero_power_v.extend(chunk_start_pt)
            zero_power_v.extend(chunk_end_pt)
