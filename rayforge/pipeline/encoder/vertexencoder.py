from __future__ import annotations
import math
from typing import List, Optional
import numpy as np
from ...core.geo import Point3D
from ...core.geo.arc import linearize_arc
from ...core.geo.bezier import linearize_bezier_segment
from ...core.ops import Ops
from ...core.ops.axis import Axis
from ...machine.kinematic_math import KinematicMath
from ...core.ops.commands import (
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    BezierToCommand,
    SetPowerCommand,
    ScanLinePowerCommand,
)
from .base import OpsEncoder
from ..artifact.base import VertexData


def transform_to_cylinder(
    verts: np.ndarray,
    diameter: float,
    colors: Optional[np.ndarray] = None,
    source_axis: Axis = Axis.Y,
) -> tuple:
    """
    Transform flat vertices to cylindrical coordinates.

    The cylinder axis is the linear axis that is neither source_axis nor Z.
    source_axis maps to the rotation angle. Z adjusts the radius.

    When source_axis == Axis.Y (default): X is the cylinder axis,
    Y -> theta — identical to the original behavior.
    When source_axis == Axis.X: Y is the cylinder axis,
    X -> theta.

    Line segments are subdivided as needed to follow the cylinder
    surface instead of cutting through the interior.

    Args:
        verts: Array of shape (N, 3) with X, Y, Z coordinates.
               Vertices are in pairs (line segments for GL_LINES).
        diameter: Cylinder diameter in mm.
        colors: Optional array of shape (N, 4) with per-vertex RGBA
                colors.
        source_axis: Which linear axis maps to the rotation angle
                     (Axis.Y or Axis.X). Default Axis.Y.

    Returns:
        Tuple of (transformed_vertices, expanded_colors). The expanded
        colors match the (possibly larger) vertex count after
        subdivision. If colors is None, the second element is also
        None.
    """
    if verts.size == 0 or diameter <= 0:
        return verts, colors

    source_idx = 0 if source_axis == Axis.X else 1
    cyl_idx = 1 - source_idx

    radius = diameter / 2.0
    max_angle_per_segment = math.radians(15)

    num_pairs = verts.shape[0] // 2
    if num_pairs == 0:
        return (
            np.empty((0, 3), dtype=np.float32),
            (
                np.empty((0, colors.shape[1]), dtype=np.float32)
                if colors is not None
                else None
            ),
        )

    p1 = verts[0::2][:num_pairs]
    p2 = verts[1::2][:num_pairs]

    cyl1 = p1[:, cyl_idx].astype(np.float64)
    src1 = p1[:, source_idx].astype(np.float64)
    z1 = p1[:, 2].astype(np.float64)
    cyl2 = p2[:, cyl_idx].astype(np.float64)
    src2 = p2[:, source_idx].astype(np.float64)
    z2 = p2[:, 2].astype(np.float64)

    theta1 = np.radians(KinematicMath.mu_to_degrees(src1, diameter))
    theta2 = np.radians(KinematicMath.mu_to_degrees(src2, diameter))

    delta_theta = (theta2 - theta1 + np.pi) % (2.0 * np.pi) - np.pi
    abs_delta = np.abs(delta_theta)

    num_subs = np.maximum(
        1, np.ceil(abs_delta / max_angle_per_segment).astype(np.int32)
    )

    total_segments = int(num_subs.sum())

    pair_indices = np.repeat(np.arange(num_pairs), num_subs)
    cum_subs = np.empty(num_pairs + 1, dtype=np.int32)
    cum_subs[0] = 0
    np.cumsum(num_subs, out=cum_subs[1:])
    local_idx = (
        np.arange(total_segments, dtype=np.int32) - cum_subs[pair_indices]
    )

    subs_f64 = num_subs[pair_indices].astype(np.float64)
    prev_t = local_idx.astype(np.float64) / subs_f64
    curr_t = (local_idx + 1).astype(np.float64) / subs_f64

    d_cyl = cyl2[pair_indices] - cyl1[pair_indices]
    d_src = src2[pair_indices] - src1[pair_indices]
    d_z = z2[pair_indices] - z1[pair_indices]
    p_cyl = cyl1[pair_indices]
    p_src = src1[pair_indices]
    p_z = z1[pair_indices]

    prev_cyl = p_cyl + prev_t * d_cyl
    prev_src = p_src + prev_t * d_src
    prev_z = p_z + prev_t * d_z
    curr_cyl = p_cyl + curr_t * d_cyl
    curr_src = p_src + curr_t * d_src
    curr_z = p_z + curr_t * d_z

    prev_eff_r = radius + prev_z
    curr_eff_r = radius + curr_z

    theta_prev = np.radians(KinematicMath.mu_to_degrees(prev_src, diameter))
    theta_curr = np.radians(KinematicMath.mu_to_degrees(curr_src, diameter))

    result_verts = np.empty((total_segments * 2, 3), dtype=np.float32)
    result_verts[0::2, cyl_idx] = prev_cyl.astype(np.float32)
    result_verts[0::2, source_idx] = (prev_eff_r * np.sin(theta_prev)).astype(
        np.float32
    )
    result_verts[0::2, 2] = (prev_eff_r * np.cos(theta_prev)).astype(
        np.float32
    )
    result_verts[1::2, cyl_idx] = curr_cyl.astype(np.float32)
    result_verts[1::2, source_idx] = (curr_eff_r * np.sin(theta_curr)).astype(
        np.float32
    )
    result_verts[1::2, 2] = (curr_eff_r * np.cos(theta_curr)).astype(
        np.float32
    )

    if colors is not None:
        c0 = colors[0::2][:num_pairs]
        c1 = colors[1::2][:num_pairs]
        result_colors = np.empty(
            (total_segments * 2, colors.shape[1]), dtype=np.float32
        )
        result_colors[0::2] = c0[pair_indices]
        result_colors[1::2] = c1[pair_indices]
    else:
        result_colors = None

    return result_verts, result_colors


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

                case BezierToCommand():
                    start_pos = current_pos
                    polyline = linearize_bezier_segment(
                        start_pos, cmd.control1, cmd.control2, cmd.end
                    )
                    if current_power > 0.0:
                        power_byte = min(255, int(current_power * 255.0))
                        color = self._grayscale_lut[power_byte]
                        for i in range(len(polyline) - 1):
                            powered_v.extend(polyline[i])
                            powered_v.extend(polyline[i + 1])
                            powered_c.extend(color)
                            powered_c.extend(color)
                    else:
                        for i in range(len(polyline) - 1):
                            zero_power_v.extend(polyline[i])
                            zero_power_v.extend(polyline[i + 1])
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
