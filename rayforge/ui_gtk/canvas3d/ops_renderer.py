"""
A renderer for visualizing toolpath operations (Ops) in 3D.
"""

import math
import logging
from typing import List, Tuple
import numpy as np
from OpenGL import GL
from ...core.ops import (
    ArcToCommand,
    Command,
    LineToCommand,
    MoveToCommand,
    ScanLinePowerCommand,
    Ops,
)
from ...shared.util.colors import ColorSet
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class OpsRenderer(BaseRenderer):
    """Renders toolpath operations (cuts and travels) as colored lines."""

    def __init__(self):
        """Initializes the OpsRenderer."""
        super().__init__()
        self.powered_vao: int = 0
        self.travel_vao: int = 0

        self.powered_vbo: int = 0
        self.powered_colors_vbo: int = 0
        self.travel_vbo: int = 0

        self.powered_vertex_count: int = 0
        self.travel_vertex_count: int = 0

    def init_gl(self):
        """
        Initializes OpenGL resources and sets up the VAO states permanently.
        """
        # Create Buffers
        self.powered_vbo = self._create_vbo()
        self.powered_colors_vbo = self._create_vbo()
        self.travel_vbo = self._create_vbo()

        # Configure VAO for Powered Moves (Cuts, Engraves, Zero-Power)
        self.powered_vao = self._create_vao()
        GL.glBindVertexArray(self.powered_vao)
        # Position attribute (location 0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        # Color attribute (location 1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_colors_vbo)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        # Configure VAO for Travel Moves
        self.travel_vao = self._create_vao()
        GL.glBindVertexArray(self.travel_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.travel_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Unbind all
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def clear(self):
        """Clears the renderer's buffers and resets vertex counts."""
        self.update_from_vertex_data(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    def prepare_vertex_data(
        self, ops: Ops, colors: ColorSet
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes an Ops object into numpy arrays of vertices. This method is
        thread-safe and can be run in the background.
        """
        logger.debug(f"OpsRenderer: Preparing {len(ops.commands)} commands.")
        powered_vertices: List[float] = []
        powered_colors: List[float] = []
        travel_vertices: List[float] = []
        zero_power_vertices: List[float] = []
        zero_power_colors: List[float] = []
        last_point: Tuple[float, float, float] = 0.0, 0.0, 0.0
        current_power = 0.0

        cut_lut = colors.get_lut("cut")
        zero_power_rgba = colors.get_rgba("zero_power")

        ops.preload_state()
        for i, command in enumerate(ops.commands):
            if not isinstance(command, Command) or command.is_marker():
                continue

            if hasattr(command, "state") and command.state is not None:
                current_power = command.state.power

            if command.end is None:
                continue

            end_point = tuple(map(float, command.end))
            start_point = last_point

            if isinstance(command, ScanLinePowerCommand):
                logger.debug(
                    f"OpsRenderer: Processing ScanLinePowerCommand at "
                    f"index {i}."
                )
                # The TextureRenderer shows the powered image. Here, we
                # only care about visualizing the zero-power toolpath moves
                # (e.g., overscan) from the ScanLine command.
                num_steps = len(command.power_values)
                if num_steps > 0:
                    p_start_vec = np.array(start_point)
                    line_vec = np.array(end_point) - p_start_vec

                    is_zero = np.array(command.power_values) == 0
                    if np.any(is_zero):
                        padded = np.concatenate(([False], is_zero, [False]))
                        diffs = np.diff(padded.astype(int))
                        starts = np.where(diffs == 1)[0]
                        ends = np.where(diffs == -1)[0]

                        if starts.size > 0:
                            logger.debug(
                                f"OpsRenderer: Found {starts.size} zero-power "
                                f"segments in ScanLine. Adding to "
                                f"zero-power buffer."
                            )
                            for z_idx, (start_idx, end_idx) in enumerate(
                                zip(starts, ends)
                            ):
                                t_start = start_idx / num_steps
                                t_end = end_idx / num_steps
                                chunk_start = p_start_vec + t_start * line_vec
                                chunk_end = p_start_vec + t_end * line_vec
                                zero_power_vertices.extend(chunk_start)
                                zero_power_vertices.extend(chunk_end)
                                zero_power_colors.extend(zero_power_rgba)
                                zero_power_colors.extend(zero_power_rgba)
                                logger.debug(
                                    f"  -> Seg {z_idx}: "
                                    f"from {chunk_start.round(2)} to "
                                    f"{chunk_end.round(2)}"
                                )

                # CRITICAL: The logical tool position moves to the end of the
                # scanline, regardless of what was drawn. This prevents
                # subsequent travel moves from being corrupted.
                last_point = command.end
                continue

            if isinstance(command, MoveToCommand):
                if not np.allclose(start_point, end_point):
                    travel_vertices.extend(start_point)
                    travel_vertices.extend(end_point)
            elif isinstance(command, (LineToCommand, ArcToCommand)):
                is_zero_power = math.isclose(current_power, 0.0)

                if is_zero_power:
                    color = zero_power_rgba
                else:
                    power_idx = min(255, int(current_power * 255.0))
                    color = tuple(cut_lut[power_idx])

                if isinstance(command, LineToCommand):
                    if is_zero_power:
                        zero_power_vertices.extend(start_point)
                        zero_power_vertices.extend(end_point)
                        zero_power_colors.extend(color)
                        zero_power_colors.extend(color)
                    else:
                        powered_vertices.extend(start_point)
                        powered_vertices.extend(end_point)
                        powered_colors.extend(color)
                        powered_colors.extend(color)
                else:  # ArcToCommand
                    arc_verts = self._tessellate_arc(
                        start_point, end_point, command
                    )
                    num_segments = len(arc_verts) // 6
                    if is_zero_power:
                        zero_power_vertices.extend(arc_verts)
                        for _ in range(num_segments * 2):
                            zero_power_colors.extend(color)
                    else:
                        powered_vertices.extend(arc_verts)
                        for _ in range(num_segments * 2):
                            powered_colors.extend(color)

            last_point = command.end

        logger.debug(
            f"OpsRenderer: Prepared vertices. Powered: "
            f"{len(powered_vertices) // 3}, "
            f"Travel: {len(travel_vertices) // 3}, "
            f"Zero-Power Cuts: {len(zero_power_vertices) // 3}"
        )

        return (
            np.array(powered_vertices, dtype=np.float32),
            np.array(powered_colors, dtype=np.float32),
            np.array(travel_vertices, dtype=np.float32),
            np.array(zero_power_vertices, dtype=np.float32),
            np.array(zero_power_colors, dtype=np.float32),
        )

    def update_from_vertex_data(
        self,
        powered_vertices: np.ndarray,
        powered_colors: np.ndarray,
        travel_vertices: np.ndarray,
    ):
        """Receives pre-processed vertex data and uploads it to the GPU."""
        self.powered_vertex_count = powered_vertices.size // 3
        self._load_buffer_data(self.powered_vbo, powered_vertices)
        self._load_buffer_data(self.powered_colors_vbo, powered_colors)
        self.travel_vertex_count = travel_vertices.size // 3
        self._load_buffer_data(self.travel_vbo, travel_vertices)

    def render(
        self,
        shader: Shader,
        mvp_matrix: np.ndarray,
        colors: ColorSet,
        show_travel_moves: bool,
    ) -> None:
        """
        Renders the toolpaths. The vertices are assumed to be in world space.

        Args:
            shader: The shader program to use for rendering lines.
            mvp_matrix: The combined Model-View-Projection matrix.
            colors: The resolved ColorSet containing color data.
            show_travel_moves: Whether to render the travel move paths.
        """
        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)

        # Draw powered moves (which use vertex colors)
        if self.powered_vertex_count > 0:
            shader.set_float("uUseVertexColor", 1.0)
            GL.glBindVertexArray(self.powered_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.powered_vertex_count)

        # Draw travel moves (uses a uniform color), if enabled
        if show_travel_moves and self.travel_vertex_count > 0:
            shader.set_float("uUseVertexColor", 0.0)
            shader.set_vec4("uColor", colors.get_rgba("travel"))
            GL.glBindVertexArray(self.travel_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.travel_vertex_count)

        shader.set_float("uUseVertexColor", 0.0)
        GL.glBindVertexArray(0)

    def _tessellate_arc(
        self,
        start_gl: Tuple[float, ...],
        end_gl: Tuple[float, ...],
        cmd: ArcToCommand,
    ) -> List[float]:
        """
        Converts an arc command into a series of line segments.

        Args:
            start_gl: The starting point of the arc in GL coordinates (X, Y, Z)
            end_gl: The ending point of the arc in GL coordinates (X, Y, Z)
            cmd: The ArcToCommand object.

        Returns:
            A list of floats representing the vertices of the line segments.
        """
        vertices = []
        center_x = start_gl[0] + cmd.center_offset[0]
        center_y = start_gl[1] + cmd.center_offset[1]
        radius = math.dist((start_gl[0], start_gl[1]), (center_x, center_y))

        if radius > 1e-6:
            start_angle = math.atan2(
                start_gl[1] - center_y, start_gl[0] - center_x
            )
            end_angle = math.atan2(end_gl[1] - center_y, end_gl[0] - center_x)
            arc_angle = end_angle - start_angle

            # Adjust angle for direction
            if cmd.clockwise and arc_angle > 0:
                arc_angle -= 2 * math.pi
            elif not cmd.clockwise and arc_angle < 0:
                arc_angle += 2 * math.pi

            # Determine number of segments based on arc length
            num_segments = max(2, int(abs(arc_angle * radius) / 0.5))
            prev_point = start_gl
            for i in range(1, num_segments + 1):
                t = i / num_segments
                angle = start_angle + arc_angle * t
                # Linear interpolation for height (Z in GL)
                z = start_gl[2] + (end_gl[2] - start_gl[2]) * t
                next_point = (
                    center_x + radius * math.cos(angle),
                    center_y + radius * math.sin(angle),
                    z,
                )
                vertices.extend(prev_point)
                vertices.extend(next_point)
                prev_point = next_point
        else:
            # If radius is negligible, draw a straight line
            vertices.extend(start_gl)
            vertices.extend(end_gl)
        return vertices

    def _load_buffer_data(self, vbo: int, data: np.ndarray):
        """Loads vertex data into a VBO."""
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            data.nbytes if data.size > 0 else 0,
            data if data.size > 0 else None,
            GL.GL_DYNAMIC_DRAW,
        )
