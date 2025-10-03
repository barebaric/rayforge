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
        powered_vertices: List[float] = []
        powered_colors: List[float] = []
        travel_vertices: List[float] = []
        zero_power_vertices: List[float] = []
        zero_power_colors: List[float] = []
        last_point: Tuple[float, float, float] = 0.0, 0.0, 0.0
        current_power = 0.0

        cut_lut = colors.get_lut("cut")
        engrave_lut = colors.get_lut("engrave")
        zero_power_rgba = colors.get_rgba("zero_power")

        ops.preload_state()
        for command in ops.commands:
            if not isinstance(command, Command) or command.is_marker_command():
                continue

            if hasattr(command, "state") and command.state is not None:
                current_power = command.state.power

            if isinstance(command, ScanLinePowerCommand):
                power_values = command.power_values
                if not power_values or command.end is None:
                    last_point = command.end or last_point
                    continue

                p_start = np.array(last_point, dtype=np.float32)
                p_end = np.array(command.end, dtype=np.float32)
                line_vec = p_end - p_start
                num_values = len(power_values)

                # Generate a line segment for each power value ("pixel")
                for i, power in enumerate(power_values):
                    t_start = i / num_values
                    t_end = (i + 1) / num_values
                    start_pt = p_start + line_vec * t_start
                    end_pt = p_start + line_vec * t_end

                    # Calculate colors for both vertices of this segment
                    if power > 0:
                        # power is 0-255, which can be used directly as a
                        # LUT index.
                        power_idx = min(255, power)
                        start_color = tuple(engrave_lut[power_idx])

                        # For the end color, use the next power value if
                        # available,
                        # otherwise use the same color
                        if i + 1 < len(power_values):
                            next_power = power_values[i + 1]
                            if next_power > 0:
                                next_power_idx = min(255, next_power)
                                end_color = tuple(engrave_lut[next_power_idx])
                            else:
                                end_color = zero_power_rgba
                        else:
                            end_color = start_color
                    else:
                        start_color = zero_power_rgba
                        # For the end color, use the next power value if
                        # available
                        if i + 1 < len(power_values):
                            next_power = power_values[i + 1]
                            if next_power > 0:
                                next_power_idx = min(255, next_power)
                                end_color = tuple(engrave_lut[next_power_idx])
                            else:
                                end_color = zero_power_rgba
                        else:
                            end_color = start_color

                    # Add vertices and their corresponding colors
                    if power > 0:
                        powered_vertices.extend(start_pt)
                        powered_vertices.extend(end_pt)
                        powered_colors.extend(start_color)
                        powered_colors.extend(end_color)
                    else:
                        zero_power_vertices.extend(start_pt)
                        zero_power_vertices.extend(end_pt)
                        zero_power_colors.extend(start_color)
                        zero_power_colors.extend(end_color)

                last_point = command.end
                continue

            if command.end is None:
                continue

            end_point = tuple(map(float, command.end))
            start_point = last_point

            if isinstance(command, MoveToCommand):
                if not np.allclose(start_point, end_point):
                    travel_vertices.extend(start_point)
                    travel_vertices.extend(end_point)
            elif isinstance(command, (LineToCommand, ArcToCommand)):
                is_zero_power = math.isclose(current_power, 0.0)

                if is_zero_power:
                    color = zero_power_rgba
                else:
                    # current_power is now 0.0-1.0, so scale to 0-255
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
    ) -> None:
        """
        Renders the toolpaths.

        Args:
            shader: The shader program to use for rendering lines.
            mvp_matrix: The combined Model-View-Projection matrix.
            colors: The resolved ColorSet containing color data.
        """
        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)

        # Draw powered moves (which use vertex colors)
        if self.powered_vertex_count > 0:
            shader.set_float("uUseVertexColor", 1.0)
            GL.glBindVertexArray(self.powered_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.powered_vertex_count)

        # Draw travel moves (uses a uniform color)
        if self.travel_vertex_count > 0:
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
