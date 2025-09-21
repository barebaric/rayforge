"""
A renderer for visualizing toolpath operations (Ops) in 3D.
"""

import math
import logging
from typing import List, Tuple
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader
from ...core.ops import (
    ArcToCommand,
    Command,
    LineToCommand,
    MoveToCommand,
    ScanLinePowerCommand,
    Ops,
)

logger = logging.getLogger(__name__)


class OpsRenderer(BaseRenderer):
    """Renders toolpath operations (cuts and travels) as colored lines."""

    def __init__(self):
        """Initializes the OpsRenderer."""
        super().__init__()
        self.cut_vao: int = 0
        self.travel_vao: int = 0
        self.cut_vertex_count: int = 0
        self.travel_vertex_count: int = 0
        self.cut_vbo: int = 0
        self.travel_vbo: int = 0
        # New VBO/VAO for variable power lines
        self.raster_vao: int = 0
        self.raster_vbo: int = 0
        self.raster_colors_vbo: int = 0
        self.raster_vertex_count: int = 0

    def init_gl(self):
        """
        Initializes OpenGL resources and sets up the VAO states permanently.
        """
        # Create Buffers
        self.cut_vbo = self._create_vbo()
        self.travel_vbo = self._create_vbo()
        self.raster_vbo = self._create_vbo()
        self.raster_colors_vbo = self._create_vbo()

        # Configure VAO for Cut Moves
        self.cut_vao = self._create_vao()
        GL.glBindVertexArray(self.cut_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cut_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Configure VAO for Travel Moves
        self.travel_vao = self._create_vao()
        GL.glBindVertexArray(self.travel_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.travel_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Configure VAO for Raster Moves
        self.raster_vao = self._create_vao()
        GL.glBindVertexArray(self.raster_vao)
        # Position attribute (location 0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.raster_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        # Color attribute (location 1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.raster_colors_vbo)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        # Unbind all
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def clear(self):
        """Clears the renderer's buffers and resets vertex counts."""
        self.update_from_ops(Ops())

    def update_from_ops(self, ops: Ops):
        """Synchronously processes an Ops object and updates vertex buffers."""
        (
            cut_verts,
            travel_verts,
            raster_verts,
            raster_colors,
        ) = self.prepare_vertex_data(ops)
        self.update_from_vertex_data(
            cut_verts, travel_verts, raster_verts, raster_colors
        )

    def prepare_vertex_data(
        self, ops: Ops
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Processes an Ops object into numpy arrays of vertices. This method is
        thread-safe and can be run in the background.
        """
        cut_vertices: List[float] = []
        travel_vertices: List[float] = []
        raster_vertices: List[float] = []
        raster_colors: List[float] = []
        last_point: Tuple[float, float, float] = 0.0, 0.0, 0.0

        # First pass: find global min and max power (for powers > 0)
        global_min_power = float("inf")
        global_max_power = 0
        for command in getattr(ops, "commands", []):
            if isinstance(command, ScanLinePowerCommand):
                power_values = getattr(command, "power_values", [])
                for p in power_values:
                    if p > 0:
                        global_min_power = min(global_min_power, p)
                        global_max_power = max(global_max_power, p)
        if global_min_power == float("inf"):
            global_min_power = 0

        # Second pass: build vertices
        for command in getattr(ops, "commands", []):
            if not isinstance(command, Command) or command.is_marker_command():
                continue

            if isinstance(command, ScanLinePowerCommand):
                power_values = getattr(command, "power_values", [])
                if not power_values or command.end is None:
                    last_point = command.end or last_point
                    continue

                if global_max_power <= 0:
                    last_point = command.end
                    continue

                p_start = np.array(last_point, dtype=np.float32)
                p_end = np.array(command.end, dtype=np.float32)
                line_vec = p_end - p_start
                num_values = len(power_values)

                num_runs = 0

                if num_values > 0:
                    run_start_idx = 0
                    for i in range(1, num_values):
                        if power_values[i] != power_values[run_start_idx]:
                            power = power_values[run_start_idx]
                            if power > 0:
                                t_start = run_start_idx / num_values
                                t_end = i / num_values
                                start_pt = p_start + line_vec * t_start
                                end_pt = p_start + line_vec * t_end
                                raster_vertices.extend(start_pt)
                                raster_vertices.extend(end_pt)
                                p_norm = (
                                    (power - global_min_power)
                                    / (global_max_power - global_min_power)
                                    if global_max_power > global_min_power
                                    else 0.0
                                )
                                color = (
                                    1.0 - p_norm,
                                    1.0 - p_norm,
                                    1.0 - p_norm,
                                    1.0,
                                )
                                raster_colors.extend(color)
                                raster_colors.extend(color)
                                num_runs += 1
                            run_start_idx = i
                    # Process the final run
                    power = power_values[run_start_idx]
                    if power > 0:
                        t_start = run_start_idx / num_values
                        start_pt = p_start + line_vec * t_start
                        raster_vertices.extend(start_pt)
                        raster_vertices.extend(p_end)
                        p_norm = (
                            (power - global_min_power)
                            / (global_max_power - global_min_power)
                            if global_max_power > global_min_power
                            else 0.0
                        )
                        color = (1.0 - p_norm, 1.0 - p_norm, 1.0 - p_norm, 1.0)
                        raster_colors.extend(color)
                        raster_colors.extend(color)
                        num_runs += 1

                last_point = command.end
                continue

            if command.end is None:
                continue

            # Convert to GL coordinates (Z-up, so direct mapping)
            end_point = (
                float(command.end[0]),
                float(command.end[1]),
                float(command.end[2]),
            )
            start_point = last_point

            if isinstance(command, MoveToCommand):
                if not np.allclose(start_point, end_point):
                    travel_vertices.extend(start_point)
                    travel_vertices.extend(end_point)
            elif isinstance(command, LineToCommand):
                cut_vertices.extend(start_point)
                cut_vertices.extend(end_point)
            elif isinstance(command, ArcToCommand):
                arc_verts = self._tessellate_arc(
                    start_point, end_point, command
                )
                cut_vertices.extend(arc_verts)

            last_point = command.end

        return (
            np.array(cut_vertices, dtype=np.float32),
            np.array(travel_vertices, dtype=np.float32),
            np.array(raster_vertices, dtype=np.float32),
            np.array(raster_colors, dtype=np.float32),
        )

    def update_from_vertex_data(
        self,
        cut_vertices: np.ndarray,
        travel_vertices: np.ndarray,
        raster_vertices: np.ndarray,
        raster_colors: np.ndarray,
    ):
        """Receives pre-processed vertex data and uploads it to the GPU."""
        self.cut_vertex_count = cut_vertices.size // 3
        self._load_buffer_data(self.cut_vbo, cut_vertices)
        self.travel_vertex_count = travel_vertices.size // 3
        self._load_buffer_data(self.travel_vbo, travel_vertices)
        self.raster_vertex_count = raster_vertices.size // 3
        self._load_buffer_data(self.raster_vbo, raster_vertices)
        self._load_buffer_data(self.raster_colors_vbo, raster_colors)

    def render(self, shader: Shader, mvp_matrix: np.ndarray) -> None:
        """
        Renders the toolpaths. The VAO state is pre-configured, so this
        method only needs to bind the appropriate VAO and draw.

        Args:
            shader: The shader program to use for rendering lines.
            mvp_matrix: The combined Model-View-Projection matrix.
        """
        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)

        # Draw rasters (which use vertex colors)
        if self.raster_vertex_count > 0:
            shader.set_vec4("uColor", [1.0, 1.0, 1.0, 1.0])
            shader.set_float("uUseVertexColor", 1.0)
            GL.glBindVertexArray(self.raster_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.raster_vertex_count)

        # Draw cuts moves (which use a uniform color)
        shader.set_float("uUseVertexColor", 0.0)
        self._draw_buffer(
            self.cut_vao,
            self.cut_vertex_count,
            shader,
            [1.0, 0.0, 1.0, 1.0],
        )

        # Draw travel moves
        self._draw_buffer(
            self.travel_vao,
            self.travel_vertex_count,
            shader,
            [1.0, 0.4, 0.0, 0.7],
        )

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

    def _draw_buffer(
        self,
        vao: int,
        vertex_count: int,
        shader: Shader,
        color: List[float],
    ):
        """
        Draws the contents of a pre-configured VAO with a uniform color.
        """
        if vertex_count > 0:
            shader.set_vec4("uColor", color)
            GL.glBindVertexArray(vao)
            GL.glDrawArrays(GL.GL_LINES, 0, vertex_count)
