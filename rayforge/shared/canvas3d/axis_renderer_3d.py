"""
Renders a 3D grid and axes for a scene.

This module provides the AxisRenderer3D class, which is responsible for
creating and drawing a grid on the XY plane, along with labeled X and Y
axes. It also includes text rendering for the axis labels.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader
from .text_renderer_3d import TextRenderer3D

logger = logging.getLogger(__name__)


class AxisRenderer3D(BaseRenderer):
    """Renders a 3D grid with axes and numeric labels on the XY plane."""

    def __init__(
        self, width_mm: float, height_mm: float, grid_size_mm: float = 10.0
    ):
        """Initializes the AxisRenderer3D with scene dimensions.

        Args:
            width_mm: The total width of the grid along the X-axis in mm.
            height_mm: The total height of the grid along the Y-axis in mm.
            grid_size_mm: The spacing between grid lines in mm.
        """
        super().__init__()
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.grid_size_mm = float(grid_size_mm)

        self.grid_vao: int = 0
        self.axis_vao: int = 0
        self.grid_vbo: int = 0
        self.axis_vbo: int = 0
        self.grid_vertex_count: int = 0
        self.axis_vertex_count: int = 0
        self.text_renderer: Optional[TextRenderer3D] = None
        self.grid_color: Tuple[float, float, float, float] = 0.4, 0.4, 0.4, 1.0
        self.axis_color: Tuple[float, float, float, float] = 1.0, 1.0, 1.0, 1.0
        self.label_color: Tuple[float, float, float, float] = (
            0.9,
            0.9,
            0.9,
            1.0,
        )

    def set_grid_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the grid lines."""
        self.grid_color = color

    def set_axis_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the main X and Y axis lines."""
        self.axis_color = color

    def set_label_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the axis labels."""
        self.label_color = color

    def init_gl(self) -> None:
        """Initializes OpenGL resources for rendering the grid and axes."""
        grid_vertices, axis_vertices = [], []

        # A small negative Z offset prevents z-fighting with objects on grid
        z_pos = -0.001

        # Define vertices for the grid lines on the XY plane
        # Lines parallel to Y-axis (constant X)
        x_ticks = np.arange(
            self.grid_size_mm, self.width_mm, self.grid_size_mm
        )
        for x in x_ticks:
            grid_vertices.extend([x, 0.0, z_pos, x, self.height_mm, z_pos])

        # Lines parallel to X-axis (constant Y)
        y_ticks = np.arange(
            self.grid_size_mm, self.height_mm, self.grid_size_mm
        )
        for y in y_ticks:
            grid_vertices.extend([0.0, y, z_pos, self.width_mm, y, z_pos])
        self.grid_vertex_count = len(grid_vertices) // 3

        # Define vertices for the main axis lines
        # X-Axis at Y=0
        axis_vertices.extend([0.0, 0.0, 0.0, self.width_mm, 0.0, 0.0])
        # Y-Axis at X=0
        axis_vertices.extend([0.0, 0.0, 0.0, 0.0, self.height_mm, 0.0])
        self.axis_vertex_count = len(axis_vertices) // 3

        # Create and configure OpenGL objects with strict state isolation.
        # This prevents side effects when interacting with other renderers.
        self._create_gl_objects(grid_vertices, axis_vertices)

        # Initialize the text renderer now that the GL state is clean
        self.text_renderer = TextRenderer3D()
        self.text_renderer.init_gl()

    def _create_gl_objects(
        self, grid_vertices: list[float], axis_vertices: list[float]
    ) -> None:
        """Creates VAOs and VBOs and uploads vertex data to the GPU."""
        # Create Grid resources
        self.grid_vao = GL.glGenVertexArrays(1)
        self.grid_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.grid_vbo)
        grid_data = np.array(grid_vertices, dtype=np.float32)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, grid_data.nbytes, grid_data, GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Create Axis resources
        self.axis_vao = GL.glGenVertexArrays(1)
        self.axis_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self.axis_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axis_vbo)
        axis_data = np.array(axis_vertices, dtype=np.float32)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, axis_data.nbytes, axis_data, GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Unbind all buffers and vertex arrays to leave a clean state
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def render(
        self,
        line_shader: Shader,
        text_shader: Shader,
        mvp: np.ndarray,
        view_matrix: np.ndarray,
    ) -> None:
        """
        Renders the grid, axes, and labels.

        Args:
            line_shader: The shader program to use for drawing lines.
            text_shader: The shader program to use for drawing text.
            mvp: The combined Model-View-Projection matrix.
            view_matrix: The view matrix, used for billboarding text.
        """
        if not (self.grid_vao and self.axis_vao and self.text_renderer):
            return

        line_shader.use()
        line_shader.set_mat4("uMVP", mvp)

        # Draw grid lines
        GL.glLineWidth(1.0)
        line_shader.set_vec4("uColor", self.grid_color)
        GL.glBindVertexArray(self.grid_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.grid_vertex_count)

        # Draw main axes
        GL.glLineWidth(2.0)
        GL.glBindVertexArray(self.axis_vao)
        line_shader.set_vec4("uColor", self.axis_color)
        GL.glDrawArrays(GL.GL_LINES, 0, self.axis_vertex_count)

        GL.glBindVertexArray(0)

        # Draw text labels for the axes
        self._render_axis_labels(text_shader, mvp, view_matrix)

    def _render_axis_labels(
        self,
        text_shader: Shader,
        mvp_matrix: np.ndarray,
        view_matrix: np.ndarray,
    ) -> None:
        """Helper method to render text labels along the axes."""
        if not self.text_renderer:
            return

        label_color = self.label_color
        label_scale = 0.1
        label_offset = label_scale * 20

        # X-axis labels
        x_ticks = np.arange(
            self.grid_size_mm, self.width_mm + 1e-5, self.grid_size_mm
        )
        for x in x_ticks:
            position = np.array([x, -label_offset, 0.0])
            self.text_renderer.render_text(
                shader=text_shader,
                text=str(int(x)),
                position=position,
                scale=label_scale,
                color=label_color,
                mvp_matrix=mvp_matrix,
                view_matrix=view_matrix,
            )

        # Y-axis labels
        y_ticks = np.arange(
            self.grid_size_mm, self.height_mm + 1e-5, self.grid_size_mm
        )
        for y in y_ticks:
            position = np.array([-label_offset, y, 0])
            self.text_renderer.render_text(
                shader=text_shader,
                text=str(int(y)),
                position=position,
                scale=label_scale,
                color=label_color,
                mvp_matrix=mvp_matrix,
                view_matrix=view_matrix,
            )

    def cleanup(self) -> None:
        """Releases all OpenGL resources used by the renderer."""
        if self.text_renderer:
            self.text_renderer.cleanup()

        try:
            # Collect all valid VAO and VBO handles
            vaos_to_delete = [
                vao for vao in [self.grid_vao, self.axis_vao] if vao
            ]
            if vaos_to_delete:
                GL.glDeleteVertexArrays(len(vaos_to_delete), vaos_to_delete)

            vbos_to_delete = [
                vbo for vbo in [self.grid_vbo, self.axis_vbo] if vbo
            ]
            if vbos_to_delete:
                GL.glDeleteBuffers(len(vbos_to_delete), vbos_to_delete)
        except Exception:
            # Log error but don't crash, as this is often called on exit
            logger.exception("Error during axis renderer cleanup")
