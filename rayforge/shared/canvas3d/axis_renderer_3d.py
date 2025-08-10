"""
Renders a 3D grid and axes for a scene.

This module provides the AxisRenderer3D class, which is responsible for
creating and drawing a grid on the XY plane, along with labeled X and Y
axes. It uses a composed PlaneRenderer for the background.
"""

from __future__ import annotations
import logging
from typing import Optional, Tuple
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader
from .text_renderer_3d import TextRenderer3D
from .plane_renderer import PlaneRenderer

logger = logging.getLogger(__name__)


class AxisRenderer3D(BaseRenderer):
    """Renders a 3D grid with axes, background, and labels on the XY plane."""

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

        # Colors
        self.background_color = 0.8, 0.8, 0.8, 0.1
        self.grid_color = 0.4, 0.4, 0.4, 1.0
        self.axis_color = 1.0, 1.0, 1.0, 1.0
        self.label_color = 0.9, 0.9, 0.9, 1.0

        # Composition
        self.background_renderer = PlaneRenderer(
            width=self.width_mm,
            height=self.height_mm,
            color=self.background_color,
            z_offset=-0.002,
        )
        self._add_child_renderer(self.background_renderer)

        self.text_renderer: Optional[TextRenderer3D] = None

        # Grid and Axes resources
        self.grid_vao, self.grid_vbo, self.grid_vertex_count = 0, 0, 0
        self.axes_vao, self.axes_vbo, self.axes_vertex_count = 0, 0, 0

    def set_background_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the background plane."""
        self.background_color = color
        self.background_renderer.color = color

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
        """Initializes OpenGL resources for all components."""
        # Delegate initialization to child renderers
        self.background_renderer.init_gl()

        self.text_renderer = TextRenderer3D()
        self.text_renderer.init_gl()
        self._add_child_renderer(self.text_renderer)

        # Initialize self-managed components using base class helpers
        self._init_grid_and_axes()

    def _init_grid_and_axes(self):
        """Creates VAOs/VBOs for the grid and axis lines."""
        grid_z_pos = -0.001
        w, h = self.width_mm, self.height_mm

        # Grid vertices
        grid_verts = []
        for x in np.arange(self.grid_size_mm, w, self.grid_size_mm):
            grid_verts.extend([x, 0.0, grid_z_pos, x, h, grid_z_pos])
        for y in np.arange(self.grid_size_mm, h, self.grid_size_mm):
            grid_verts.extend([0.0, y, grid_z_pos, w, y, grid_z_pos])

        # Axis vertices
        axis_verts = [0.0, 0.0, 0.0, w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, h, 0.0]

        # Create Grid resources
        self.grid_vao = self._create_vao()
        self.grid_vbo = self._create_vbo()
        self.grid_vertex_count = len(grid_verts) // 3
        GL.glBindVertexArray(self.grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.grid_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(grid_verts, dtype=np.float32).nbytes,
            np.array(grid_verts, dtype=np.float32),
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Create Axis resources
        self.axes_vao = self._create_vao()
        self.axes_vbo = self._create_vbo()
        self.axes_vertex_count = len(axis_verts) // 3
        GL.glBindVertexArray(self.axes_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(axis_verts, dtype=np.float32).nbytes,
            np.array(axis_verts, dtype=np.float32),
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindVertexArray(0)

    def render(
        self,
        line_shader: Shader,
        text_shader: Shader,
        mvp: np.ndarray,
        view_matrix: np.ndarray,
    ) -> None:
        """
        Orchestrates the rendering of all components in the correct order.

        Args:
            line_shader: The shader program to use for drawing lines/solids.
            text_shader: The shader program to use for drawing text.
            mvp: The combined Model-View-Projection matrix.
            view_matrix: The view matrix, used for billboarding text.
        """
        if not all((self.grid_vao, self.axes_vao, self.text_renderer)):
            return

        # Enable blending for transparent objects
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        line_shader.use()

        GL.glDepthMask(GL.GL_FALSE)
        self.background_renderer.render(line_shader, mvp)
        GL.glDepthMask(GL.GL_TRUE)

        line_shader.set_vec4("uColor", self.grid_color)
        GL.glLineWidth(1.0)
        GL.glBindVertexArray(self.grid_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.grid_vertex_count)

        line_shader.set_vec4("uColor", self.axis_color)
        GL.glLineWidth(2.0)
        GL.glBindVertexArray(self.axes_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.axes_vertex_count)

        GL.glBindVertexArray(0)

        self._render_axis_labels(text_shader, mvp, view_matrix)
        GL.glDisable(GL.GL_BLEND)

    def _render_axis_labels(
        self,
        text_shader: Shader,
        mvp_matrix: np.ndarray,
        view_matrix: np.ndarray,
    ) -> None:
        """Helper method to render text labels along the axes."""
        if not self.text_renderer:
            return
        label_scale = 0.1
        label_offset = label_scale * 20
        for x in np.arange(
            self.grid_size_mm, self.width_mm + 1e-5, self.grid_size_mm
        ):
            pos = np.array([x, -label_offset, 0.0])
            self.text_renderer.render_text(
                text_shader,
                str(int(x)),
                pos,
                label_scale,
                self.label_color,
                mvp_matrix,
                view_matrix,
            )
        for y in np.arange(
            self.grid_size_mm, self.height_mm + 1e-5, self.grid_size_mm
        ):
            pos = np.array([-label_offset, y, 0.0])
            self.text_renderer.render_text(
                text_shader,
                str(int(y)),
                pos,
                label_scale,
                self.label_color,
                mvp_matrix,
                view_matrix,
            )
