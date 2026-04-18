"""
Renders a cylinder wireframe for visualizing rotary mode workpieces.
"""

import logging
import math
from typing import Tuple
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, RenderContext, Shader

logger = logging.getLogger(__name__)


class CylinderRenderer(BaseRenderer):
    """Renders a wireframe cylinder to visualize rotary mode workpieces."""

    def __init__(
        self,
        diameter: float,
        length: float,
        rings: int = 16,
        length_segments: int = 8,
    ):
        super().__init__()
        self.diameter = diameter
        self.length = length
        self.rings = rings
        self.length_segments = length_segments

        self.vao: int = 0
        self.vbo: int = 0
        self.vertex_count = 0
        self._color: Tuple[float, float, float, float] = (
            0.5,
            0.5,
            0.5,
            0.3,
        )

    def set_color(self, color: Tuple[float, float, float, float]):
        """Sets the wireframe color."""
        self._color = color

    def init_gl(self) -> None:
        """Generates cylinder wireframe vertices and initializes OpenGL."""
        vertices = []
        radius = self.diameter / 2.0

        for i in range(self.length_segments + 1):
            cyl_pos = (i / self.length_segments) * self.length
            for j in range(self.rings):
                theta1 = (j / self.rings) * 2.0 * math.pi
                theta2 = ((j + 1) / self.rings) * 2.0 * math.pi

                r1a = radius * math.sin(theta1)
                r1b = radius * math.cos(theta1)
                r2a = radius * math.sin(theta2)
                r2b = radius * math.cos(theta2)

                p1 = [cyl_pos, r1a, r1b]
                p2 = [cyl_pos, r2a, r2b]
                vertices.extend(p1)
                vertices.extend(p2)

        for j in range(self.rings):
            theta = (j / self.rings) * 2.0 * math.pi
            ra = radius * math.sin(theta)
            rb = radius * math.cos(theta)

            for i in range(self.length_segments):
                cyl1 = (i / self.length_segments) * self.length
                cyl2 = ((i + 1) / self.length_segments) * self.length
                p1 = [cyl1, ra, rb]
                p2 = [cyl2, ra, rb]
                vertices.extend(p1)
                vertices.extend(p2)

        self.vertex_count = len(vertices) // 3
        vertex_data = np.array(vertices, dtype=np.float32)

        self.vao = self._create_vao()
        self.vbo = self._create_vbo()

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            vertex_data.nbytes,
            vertex_data,
            GL.GL_STATIC_DRAW,
        )

        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def render(
        self, ctx: RenderContext, shader: Shader, mvp_matrix: np.ndarray
    ) -> None:
        """
        Renders the cylinder wireframe.

        Args:
            shader: The shader program to use.
            mvp_matrix: The Model-View-Projection matrix.
        """
        if not self.vao or self.vertex_count == 0:
            return

        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)
        shader.set_vec4("uColor", self._color)
        shader.set_float("uUseVertexColor", 0.0)
        shader.set_float("uHasNormals", 0.0)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.vertex_count)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
