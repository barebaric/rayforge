"""
Renders a cylinder wireframe for visualizing rotary mode workpieces.
"""

import logging
import math
from typing import Tuple
import numpy as np
from OpenGL import GL
from ....core.ops.axis import Axis
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class CylinderRenderer(BaseRenderer):
    """Renders a wireframe cylinder to visualize rotary mode workpieces."""

    def __init__(
        self,
        diameter: float,
        length: float,
        rings: int = 16,
        length_segments: int = 8,
        source_axis: Axis = Axis.Y,
    ):
        """
        Initializes the CylinderRenderer.

        Args:
            diameter: The diameter of the cylinder in mm.
            length: The length of the cylinder along the cylinder axis
                    in mm.
            rings: Number of circular rings around the circumference.
            length_segments: Number of segments along the length.
            source_axis: Which linear axis maps to the rotation angle.
                         Axis.Y (default): cylinder along X.
                         Axis.X: cylinder along Y.
        """
        super().__init__()
        self.diameter = diameter
        self.length = length
        self.rings = rings
        self.length_segments = length_segments
        self._source_axis = source_axis

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
        cyl_idx = 0 if self._source_axis == Axis.Y else 1

        for i in range(self.length_segments + 1):
            cyl_pos = (i / self.length_segments) * self.length
            for j in range(self.rings):
                theta1 = (j / self.rings) * 2.0 * math.pi
                theta2 = ((j + 1) / self.rings) * 2.0 * math.pi

                r1a = radius * math.sin(theta1)
                r1b = radius * math.cos(theta1)
                r2a = radius * math.sin(theta2)
                r2b = radius * math.cos(theta2)

                p1 = [0.0, 0.0, 0.0]
                p2 = [0.0, 0.0, 0.0]
                p1[cyl_idx] = cyl_pos
                p1[1 - cyl_idx] = r1a
                p1[2] = r1b
                p2[cyl_idx] = cyl_pos
                p2[1 - cyl_idx] = r2a
                p2[2] = r2b
                vertices.extend(p1)
                vertices.extend(p2)

        for j in range(self.rings):
            theta = (j / self.rings) * 2.0 * math.pi
            ra = radius * math.sin(theta)
            rb = radius * math.cos(theta)

            for i in range(self.length_segments):
                cyl1 = (i / self.length_segments) * self.length
                cyl2 = ((i + 1) / self.length_segments) * self.length
                p1 = [0.0, 0.0, 0.0]
                p2 = [0.0, 0.0, 0.0]
                p1[cyl_idx] = cyl1
                p1[1 - cyl_idx] = ra
                p1[2] = rb
                p2[cyl_idx] = cyl2
                p2[1 - cyl_idx] = ra
                p2[2] = rb
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

    def render(self, shader: Shader, mvp_matrix: np.ndarray) -> None:
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
