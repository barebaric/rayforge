"""
A simple renderer for a 2D plane in 3D space.
"""

from __future__ import annotations
import logging
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class PlaneRenderer(BaseRenderer):
    """Renders a single, colored plane on the XY axis."""

    def __init__(
        self,
        width: float,
        height: float,
        color: tuple[float, float, float, float],
        z_offset: float = 0.0,
    ):
        """Initializes the PlaneRenderer."""
        super().__init__()
        self.width = width
        self.height = height
        self.color = color
        self.z_offset = z_offset
        self.vao: int = 0
        self.vbo: int = 0
        self.vertex_count: int = 0

    def init_gl(self) -> None:
        """Creates the VAO and VBO for the plane."""
        vertices = [
            0.0,
            0.0,
            self.z_offset,
            self.width,
            0.0,
            self.z_offset,
            0.0,
            self.height,
            self.z_offset,
            self.width,
            0.0,
            self.z_offset,
            self.width,
            self.height,
            self.z_offset,
            0.0,
            self.height,
            self.z_offset,
        ]
        self.vertex_count = len(vertices) // 3

        # Use the base class helpers to create and track resources
        self.vao = self._create_vao()
        self.vbo = self._create_vbo()

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        data = np.array(vertices, dtype=np.float32)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def render(self, shader: Shader, mvp: np.ndarray) -> None:
        """Draws the plane."""
        if not self.vao:
            return

        shader.set_mat4("uMVP", mvp)
        shader.set_vec4("uColor", self.color)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
        GL.glBindVertexArray(0)
