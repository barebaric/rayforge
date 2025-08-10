"""
Renders a sphere composed of triangle strips using modern OpenGL.
"""

import logging
import math
from typing import Union
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class SphereRenderer(BaseRenderer):
    """Manages the VAO, VBO, and rendering logic for a sphere."""

    def __init__(
        self, radius: float, latitude_segments: int, longitude_segments: int
    ):
        """Initializes the SphereRenderer."""
        super().__init__()
        self.radius = radius
        self.latitude_segments = latitude_segments
        self.longitude_segments = longitude_segments
        self.vao: int = 0
        self.vbo: int = 0
        self.vertex_count = 0

    def init_gl(self) -> None:
        """Generates sphere vertices and initializes OpenGL buffers."""
        vertices = []
        pi = math.pi

        # Generate vertices for triangle strips
        for i in range(self.latitude_segments):
            # phi is the elevation angle from the x-z plane.
            phi1 = pi / 2.0 - (i * pi / self.latitude_segments)
            phi2 = pi / 2.0 - ((i + 1) * pi / self.latitude_segments)

            for j in range(self.longitude_segments + 1):
                # theta is the azimuthal angle in the x-z plane.
                theta = j * 2.0 * pi / self.longitude_segments

                # Vertex on the first latitude ring
                x1 = self.radius * math.cos(phi1) * math.cos(theta)
                y1 = self.radius * math.sin(phi1)
                z1 = self.radius * math.cos(phi1) * math.sin(theta)
                vertices.extend([x1, y1, z1])

                # Vertex on the second latitude ring
                x2 = self.radius * math.cos(phi2) * math.cos(theta)
                y2 = self.radius * math.sin(phi2)
                z2 = self.radius * math.cos(phi2) * math.sin(theta)
                vertices.extend([x2, y2, z2])

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

        # Position attribute (location = 0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Unbind VBO and VAO
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)

    def render(
        self,
        shader: Shader,
        proj_matrix: np.ndarray,
        view_matrix: np.ndarray,
        position: np.ndarray,
        color: Union[list, tuple],
        scale: float = 1.0,
    ) -> None:
        """
        Renders the sphere at a given position, color, and scale.

        Args:
            shader: The shader program to use for rendering.
            proj_matrix: The projection matrix.
            view_matrix: The view matrix.
            position: The world-space position (vec3) of the sphere.
            color: The color of the sphere as a list/tuple (r, g, b, a).
            scale: A uniform scaling factor for the sphere.
        """
        if not self.vao:
            return

        shader.use()

        # Build a model matrix to scale and then translate the sphere.
        model_matrix = np.diag([scale, scale, scale, 1.0]).astype(np.float32)
        model_matrix[:3, 3] = position[:3]

        # Calculate the final Model-View-Projection matrix.
        mvp_matrix = proj_matrix @ view_matrix @ model_matrix

        shader.set_mat4("uMVP", mvp_matrix)
        shader.set_vec4("uColor", color)

        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, self.vertex_count)
        GL.glBindVertexArray(0)
