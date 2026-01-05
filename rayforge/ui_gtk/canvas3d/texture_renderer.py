"""
A renderer for visualizing texture-based artifacts using GPU texture rendering.
"""

import logging
from typing import List, Dict, Any
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader
from ...pipeline.artifact.base import TextureData


logger = logging.getLogger(__name__)


class TextureArtifactRenderer(BaseRenderer):
    """
    Renders texture-based artifacts as textured quads for high-performance
    visualization.

    This renderer uses a single quad with a texture containing power values,
    allowing for instant rendering of complex raster operations that would
    otherwise require millions of individual lines.
    """

    def __init__(self):
        """Initializes the TextureArtifactRenderer."""
        super().__init__()
        self.vao: int = 0
        self.vbo: int = 0
        self.texture: int = 0
        self.color_lut_texture: int = 0
        self.is_initialized: bool = False
        # A list to hold dictionaries for each texture instance
        self.instances: List[Dict[str, Any]] = []

    def init_gl(self):
        """
        Initializes OpenGL resources for rendering textured quads.

        Creates the VAO/VBO for a quad and OpenGL Textures for the texture
        data and color lookup table (LUT).
        """
        if self.is_initialized:
            return

        # Create vertex buffer for a simple quad
        self.vbo = self._create_vbo()
        self.vao = self._create_vao()
        self.texture = self._create_texture()
        self.color_lut_texture = self._create_texture()

        # Define quad vertices (position, texture coordinates)
        # fmt: off
        quad_vertices = np.array(
            [
                # Position (x, y, z)  Texture Coords (s, t)
                0.0, 0.0, 0.0, 0.0, 1.0,  # Bottom-left
                1.0, 0.0, 0.0, 1.0, 1.0,  # Bottom-right
                1.0, 1.0, 0.0, 1.0, 0.0,  # Top-right
                0.0, 1.0, 0.0, 0.0, 0.0,  # Top-left
            ],
            dtype=np.float32,
        )
        # fmt: on

        # Upload vertex data
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            quad_vertices.nbytes,
            quad_vertices,
            GL.GL_STATIC_DRAW,
        )

        # Set up vertex attributes
        GL.glBindVertexArray(self.vao)

        # Position attribute (location 0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(0)
        )
        GL.glEnableVertexAttribArray(0)

        # Texture coordinate attribute (location 1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(3 * 4)
        )
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)

        # Set up 2D texture for power data
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # Set up 2D texture (with height=1) for color LUT for compatibility
        # with the sampler2D in the shader.
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self.is_initialized = True
        logger.debug("TextureArtifactRenderer initialized")

    def _cleanup_self(self):
        """Cleans up OpenGL resources specific to this renderer."""
        if not self.is_initialized:
            return

        try:
            self.clear()
            self.is_initialized = False
        except Exception as e:
            logger.warning(f"TextureArtifactRenderer cleanup warning: {e}")

    def clear(self):
        """Clears all instances and their associated textures."""
        if not self.is_initialized:
            return
        # This needs to be called on the GL thread.
        textures_to_delete = [
            instance["texture_id"] for instance in self.instances
        ]
        if textures_to_delete:
            GL.glDeleteTextures(textures_to_delete)
        self.instances.clear()

    def add_instance(
        self, texture_data: TextureData, final_model_matrix: np.ndarray
    ):
        """Adds a texture artifact to be rendered in the next frame."""
        if not self.is_initialized:
            return

        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )

        height, width = texture_data.power_texture_data.shape
        # Tell OpenGL that our data is tightly packed (1-byte alignment).
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_R8,
            width,
            height,
            0,
            GL.GL_RED,
            GL.GL_UNSIGNED_BYTE,
            texture_data.power_texture_data,
        )
        # Restore the default alignment.
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self.instances.append(
            {"texture_id": texture_id, "model_matrix": final_model_matrix}
        )

    def update_color_lut(self, lut_data: np.ndarray):
        """
        Updates the color lookup table texture, now using GL_TEXTURE_2D.
        """
        if not self.is_initialized:
            return

        lut_data = np.ascontiguousarray(lut_data, dtype=np.float32)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            lut_data.shape[0],  # width = 256
            1,  # height = 1
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            lut_data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def render(self, view_proj_scene_matrix: np.ndarray, shader: Shader):
        """
        Renders all texture instances.

        Args:
            view_proj_scene_matrix: The combined Projection * View * SceneModel
              matrix (P*V*M_scene), not transposed.
            shader: The shader to use for rendering.
        """
        if not self.is_initialized or not self.instances:
            return

        shader.use()

        # Bind shared resources once
        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        shader.set_int("uColorLUT", 1)
        GL.glBindTexture(
            GL.GL_TEXTURE_2D, self.color_lut_texture
        )  # Bind as 2D
        GL.glBindVertexArray(self.vao)

        for instance in self.instances:
            # Combine the base VPS with this instance's specific model matrix
            final_mvp = view_proj_scene_matrix @ instance["model_matrix"]
            shader.set_mat4("uMVP", final_mvp.T)  # Transpose for OpenGL

            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, instance["texture_id"])

            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

        # Unbind all
        GL.glBindVertexArray(0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)  # Unbind as 2D
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
