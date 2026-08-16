"""
Renders workpiece base images as textured quads on the engrave plane.

Each visible workpiece with a source image contributes one instance:
an RGBA texture placed by a 4x4 model matrix that maps the unit quad
onto the workpiece's world-space footprint.  The image is drawn
unmodified (no laser colour LUT) so it matches the base image shown on
the 2D canvas.

Texture uploads run on the GL thread via :meth:`set_images`; the
actual pixel rendering happens off-thread in the scene presenter.
"""

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from OpenGL import GL

from .base import BaseRenderer

if TYPE_CHECKING:
    from ..gl_utils import ShaderSet
    from ..render_context import RenderContext

logger = logging.getLogger(__name__)


class WorkpieceImageRenderer(BaseRenderer):
    """Draws workpiece base-image quads with the unlit image shader."""

    visibility_key = "show_workpiece_image"

    def __init__(self):
        super().__init__()
        self.is_initialized = False
        self.instances: list[dict[str, Any]] = []
        self._vao: int = 0
        self._vbo: int = 0
        self._mvp: np.ndarray | None = None

    def prepare(self, ctx: "RenderContext") -> None:
        """Caches the per-frame camera MVP matrix."""
        self._mvp = ctx.camera.mvp_ui

    def init_gl(self) -> None:
        """Creates the quad VAO/VBO used by every instance."""
        if self.is_initialized:
            return
        self._vbo = self._create_vbo()
        self._vao = self._create_vao()

        # A unit quad on the z=0 engrave plane; the per-instance model
        # matrix maps it into world space.
        # fmt: off
        quad_vertices = np.array(
            [
                0.0, 0.0, 0.0, 0.0, 1.0,
                1.0, 0.0, 0.0, 1.0, 1.0,
                1.0, 1.0, 0.0, 1.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        # fmt: on

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            quad_vertices.nbytes,
            quad_vertices,
            GL.GL_STATIC_DRAW,
        )

        GL.glBindVertexArray(self._vao)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(0)
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(3 * 4)
        )
        GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

        self.is_initialized = True
        logger.debug("WorkpieceImageRenderer initialized")

    def _cleanup_self(self):
        """Deletes instance textures on cleanup."""
        if not self.is_initialized:
            return
        self.clear()
        self.is_initialized = False

    def _create_gl_texture(self, pixels: np.ndarray) -> int:
        """Uploads RGBA pixels as an sRGB texture with mipmaps."""
        texture_id = GL.glGenTextures(1)
        height, width = pixels.shape[:2]
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_LINEAR_MIPMAP_LINEAR,
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
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_SRGB8_ALPHA8,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_UNSIGNED_BYTE,
            pixels,
        )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        return texture_id

    def set_images(self, images: list[dict[str, Any]]) -> None:
        """Replaces all workpiece image instances.

        Each entry must provide ``pixels`` (an RGBA uint8 array) and
        ``model_matrix`` (a 4x4 float32 array mapping the unit quad to
        world space).  Runs on the GL thread.
        """
        if not self.is_initialized:
            return
        for instance in self.instances:
            GL.glDeleteTextures([instance["texture_id"]])
        self.instances.clear()

        for image in images:
            pixels = np.ascontiguousarray(image["pixels"], dtype=np.uint8)
            if pixels.ndim != 3 or pixels.shape[2] != 4:
                logger.warning("Skipping workpiece image with bad pixel data.")
                continue
            model_matrix = np.asarray(image["model_matrix"], dtype=np.float32)
            texture_id = self._create_gl_texture(pixels)
            self.instances.append(
                {
                    "texture_id": texture_id,
                    "model_matrix": model_matrix,
                }
            )

    def clear(self) -> None:
        """Deletes all instance textures."""
        if not self.is_initialized:
            return
        textures = [instance["texture_id"] for instance in self.instances]
        if textures:
            GL.glDeleteTextures(textures)
        self.instances.clear()

    def render(self, ctx: "RenderContext", shaders: "ShaderSet", **kwargs):
        """Draws every workpiece base image quad."""
        if not self.is_initialized or not self.instances:
            return
        shader = shaders.image
        if shader is None or self._mvp is None:
            return

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)
        shader.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        shader.set_float("uAlpha", 1.0)

        GL.glBindVertexArray(self._vao)
        for instance in self.instances:
            shader.set_mat4("uMVP", self._mvp @ instance["model_matrix"])
            GL.glBindTexture(GL.GL_TEXTURE_2D, instance["texture_id"])
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        GL.glBindVertexArray(0)
