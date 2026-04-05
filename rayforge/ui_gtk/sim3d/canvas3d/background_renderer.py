"""
A renderer for a gradient background that gives a raytraced studio appearance.
"""

import logging

from typing import Optional

import numpy as np
from OpenGL import GL

from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class BackgroundRenderer(BaseRenderer):
    """Renders a fullscreen gradient quad behind the 3D scene."""

    def __init__(self):
        super().__init__()
        self.vao: int = 0
        self.vbo: int = 0
        self._bg_color = (0.11, 0.12, 0.14)
        self._bg_color_light = (0.18, 0.20, 0.23)
        self._shader: Optional[Shader] = None

    def set_colors(self, bg_color: tuple, bg_color_light: tuple):
        self._bg_color = bg_color
        self._bg_color_light = bg_color_light

    def init_gl(self):
        from .shaders import (
            BACKGROUND_VERTEX_SHADER,
            BACKGROUND_FRAGMENT_SHADER,
        )

        self._shader = Shader(
            BACKGROUND_VERTEX_SHADER, BACKGROUND_FRAGMENT_SHADER
        )
        logger.info("Background gradient shader compiled successfully")

        vertices = np.array(
            [
                -1,
                -1,
                0,
                1,
                -1,
                0,
                -1,
                1,
                0,
                1,
                1,
                0,
            ],
            dtype=np.float32,
        )

        self.vao = self._create_vao()
        self.vbo = self._create_vbo()

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def render(self):
        if not self._shader or not self.vao:
            return

        self._shader.use()
        self._shader.set_vec3("uBgColor", self._bg_color)
        self._shader.set_vec3("uBgColorLight", self._bg_color_light)

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_BLEND)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
