"""
A ring-buffer GPU renderer for progressively revealing raster scanlines
during simulation playback.

The buffer has a fixed vertex capacity.  As the playhead advances through
ScanLinePowerCommands, their powered line-segments are uploaded into the
ring, wrapping around as needed.  When all scanlines for a given texture
instance have been fully executed the texture is un-dimmed and those ring
slots become available for recycling.
"""

import numpy as np
from OpenGL import GL

from .gl_utils import BaseRenderer, RenderContext, Shader, set_line_width


class RingBufferRenderer(BaseRenderer):
    """
    Renders scanline line-segments from a fixed-size ring buffer.

    The caller encodes powered pixel-segments for each ScanLinePowerCommand
    and appends them in command-index order.  ``render()`` draws only the
    first *n* vertices, where *n* corresponds to the playhead position.
    """

    def __init__(self, capacity_vertices: int = 4_000_000):
        super().__init__()
        self._capacity = capacity_vertices
        self.vao: int = 0
        self.pos_vbo: int = 0
        self.pow_vbo: int = 0
        self.vertex_count: int = 0
        self._color_lut_texture: int = 0

    def init_gl(self):
        self.pos_vbo = self._create_vbo()
        self.pow_vbo = self._create_vbo()
        self._color_lut_texture = self._create_texture()

        zeros = np.zeros(self._capacity * 3, dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pos_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, zeros.nbytes, zeros, GL.GL_DYNAMIC_DRAW
        )

        zeros_pow = np.zeros(self._capacity * 4, dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pow_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            zeros_pow.nbytes,
            zeros_pow,
            GL.GL_DYNAMIC_DRAW,
        )

        self.vao = self._create_vao()
        GL.glBindVertexArray(self.vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pos_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pow_vbo)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def upload(self, positions: np.ndarray, power_values: np.ndarray):
        pos = np.ascontiguousarray(positions, dtype=np.float32).ravel()
        pow_flat = np.ascontiguousarray(power_values, dtype=np.float32).ravel()
        n = pos.size // 3
        assert pow_flat.size == n
        assert n <= self._capacity, (
            f"Scanline overlay has {n} vertices but ring capacity is "
            f"{self._capacity}"
        )

        pow_vec4 = np.zeros(n * 4, dtype=np.float32)
        pow_vec4[0::4] = pow_flat
        pow_vec4[3::4] = 1.0

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pos_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, pos.nbytes, pos)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pow_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, pow_vec4.nbytes, pow_vec4)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.vertex_count = n

    def update_color_lut(self, lut_data: np.ndarray):
        if not self._color_lut_texture:
            return
        lut = np.ascontiguousarray(lut_data, dtype=np.float32)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_lut_texture)
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
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            lut.shape[0],
            1,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            lut,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def clear(self):
        self.vertex_count = 0

    def render(
        self,
        ctx: RenderContext,
        shader: Shader,
        mvp_matrix: np.ndarray,
        executed_vertex_count: int = -1,
        alpha_pending: float = 0.2,
    ):
        if self.vertex_count == 0:
            return

        draw_count = self.vertex_count
        if executed_vertex_count >= 0:
            draw_count = min(executed_vertex_count, self.vertex_count)

        if draw_count == 0:
            return

        line_width = ctx.line_width
        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)
        shader.set_float("uHasNormals", 0.0)
        shader.set_float("uUsePowerLUT", 1.0)
        shader.set_vec4(
            "uZeroPowerColor", ctx.color_set.get_rgba("zero_power")
        )
        shader.set_int("uExecutedVertexCount", -1)
        shader.set_float("uAlphaPending", alpha_pending)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_lut_texture)
        shader.set_int("uColorLUT", 1)

        GL.glDisable(GL.GL_DEPTH_TEST)
        set_line_width(line_width)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_LINES, 0, draw_count)
        GL.glBindVertexArray(0)
        set_line_width(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)

        shader.set_float("uUsePowerLUT", 0.0)
        shader.set_float("uUseVertexColor", 0.0)
        shader.set_int("uExecutedVertexCount", -1)
