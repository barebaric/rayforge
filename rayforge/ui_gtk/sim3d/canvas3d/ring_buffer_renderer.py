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
        self.col_vbo: int = 0
        self.vertex_count: int = 0

    def init_gl(self):
        self.pos_vbo = self._create_vbo()
        self.col_vbo = self._create_vbo()

        zeros = np.zeros(self._capacity * 3, dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pos_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, zeros.nbytes, zeros, GL.GL_DYNAMIC_DRAW
        )

        zeros_col = np.zeros(self._capacity * 4, dtype=np.float32)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            zeros_col.nbytes,
            zeros_col,
            GL.GL_DYNAMIC_DRAW,
        )

        self.vao = self._create_vao()
        GL.glBindVertexArray(self.vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pos_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def upload(self, positions: np.ndarray, colors: np.ndarray):
        """
        Upload *all* scanline segment data, replacing any previous contents.

        *positions*: flat float32 array, shape ``(N*3,)`` or ``(N, 3)``.
        *colors*:    flat float32 array, shape ``(N*4,)`` or ``(N, 4)``.
        """
        pos = np.ascontiguousarray(positions, dtype=np.float32).ravel()
        col = np.ascontiguousarray(colors, dtype=np.float32).ravel()
        n = pos.size // 3
        assert col.size == n * 4
        assert n <= self._capacity, (
            f"Scanline overlay has {n} vertices but ring capacity is "
            f"{self._capacity}"
        )

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pos_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, pos.nbytes, pos)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.col_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, col.nbytes, col)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.vertex_count = n

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
        shader.set_float("uUseVertexColor", 1.0)
        shader.set_int("uExecutedVertexCount", -1)
        shader.set_float("uAlphaPending", alpha_pending)

        GL.glDisable(GL.GL_DEPTH_TEST)
        set_line_width(line_width)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_LINES, 0, draw_count)
        GL.glBindVertexArray(0)
        set_line_width(1.0)
        GL.glEnable(GL.GL_DEPTH_TEST)

        shader.set_float("uUseVertexColor", 0.0)
        shader.set_int("uExecutedVertexCount", -1)
