"""
A renderer for visualizing toolpath operations (Ops) in 3D.
"""

import numpy as np
from typing import Optional
from OpenGL import GL
from ..gl_utils import BaseRenderer, RenderContext, Shader, set_line_width


class OpsRenderer(BaseRenderer):
    """Renders toolpath operations (cuts and travels) as colored lines."""

    def __init__(self):
        """Initializes the OpsRenderer."""
        super().__init__()
        self.powered_vao: int = 0
        self.travel_vao: int = 0

        self.powered_vbo: int = 0
        self.powered_powers_vbo: int = 0
        self.powered_index_vbo: int = 0
        self.travel_vbo: int = 0
        self.travel_index_vbo: int = 0

        self.powered_vertex_count: int = 0
        self.travel_vertex_count: int = 0

        self._color_lut_texture: int = 0
        self._num_laser_luts: int = 1

    def init_gl(self):
        self.powered_vbo = self._create_vbo()
        self.powered_powers_vbo = self._create_vbo()
        self.powered_index_vbo = self._create_vbo()
        self.travel_vbo = self._create_vbo()
        self.travel_index_vbo = self._create_vbo()
        self._color_lut_texture = self._create_texture()

        self.powered_vao = self._create_vao()
        GL.glBindVertexArray(self.powered_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_powers_vbo)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_index_vbo)
        GL.glVertexAttribPointer(3, 1, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(3)

        self.travel_vao = self._create_vao()
        GL.glBindVertexArray(self.travel_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.travel_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.travel_index_vbo)
        GL.glVertexAttribPointer(3, 1, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(3)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def clear(self):
        """Clears the renderer's buffers and resets vertex counts."""
        self.update_from_vertex_data(
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
        )

    def update_from_vertex_data(
        self,
        powered_vertices: np.ndarray,
        power_values: np.ndarray,
        travel_vertices: np.ndarray,
        laser_indices: Optional[np.ndarray] = None,
    ):
        self.powered_vertex_count = powered_vertices.size // 3
        self._load_buffer_data(self.powered_vbo, powered_vertices)
        pv_flat = np.ascontiguousarray(power_values, dtype=np.float32).ravel()
        n = pv_flat.size
        if laser_indices is not None and laser_indices.size > 0:
            li_flat = np.ascontiguousarray(
                laser_indices, dtype=np.float32
            ).ravel()
        else:
            li_flat = np.zeros(n, dtype=np.float32)
        power_vec4 = np.zeros(n * 4, dtype=np.float32)
        power_vec4[0::4] = pv_flat
        power_vec4[1::4] = li_flat
        power_vec4[3::4] = 1.0
        self._load_buffer_data(self.powered_powers_vbo, power_vec4)
        indices = np.arange(self.powered_vertex_count, dtype=np.float32)
        self._load_buffer_data(self.powered_index_vbo, indices)
        self.travel_vertex_count = travel_vertices.size // 3
        self._load_buffer_data(self.travel_vbo, travel_vertices)
        travel_indices = np.arange(self.travel_vertex_count, dtype=np.float32)
        self._load_buffer_data(self.travel_index_vbo, travel_indices)

    def update_color_lut(self, lut_data: np.ndarray, num_lasers: int = 1):
        if not self._color_lut_texture:
            return
        self._num_laser_luts = num_lasers
        lut = np.ascontiguousarray(lut_data, dtype=np.float32)
        if lut.ndim == 3:
            width, height = lut.shape[1], lut.shape[0]
        else:
            width, height = lut.shape[0], 1
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
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            lut,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def render(
        self,
        ctx: RenderContext,
        shader: Shader,
        mvp_matrix: np.ndarray,
        executed_vertex_count: int = -1,
        alpha_pending: float = 0.2,
        executed_travel_vertex_count: int = -1,
    ) -> None:
        """
        Renders the toolpaths. The vertices are assumed to be in world space.

        Args:
            ctx: The current render context (carries color set, line width,
              and travel-move visibility).
            shader: The shader program to use for rendering lines.
            mvp_matrix: The combined Model-View-Projection matrix.
            executed_vertex_count: Powered vertices drawn at full alpha.
                -1 means all vertices at full alpha (IDLE mode).
            alpha_pending: Alpha multiplier for pending vertices (0..1).
            executed_travel_vertex_count: Travel vertices drawn at full
                alpha during simulation. -1 means use ctx.show_travel_moves
                instead.
        """
        colors = ctx.color_set
        show_travel_moves = ctx.show_travel_moves
        line_width = ctx.line_width

        if executed_vertex_count > self.powered_vertex_count:
            raise ValueError(
                f"executed_vertex_count ({executed_vertex_count}) "
                f"> powered_vertex_count ({self.powered_vertex_count})"
            )

        shader.use()
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        shader.set_mat4("uMVP", mvp_matrix)
        shader.set_float("uHasNormals", 0.0)

        shader.set_int("uExecutedVertexCount", executed_vertex_count)
        shader.set_float("uAlphaPending", alpha_pending)
        shader.set_float("uEmissive", 1.0)

        if self.powered_vertex_count > 0:
            set_line_width(line_width)
            shader.set_float("uUsePowerLUT", 1.0)
            shader.set_int("uNumLaserLUTs", self._num_laser_luts)
            shader.set_vec4("uZeroPowerColor", colors.get_rgba("zero_power"))
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_lut_texture)
            shader.set_int("uColorLUT", 1)
            GL.glBindVertexArray(self.powered_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.powered_vertex_count)
            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
            GL.glActiveTexture(GL.GL_TEXTURE0)

        should_draw_travel = self.travel_vertex_count > 0 and (
            executed_travel_vertex_count >= 0 or show_travel_moves
        )
        if should_draw_travel:
            set_line_width(line_width)
            shader.set_float("uUsePowerLUT", 0.0)
            shader.set_float("uUseVertexColor", 0.0)
            shader.set_int(
                "uExecutedVertexCount", executed_travel_vertex_count
            )
            shader.set_vec4("uColor", colors.get_rgba("travel"))
            GL.glBindVertexArray(self.travel_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.travel_vertex_count)

        set_line_width(1.0)
        shader.set_float("uUsePowerLUT", 0.0)
        shader.set_float("uUseVertexColor", 0.0)
        shader.set_int("uExecutedVertexCount", -1)
        shader.set_float("uEmissive", 0.0)
        GL.glBindVertexArray(0)

    def _load_buffer_data(self, vbo: int, data: np.ndarray):
        """Loads vertex data into a VBO."""
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            data.nbytes if data.size > 0 else 0,
            data if data.size > 0 else None,
            GL.GL_DYNAMIC_DRAW,
        )
