"""
A renderer for visualizing toolpath operations (Ops) in 3D.
"""

import logging
import numpy as np
from OpenGL import GL
from ...shared.util.colors import ColorSet
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class OpsRenderer(BaseRenderer):
    """Renders toolpath operations (cuts and travels) as colored lines."""

    def __init__(self):
        """Initializes the OpsRenderer."""
        super().__init__()
        self.powered_vao: int = 0
        self.travel_vao: int = 0

        self.powered_vbo: int = 0
        self.powered_colors_vbo: int = 0
        self.powered_index_vbo: int = 0
        self.travel_vbo: int = 0

        self.powered_vertex_count: int = 0
        self.travel_vertex_count: int = 0

    def init_gl(self):
        self.powered_vbo = self._create_vbo()
        self.powered_colors_vbo = self._create_vbo()
        self.powered_index_vbo = self._create_vbo()
        self.travel_vbo = self._create_vbo()

        self.powered_vao = self._create_vao()
        GL.glBindVertexArray(self.powered_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_colors_vbo)
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
        powered_colors: np.ndarray,
        travel_vertices: np.ndarray,
    ):
        """Receives pre-processed vertex data and uploads it to the GPU."""
        self.powered_vertex_count = powered_vertices.size // 3
        self._load_buffer_data(self.powered_vbo, powered_vertices)
        self._load_buffer_data(self.powered_colors_vbo, powered_colors)
        indices = np.arange(self.powered_vertex_count, dtype=np.float32)
        self._load_buffer_data(self.powered_index_vbo, indices)
        self.travel_vertex_count = travel_vertices.size // 3
        self._load_buffer_data(self.travel_vbo, travel_vertices)

    def render(
        self,
        shader: Shader,
        mvp_matrix: np.ndarray,
        colors: ColorSet,
        show_travel_moves: bool,
        executed_vertex_count: int = -1,
        alpha_pending: float = 0.2,
        executed_travel_vertex_count: int = -1,
    ) -> None:
        """
        Renders the toolpaths. The vertices are assumed to be in world space.

        Args:
            shader: The shader program to use for rendering lines.
            mvp_matrix: The combined Model-View-Projection matrix.
            colors: The resolved ColorSet containing color data.
            show_travel_moves: Whether to render the travel move paths.
            executed_vertex_count: Powered vertices drawn at full alpha.
                -1 means all vertices at full alpha (IDLE mode).
            alpha_pending: Alpha multiplier for pending vertices (0..1).
            executed_travel_vertex_count: Travel vertices to draw
                progressively during simulation. -1 means use
                show_travel_moves instead.
        """
        if executed_vertex_count > self.powered_vertex_count:
            raise ValueError(
                f"executed_vertex_count ({executed_vertex_count}) "
                f"> powered_vertex_count ({self.powered_vertex_count})"
            )

        logger.debug(
            f"[RENDER] powered={self.powered_vertex_count} "
            f"travel={self.travel_vertex_count} "
            f"exec={executed_vertex_count} "
            f"travel_exec={executed_travel_vertex_count}"
        )

        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)
        shader.set_float("uHasNormals", 0.0)

        logger.debug(
            f"[RENDER] powered={self.powered_vertex_count} "
            f"travel={self.travel_vertex_count} "
            f"exec={executed_vertex_count} "
            f"travel_exec={executed_travel_vertex_count}"
        )

        shader.set_int("uExecutedVertexCount", executed_vertex_count)
        shader.set_float("uAlphaPending", alpha_pending)

        # Draw powered moves (which use vertex colors)
        if self.powered_vertex_count > 0:
            shader.set_float("uUseVertexColor", 1.0)
            GL.glBindVertexArray(self.powered_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.powered_vertex_count)

        # Draw travel moves
        travel_draw = 0
        if executed_travel_vertex_count >= 0:
            travel_draw = min(
                executed_travel_vertex_count, self.travel_vertex_count
            )
        elif show_travel_moves:
            travel_draw = self.travel_vertex_count

        if travel_draw > 0:
            shader.set_float("uUseVertexColor", 0.0)
            shader.set_int("uExecutedVertexCount", -1)
            shader.set_vec4("uColor", colors.get_rgba("travel"))
            GL.glBindVertexArray(self.travel_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, travel_draw)

        shader.set_float("uUseVertexColor", 0.0)
        shader.set_int("uExecutedVertexCount", -1)
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
