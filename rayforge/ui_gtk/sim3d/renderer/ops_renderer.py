"""
A renderer for visualizing toolpath operations (Ops) in 3D.
"""

import logging

import numpy as np
from OpenGL import GL

from ....simulator.scene3d import VertexLayer
from ...shared.color_lut_provider import ColorLutProvider
from ..gl_utils import ShaderSet, set_line_width
from ..render_context import RenderContext
from .base import BaseRenderer

logger = logging.getLogger(__name__)


class OpsRenderer(BaseRenderer):
    """Renders toolpath operations (cuts and travels) as colored lines."""

    def __init__(self, is_rotary: bool = False):
        """Initializes the OpsRenderer."""
        super().__init__()
        self.is_rotary = is_rotary
        self.powered_vao: int = 0
        self.travel_vao: int = 0

        self.powered_vbo: int = 0
        self.powered_powers_vbo: int = 0
        self.travel_vbo: int = 0

        self.powered_vertex_count: int = 0
        self.travel_vertex_count: int = 0

        self.powered_offsets: np.ndarray = np.array([], dtype=np.int32)
        self.travel_offsets: np.ndarray = np.array([], dtype=np.int32)
        self._powered_positions: np.ndarray = np.array([], dtype=np.float32)
        self._travel_positions: np.ndarray = np.array([], dtype=np.float32)
        self._exec_powered = -1
        self._exec_travel = -1
        self._partial_powered_id = -1
        self._partial_powered_end = np.zeros(3, dtype=np.float32)
        self._partial_travel_id = -1
        self._partial_travel_end = np.zeros(3, dtype=np.float32)

        self._color_lut_texture: int = 0
        self._num_laser_luts: int = 1

    def init_gl(self):
        self.powered_vbo = self._create_vbo()
        self.powered_powers_vbo = self._create_vbo()
        self.travel_vbo = self._create_vbo()
        self._color_lut_texture = self._create_texture()

        self.powered_vao = self._create_vao()
        GL.glBindVertexArray(self.powered_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_vbo)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.powered_powers_vbo)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(1)

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

    def update_from_vertex_layer(
        self, vl: VertexLayer, show_travel_moves: bool
    ):
        """Uploads a compiled vertex layer into the renderer's buffers."""
        powered_verts = vl.powered_verts.to_numpy()
        powered_attrib = vl.powered_attrib.to_numpy()
        travel_verts = vl.travel_verts.to_numpy()
        zero_power_verts = vl.zero_power_verts.to_numpy()

        if show_travel_moves:
            pv_final = np.concatenate((powered_verts, zero_power_verts))
            zero_count = zero_power_verts.size // 3
            zero_attrib = np.zeros(zero_count * 4, dtype=np.float32)
            zero_attrib[3::4] = 1.0
            attrib = np.concatenate((powered_attrib.ravel(), zero_attrib))
            tv_final = travel_verts
        else:
            pv_final = powered_verts
            attrib = powered_attrib
            tv_final = np.array([], dtype=np.float32)

        powered_count = powered_verts.size // 3
        zero_count = zero_power_verts.size // 3
        logger.debug(
            f"[UPLOAD] is_rotary={vl.is_rotary} "
            f"powered={powered_count} "
            f"zero_power={zero_count} "
            f"total={pv_final.size // 3} "
            f"travel={tv_final.size // 3} "
            f"show_travel={show_travel_moves}"
        )

        self.update_from_vertex_data(pv_final, attrib, tv_final)

    def update_from_vertex_data(
        self,
        powered_vertices: np.ndarray,
        powered_attrib: np.ndarray,
        travel_vertices: np.ndarray,
    ):
        self.powered_vertex_count = powered_vertices.size // 3
        self._powered_positions = np.ascontiguousarray(
            powered_vertices, dtype=np.float32
        )
        self._load_buffer_data(self.powered_vbo, powered_vertices)
        self._load_buffer_data(
            self.powered_powers_vbo,
            np.ascontiguousarray(powered_attrib, dtype=np.float32),
        )
        self.travel_vertex_count = travel_vertices.size // 3
        self._travel_positions = np.ascontiguousarray(
            travel_vertices, dtype=np.float32
        )
        self._load_buffer_data(self.travel_vbo, travel_vertices)

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

    def update_color_lut_from(self, provider: ColorLutProvider):
        """Updates the colour LUT from a shared ColorLutProvider."""
        self.update_color_lut(provider.cut_lut(), provider.num_lasers)

    def prepare(self, ctx: RenderContext) -> None:
        """
        Computes the executed-vertex counts for this frame.

        Reads the playhead from ``ctx.playback.op_player`` and maps it
        through the renderer's command offsets, stashing the resulting
        counts so ``render`` can publish them back into ``ctx``.  When
        the playhead falls inside a command, a fractional executed count
        is split into an int count plus a partial boundary segment (the
        moved end vertex + interpolated endpoint) for a smooth reveal.
        """
        exec_powered = -1
        exec_travel = -1
        self._partial_powered_id = -1
        self._partial_powered_end = np.zeros(3, dtype=np.float32)
        self._partial_travel_id = -1
        self._partial_travel_end = np.zeros(3, dtype=np.float32)
        op_player = ctx.playback.op_player
        if op_player:
            p, frac = self._playback_progress(op_player)
            (
                exec_powered,
                self._partial_powered_id,
                self._partial_powered_end,
            ) = self._fractional_exec_count(
                self.powered_offsets,
                self._powered_positions,
                p,
                frac,
            )
            exec_travel, self._partial_travel_id, self._partial_travel_end = (
                self._fractional_exec_count(
                    self.travel_offsets,
                    self._travel_positions,
                    p,
                    frac,
                )
            )
        self._exec_powered = exec_powered
        self._exec_travel = exec_travel

    @staticmethod
    def _playback_progress(op_player):
        """Return ``(in_progress_command, fraction)`` from the player.

        Falls back to the next command with no fraction for minimal
        player stubs used in tests.
        """
        prog = getattr(op_player, "playback_progress", None)
        if prog is not None:
            return prog()
        return op_player.current_index + 1, 0.0

    @staticmethod
    def _fractional_exec_count(
        offsets: np.ndarray,
        positions: np.ndarray,
        p: int,
        frac: float,
    ):
        """Map ``(in_progress_command, fraction)`` to executed vertices.

        Returns ``(executed_count, partial_vertex_id, partial_end)``.
        ``executed_count`` is the int uniform value (fully drawn
        vertices plus the boundary segment); ``partial_vertex_id`` is
        the end vertex of the boundary segment that ``render`` moves to
        ``partial_end`` (or -1 when no partial segment exists).
        """
        if len(offsets) == 0:
            return -1, -1, np.zeros(3, dtype=np.float32)
        total = positions.size // 3 if positions is not None else 0
        if len(offsets) < 2:
            return total, -1, np.zeros(3, dtype=np.float32)
        if p + 1 >= len(offsets):
            p = len(offsets) - 2
            frac = 1.0
        p = max(p, 0)
        base = int(offsets[p])
        span = int(offsets[p + 1]) - base
        exec_f = base + frac * span
        zero = np.zeros(3, dtype=np.float32)
        if total == 0:
            # No position data uploaded: fall back to the raw count.
            return int(exec_f), -1, zero
        if exec_f >= total:
            return total, -1, zero
        if exec_f <= 0:
            return 0, -1, zero
        seg = int(exec_f) // 2
        f_in_seg = exec_f - 2 * seg
        if f_in_seg <= 1e-9:
            return 2 * seg, -1, zero
        if positions is None or 2 * seg + 1 >= total:
            return 2 * seg + 2, -1, zero
        v0 = positions[2 * seg * 3 : 2 * seg * 3 + 3]
        v1 = positions[(2 * seg + 1) * 3 : (2 * seg + 1) * 3 + 3]
        partial_end = (v0 + (v1 - v0) * (f_in_seg / 2.0)).astype(np.float32)
        return 2 * seg + 2, 2 * seg + 1, partial_end

    def render(self, ctx: RenderContext, shaders: ShaderSet, **kwargs) -> None:
        """
        Renders the toolpaths. The vertices are assumed to be in world space.

        Publishes the executed-vertex counts computed in ``prepare`` into
        ``ctx``, then pulls the MVP and pending alpha from it.

        Args:
            ctx: The current render context (carries color set, line width,
              travel-move visibility, and per-frame execution state).
            shaders: The shader set; the ``main`` program is used.
        """
        ctx.playback.executed_vertex_count = self._exec_powered
        ctx.playback.executed_travel_vertex_count = self._exec_travel

        shader = shaders.main
        if shader is None:
            return

        mvp = ctx.kinematics.mvp_for(self.is_rotary)
        if mvp is None:
            return

        colors = ctx.camera.color_set
        show_travel_moves = ctx.camera.show_travel_moves
        line_width = ctx.camera.line_width
        executed_vertex_count = ctx.playback.executed_vertex_count
        executed_travel_vertex_count = (
            ctx.playback.executed_travel_vertex_count
        )
        alpha_pending = ctx.playback.alpha_pending

        if executed_vertex_count > self.powered_vertex_count:
            raise ValueError(
                f"executed_vertex_count ({executed_vertex_count}) "
                f"> powered_vertex_count ({self.powered_vertex_count})"
            )

        shader.use()
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        # The toolpath draws on top of the raster texture; never cull it
        # by the surface depth (which would split lines on a cylinder's
        # curved face).  Depth writes stay off so the trail/ring drawn
        # afterwards is unaffected.
        GL.glDepthMask(GL.GL_FALSE)
        GL.glDepthFunc(GL.GL_ALWAYS)
        shader.set_mat4("uMVP", mvp)
        shader.set_float("uHasNormals", 0.0)

        shader.set_int("uExecutedVertexCount", executed_vertex_count)
        shader.set_float("uAlphaPending", alpha_pending)
        shader.set_float("uEmissive", 1.0)
        if self._partial_powered_id >= 0:
            shader.set_int("uPartialVertexID", self._partial_powered_id)
            shader.set_vec3("uPartialEnd", self._partial_powered_end)
        else:
            shader.set_int("uPartialVertexID", -1)
            shader.set_vec3("uPartialEnd", (0.0, 0.0, 0.0))

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

        should_draw_travel = self.travel_vertex_count > 0 and (
            executed_travel_vertex_count >= 0 or show_travel_moves
        )
        if should_draw_travel:
            set_line_width(line_width)
            shader.set_float("uUsePowerLUT", 0.0)
            shader.set_float("uUseVertexColor", 0.0)
            shader.set_float("uEmissive", 0.0)
            shader.set_int(
                "uExecutedVertexCount", executed_travel_vertex_count
            )
            if self._partial_travel_id >= 0:
                shader.set_int("uPartialVertexID", self._partial_travel_id)
                shader.set_vec3("uPartialEnd", self._partial_travel_end)
            else:
                shader.set_int("uPartialVertexID", -1)
                shader.set_vec3("uPartialEnd", (0.0, 0.0, 0.0))
            shader.set_vec4("uColor", colors.get_rgba("travel"))
            GL.glBindVertexArray(self.travel_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self.travel_vertex_count)

    def _load_buffer_data(self, vbo: int, data: np.ndarray):
        """Loads vertex data into a VBO."""
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            data.nbytes if data.size > 0 else 0,
            data if data.size > 0 else None,
            GL.GL_DYNAMIC_DRAW,
        )
