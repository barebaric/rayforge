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

from ....simulator.scene3d import ScanlineOverlayLayer
from ...shared.color_lut_provider import ColorLutProvider
from ..gl_utils import ShaderSet, set_line_width
from ..render_context import RenderContext
from .base import BaseRenderer


class RingBufferRenderer(BaseRenderer):
    """
    Renders scanline line-segments from a fixed-size ring buffer.

    The caller encodes powered pixel-segments for each ScanLinePowerCommand
    and appends them in command-index order.  ``render()`` draws only the
    first *n* vertices, where *n* corresponds to the playhead position.
    """

    def __init__(
        self, capacity_vertices: int = 4_000_000, is_rotary: bool = False
    ):
        super().__init__()
        self._capacity = capacity_vertices
        self.is_rotary = is_rotary
        self.vao: int = 0
        self.pos_vbo: int = 0
        self.pow_vbo: int = 0
        self.vertex_count: int = 0
        self.ring_offsets: np.ndarray = np.array([], dtype=np.int32)
        self._positions: np.ndarray = np.array([], dtype=np.float32)
        self._exec_ring = -1
        self._partial_ring_id = -1
        self._partial_ring_end = np.zeros(3, dtype=np.float32)
        self._color_lut_texture: int = 0
        self._num_laser_luts: int = 1

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

    def upload(
        self,
        positions: np.ndarray,
        attrib: np.ndarray,
    ):
        pos = np.ascontiguousarray(positions, dtype=np.float32).ravel()
        n = pos.size // 3
        assert n <= self._capacity, (
            f"Scanline overlay has {n} vertices but ring capacity is "
            f"{self._capacity}"
        )

        self._positions = pos
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pos_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, pos.nbytes, pos)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.pow_vbo)
        a = np.ascontiguousarray(attrib, dtype=np.float32)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, a.nbytes, a)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        self.vertex_count = n

    def update_from_overlay_layer(self, ol: ScanlineOverlayLayer):
        """Uploads a compiled scanline overlay layer into the ring buffer."""
        positions = ol.positions.to_numpy()
        attrib = ol.overlay_attrib.to_numpy()
        self.update_from_overlay_layer_payload(positions, attrib)

    def update_from_overlay_layer_payload(
        self, positions: np.ndarray, attrib: np.ndarray
    ):
        """Uploads pre-decompressed overlay arrays into the ring buffer."""
        self.upload(positions.ravel(), attrib)

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
        self.update_color_lut(provider.ring_lut_2d(), provider.num_lasers)

    def clear(self):
        self.vertex_count = 0

    def prepare(self, ctx: RenderContext) -> None:
        """
        Computes the executed-vertex count for this frame.

        Reads the playhead from ``ctx.playback.op_player`` and maps it
        through the renderer's command offsets, stashing the resulting
        count so ``render`` can publish it back into ``ctx``.  When the
        playhead falls inside a command, a fractional executed count is
        split into an int count plus a partial boundary segment for a
        smooth reveal.
        """
        exec_ring = -1
        self._partial_ring_id = -1
        self._partial_ring_end = np.zeros(3, dtype=np.float32)
        op_player = ctx.playback.op_player
        if op_player:
            p, frac = op_player.playback_progress()
            exec_ring, self._partial_ring_id, self._partial_ring_end = (
                self._fractional_exec_count(
                    self.ring_offsets, self._positions, p, frac
                )
            )
        self._exec_ring = exec_ring

    @staticmethod
    def _fractional_exec_count(offsets, positions, p, frac):
        """Map ``(in_progress_command, fraction)`` to executed vertices.

        Returns ``(executed_count, partial_vertex_id, partial_end)``;
        see :meth:`OpsRenderer._fractional_exec_count` for details.
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

    def render(self, ctx: RenderContext, shaders: ShaderSet, **kwargs):
        if self.vertex_count == 0:
            return

        ctx.playback.executed_vertex_count = self._exec_ring

        shader = shaders.main
        if shader is None:
            return

        mvp = ctx.kinematics.mvp_for(self.is_rotary)
        if mvp is None:
            return

        draw_count = self.vertex_count
        executed_vertex_count = ctx.playback.executed_vertex_count

        line_width = ctx.camera.line_width
        shader.use()
        shader.set_mat4("uMVP", mvp)
        shader.set_float("uHasNormals", 0.0)
        shader.set_float("uUsePowerLUT", 1.0)
        shader.set_int("uNumLaserLUTs", self._num_laser_luts)
        shader.set_vec4(
            "uZeroPowerColor", ctx.camera.color_set.get_rgba("zero_power")
        )
        shader.set_int("uExecutedVertexCount", executed_vertex_count)
        shader.set_float("uAlphaPending", ctx.playback.alpha_pending)
        if self._partial_ring_id >= 0:
            shader.set_int("uPartialVertexID", self._partial_ring_id)
            shader.set_vec3("uPartialEnd", self._partial_ring_end)
        else:
            shader.set_int("uPartialVertexID", -1)
            shader.set_vec3("uPartialEnd", (0.0, 0.0, 0.0))

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._color_lut_texture)
        shader.set_int("uColorLUT", 1)

        # The scanline trail must always draw on top of the toolpath and
        # the raster texture; never cull it by surface depth (which would
        # let travel lines or the texture's depth split the trail on a
        # cylinder).  Depth writes stay off so later geometry is unaffected.
        GL.glDepthFunc(GL.GL_ALWAYS)
        GL.glDepthMask(GL.GL_FALSE)
        set_line_width(line_width)
        GL.glBindVertexArray(self.vao)
        GL.glDrawArrays(GL.GL_LINES, 0, draw_count)
