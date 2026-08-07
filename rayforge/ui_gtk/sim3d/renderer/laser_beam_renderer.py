"""
A renderer for a laser beam that appears as a glowing vertical line from
above the workpiece down to the current cutting position.
"""

import logging
import math
from typing import List, Optional, Tuple

import numpy as np
from OpenGL import GL

from ....core.color import hex_to_rgba
from ....machine.models.laser import LaserHead
from ..gl_utils import RenderContext, ShaderSet
from .base import BaseRenderer

logger = logging.getLogger(__name__)

SEGMENTS = 16


def _build_cylinder_verts():
    verts = []
    for i in range(SEGMENTS):
        a0 = 2.0 * math.pi * i / SEGMENTS
        a1 = 2.0 * math.pi * (i + 1) / SEGMENTS
        c0, s0 = math.cos(a0), math.sin(a0)
        c1, s1 = math.cos(a1), math.sin(a1)
        verts.extend([c0, s0, 0.0])
        verts.extend([c1, s1, 0.0])
        verts.extend([c0, s0, 1.0])
        verts.extend([c1, s1, 1.0])
        verts.extend([c1, s1, 0.0])
        verts.extend([c0, s0, 1.0])
    return verts


def _build_disc_verts(z):
    verts = []
    for i in range(SEGMENTS):
        a0 = 2.0 * math.pi * i / SEGMENTS
        a1 = 2.0 * math.pi * (i + 1) / SEGMENTS
        verts.extend([0.0, 0.0, z])
        verts.extend([math.cos(a0), math.sin(a0), z])
        verts.extend([math.cos(a1), math.sin(a1), z])
    return verts


class LaserBeamRenderer(BaseRenderer):
    """Renders a glowing laser beam as a world-space cylinder with caps."""

    def __init__(self):
        super().__init__()
        self.vao: int = 0
        self.vbo: int = 0
        self.vertex_count: int = 0
        self._beams: List[Tuple[np.ndarray, float, tuple]] = []
        self.laser_light_pos: Optional[np.ndarray] = None

    def init_gl(self):
        self.vao = self._create_vao()
        self.vbo = self._create_vbo()

        verts = _build_cylinder_verts()
        verts += _build_disc_verts(0.0)
        verts += _build_disc_verts(1.0)

        self.vertex_count = len(verts) // 3
        data = np.array(verts, dtype=np.float32)

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def prepare(self, ctx: RenderContext) -> None:
        """Computes and caches the laser beams from the current state."""
        self._beams = []
        self.laser_light_pos = None

        op_player = ctx.op_player
        machine = ctx.machine
        if op_player is None or machine is None:
            ctx.laser_light_pos = None
            return

        state = op_player.state
        viewport = ctx.viewport
        if viewport is None:
            ctx.laser_light_pos = None
            return

        ra = ctx.rotary_axis
        doc = ctx.doc
        margin_shift = ctx.margin_shift

        asm = machine.assembly
        wcs = viewport.wcs_offset_mm
        heads = asm.head_positions(state, wcs_offset=wcs)
        vis_mat = margin_shift.astype(np.float32)
        for name, (hx, hy, hz) in heads.items():
            head_pos = vis_mat @ np.array([hx, hy, hz, 1.0], dtype=np.float32)
            beam_height = 50.0
            beam_color = (1.0, 0.3, 0.1, 1.0)
            if name.startswith("head_"):
                try:
                    idx = int(name.split("_")[1])
                    laser = machine.heads[idx]
                    if not isinstance(laser, LaserHead):
                        continue
                    if laser.focal_distance > 0:
                        beam_height = laser.focal_distance
                    beam_color = hex_to_rgba(laser.cut_color)
                except (ValueError, IndexError):
                    pass
            if not state.laser_on:
                continue
            if ra is not None and asm.has_rotary:
                current_layer = (
                    op_player.get_current_layer(doc)
                    if doc is not None
                    else None
                )
                diameter = (
                    current_layer.rotary_diameter if current_layer else 0.0
                )
                rotary_heads = asm.head_rotary_positions(state, diameter)
                if name in rotary_heads:
                    beam_pos = vis_mat @ np.array(
                        [*rotary_heads[name], 1.0], dtype=np.float32
                    )
                else:
                    beam_pos = head_pos.copy()
            else:
                beam_pos = head_pos.copy()
            self._beams.append((beam_pos[:3], beam_height, beam_color))
            self.laser_light_pos = beam_pos[:3].astype(np.float32)

        ctx.laser_light_pos = self.laser_light_pos

    def render(self, ctx: RenderContext, shaders: ShaderSet):
        if not self.vao:
            return

        shader = shaders.main
        if shader is None:
            return

        proj_matrix = ctx.proj_matrix
        view_matrix = ctx.view_matrix
        viewport_height = ctx.viewport_height

        p11 = float(proj_matrix[1, 1])
        if abs(p11) < 1e-6:
            return
        is_persp = abs(float(proj_matrix[3, 2])) > 0.1

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        shader.use()
        shader.set_float("uHasNormals", 0.0)
        shader.set_float("uUseVertexColor", 0.0)
        shader.set_int("uExecutedVertexCount", -1)
        GL.glBindVertexArray(self.vao)

        for position, beam_height, color in self._beams:
            if is_persp:
                view_pos = view_matrix.astype(np.float64) @ np.array(
                    [
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                        1.0,
                    ],
                    dtype=np.float64,
                )
                depth = max(-view_pos[2], 0.1)
                wpp = 2.0 * depth / (p11 * max(viewport_height, 1))
            else:
                wpp = 2.0 / (p11 * max(viewport_height, 1))

            cr, cg, cb = color[:3]
            wr = min(cr * 0.5 + 0.5, 1.0)
            wg = min(cg * 0.5 + 0.5, 1.0)
            wb = min(cb * 0.5 + 0.5, 1.0)

            num_passes = 16
            for i in range(num_passes, 0, -1):
                t = i / num_passes
                radius_px = 0.5 + t * 10.0
                alpha = 0.08 * (1.0 - t) ** 2
                pass_color = (
                    wr + (1.0 - wr) * (1.0 - t),
                    wg + (1.0 - wg) * (1.0 - t),
                    wb + (1.0 - wb) * (1.0 - t),
                    alpha,
                )

                r = radius_px * wpp
                model = np.eye(4, dtype=np.float32)
                model[0, 0] = np.float32(r)
                model[1, 1] = np.float32(r)
                model[2, 2] = np.float32(beam_height)
                model[0, 3] = np.float32(position[0])
                model[1, 3] = np.float32(position[1])
                model[2, 3] = np.float32(position[2])

                mvp = (proj_matrix @ view_matrix @ model).T
                shader.set_mat4("uMVP", mvp)
                shader.set_float("uEmissive", 1.0)
                shader.set_vec4("uColor", pass_color)
                GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE)
                GL.glDrawArrays(GL.GL_TRIANGLES, 0, self.vertex_count)
