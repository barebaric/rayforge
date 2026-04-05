"""
A renderer for a laser beam that appears as a glowing vertical line from
above the workpiece down to the current cutting position.
"""

import logging
import math

import numpy as np
from OpenGL import GL

from .gl_utils import BaseRenderer, Shader

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

    def render(
        self,
        shader: Shader,
        proj_matrix: np.ndarray,
        view_matrix: np.ndarray,
        position: np.ndarray,
        beam_height: float = 50.0,
        viewport_height: int = 800,
        color: tuple = (1.0, 0.3, 0.1, 1.0),
    ):
        if not self.vao:
            return

        p11 = float(proj_matrix[1, 1])
        if abs(p11) < 1e-6:
            return
        is_persp = abs(float(proj_matrix[3, 2])) > 0.1
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

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        shader.use()
        shader.set_float("uHasNormals", 0.0)
        shader.set_float("uUseVertexColor", 0.0)
        shader.set_int("uExecutedVertexCount", -1)
        GL.glBindVertexArray(self.vao)

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

        shader.set_float("uEmissive", 0.0)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
