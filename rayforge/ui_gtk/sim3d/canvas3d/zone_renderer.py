import logging
import math
from typing import List, Tuple

import numpy as np
from OpenGL import GL

from ....machine.models.zone import Zone, ZoneShape
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)

_DEFAULT_FILL_COLOR = (1.0, 0.0, 0.0, 0.05)
_DEFAULT_EDGE_COLOR = (1.0, 0.0, 0.0, 0.2)
_CYLINDER_SEGMENTS = 16


def _rect_triangles(p: dict) -> List[float]:
    x, y = p.get("x", 0.0), p.get("y", 0.0)
    w, h = p.get("w", 10.0), p.get("h", 10.0)
    z = 0.003
    return [
        x,
        y,
        z,
        x + w,
        y,
        z,
        x,
        y + h,
        z,
        x + w,
        y,
        z,
        x + w,
        y + h,
        z,
        x,
        y + h,
        z,
    ]


def _rect_edges(p: dict) -> List[float]:
    x, y = p.get("x", 0.0), p.get("y", 0.0)
    w, h = p.get("w", 10.0), p.get("h", 10.0)
    z = 0.004
    return [
        x,
        y,
        z,
        x + w,
        y,
        z,
        x + w,
        y,
        z,
        x + w,
        y + h,
        z,
        x + w,
        y + h,
        z,
        x,
        y + h,
        z,
        x,
        y + h,
        z,
        x,
        y,
        z,
    ]


def _box_triangles(p: dict) -> List[float]:
    x, y, z = p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)
    w, h, d = p.get("w", 10.0), p.get("h", 10.0), p.get("d", 10.0)
    x2, y2, z2 = x + w, y + h, z + d
    faces = [
        [x, y, z, x2, y, z, x, y2, z, x2, y, z, x2, y2, z, x, y2, z],
        [x, y, z2, x2, y, z2, x, y2, z2, x2, y, z2, x2, y2, z2, x, y2, z2],
        [x, y, z, x, y2, z, x, y, z2, x, y2, z, x, y2, z2, x, y, z2],
        [x2, y, z, x2, y2, z, x2, y, z2, x2, y2, z, x2, y2, z2, x2, y, z2],
        [x, y, z, x2, y, z, x, y, z2, x2, y, z, x2, y, z2, x, y, z2],
        [x, y2, z, x2, y2, z, x, y2, z2, x2, y2, z, x2, y2, z2, x, y2, z2],
    ]
    verts = []
    for face in faces:
        verts.extend(face)
    return verts


def _box_edges(p: dict) -> List[float]:
    x, y, z = p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)
    w, h, d = p.get("w", 10.0), p.get("h", 10.0), p.get("d", 10.0)
    x2, y2, z2 = x + w, y + h, z + d
    return [
        x,
        y,
        z,
        x2,
        y,
        z,
        x2,
        y,
        z,
        x2,
        y2,
        z,
        x2,
        y2,
        z,
        x,
        y2,
        z,
        x,
        y2,
        z,
        x,
        y,
        z,
        x,
        y,
        z2,
        x2,
        y,
        z2,
        x2,
        y,
        z2,
        x2,
        y2,
        z2,
        x2,
        y2,
        z2,
        x,
        y2,
        z2,
        x,
        y2,
        z2,
        x,
        y,
        z2,
        x,
        y,
        z,
        x,
        y,
        z2,
        x2,
        y,
        z,
        x2,
        y,
        z2,
        x2,
        y2,
        z,
        x2,
        y2,
        z2,
        x,
        y2,
        z,
        x,
        y2,
        z2,
    ]


def _cylinder_triangles(p: dict) -> List[float]:
    cx, cy = p.get("x", 0.0), p.get("y", 0.0)
    cz = p.get("z", 0.0)
    radius = p.get("radius", 5.0)
    height = p.get("height", 10.0)
    n = _CYLINDER_SEGMENTS
    verts: List[float] = []
    for i in range(n):
        a1 = 2.0 * math.pi * i / n
        a2 = 2.0 * math.pi * (i + 1) / n
        c1x = cx + radius * math.cos(a1)
        c1y = cy + radius * math.sin(a1)
        c2x = cx + radius * math.cos(a2)
        c2y = cy + radius * math.sin(a2)
        verts.extend(
            [
                cx,
                cy,
                cz,
                c1x,
                c1y,
                cz,
                c2x,
                c2y,
                cz,
            ]
        )
        zt = cz + height
        verts.extend(
            [
                cx,
                cy,
                zt,
                c2x,
                c2y,
                zt,
                c1x,
                c1y,
                zt,
            ]
        )
        verts.extend(
            [
                c1x,
                c1y,
                cz,
                c2x,
                c2y,
                cz,
                c2x,
                c2y,
                zt,
            ]
        )
        verts.extend(
            [
                c1x,
                c1y,
                cz,
                c2x,
                c2y,
                zt,
                c1x,
                c1y,
                zt,
            ]
        )
    return verts


def _cylinder_edges(p: dict) -> List[float]:
    cx, cy = p.get("x", 0.0), p.get("y", 0.0)
    cz = p.get("z", 0.0)
    radius = p.get("radius", 5.0)
    height = p.get("height", 10.0)
    n = _CYLINDER_SEGMENTS
    verts: List[float] = []
    for i in range(n):
        a1 = 2.0 * math.pi * i / n
        a2 = 2.0 * math.pi * (i + 1) / n
        c1x = cx + radius * math.cos(a1)
        c1y = cy + radius * math.sin(a1)
        c2x = cx + radius * math.cos(a2)
        c2y = cy + radius * math.sin(a2)
        verts.extend([c1x, c1y, cz, c2x, c2y, cz])
        zt = cz + height
        verts.extend([c1x, c1y, zt, c2x, c2y, zt])
        verts.extend([c1x, c1y, cz, c1x, c1y, zt])
    return verts


_TRIANGLE_FUNCS = {
    ZoneShape.RECT: _rect_triangles,
    ZoneShape.BOX: _box_triangles,
    ZoneShape.CYLINDER: _cylinder_triangles,
}

_EDGE_FUNCS = {
    ZoneShape.RECT: _rect_edges,
    ZoneShape.BOX: _box_edges,
    ZoneShape.CYLINDER: _cylinder_edges,
}


class ZoneRenderer(BaseRenderer):
    def __init__(self):
        super().__init__()
        self._fill_vao = 0
        self._fill_vbo = 0
        self._fill_vertex_count = 0
        self._edge_vao = 0
        self._edge_vbo = 0
        self._edge_vertex_count = 0
        self._fill_color: Tuple[float, ...] = _DEFAULT_FILL_COLOR
        self._edge_color: Tuple[float, ...] = _DEFAULT_EDGE_COLOR

    def update_zones(self, zones: List[Zone]):
        fill_verts: List[float] = []
        edge_verts: List[float] = []
        for zone in zones:
            if not zone.enabled:
                continue
            tri_fn = _TRIANGLE_FUNCS.get(zone.shape)
            edge_fn = _EDGE_FUNCS.get(zone.shape)
            if tri_fn:
                fill_verts.extend(tri_fn(zone.params))
            if edge_fn:
                edge_verts.extend(edge_fn(zone.params))

        if self._fill_vao:
            GL.glDeleteVertexArrays(1, [self._fill_vao])
            self._fill_vao = 0
        if self._fill_vbo:
            GL.glDeleteBuffers(1, [self._fill_vbo])
            self._fill_vbo = 0
        if self._edge_vao:
            GL.glDeleteVertexArrays(1, [self._edge_vao])
            self._edge_vao = 0
        if self._edge_vbo:
            GL.glDeleteBuffers(1, [self._edge_vbo])
            self._edge_vbo = 0

        if fill_verts:
            self._fill_vao = self._create_vao()
            self._fill_vbo = self._create_vbo()
            self._fill_vertex_count = len(fill_verts) // 3
            data = np.array(fill_verts, dtype=np.float32)
            GL.glBindVertexArray(self._fill_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._fill_vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW
            )
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)

        if edge_verts:
            self._edge_vao = self._create_vao()
            self._edge_vbo = self._create_vbo()
            self._edge_vertex_count = len(edge_verts) // 3
            data = np.array(edge_verts, dtype=np.float32)
            GL.glBindVertexArray(self._edge_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._edge_vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW
            )
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glEnableVertexAttribArray(0)

        GL.glBindVertexArray(0)

    def init_gl(self) -> None:
        pass

    def render(self, shader: Shader, mvp: np.ndarray) -> None:
        if not self._fill_vao and not self._edge_vao:
            return

        shader.use()
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        shader.set_float("uHasNormals", 0.0)
        shader.set_float("uUseVertexColor", 0.0)

        if self._fill_vao:
            GL.glDepthMask(GL.GL_FALSE)
            shader.set_mat4("uMVP", mvp)
            shader.set_vec4("uColor", self._fill_color)
            GL.glBindVertexArray(self._fill_vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._fill_vertex_count)
            GL.glDepthMask(GL.GL_TRUE)

        if self._edge_vao:
            shader.set_mat4("uMVP", mvp)
            shader.set_vec4("uColor", self._edge_color)
            GL.glBindVertexArray(self._edge_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self._edge_vertex_count)

        GL.glBindVertexArray(0)
