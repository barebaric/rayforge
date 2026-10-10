"""A standalone GLArea widget that renders a bed height map in 3D.

Draws the probed surface as a vertex-colored triangle grid (turbo
colormap over the Z range) with the probe points as dots and a
wireframe over the control grid. The camera orbits with pointer drag
and zooms with the wheel; the widget is self-contained — no sim3d
scene dependency — reusing only the sim3d ``Shader`` helper for
program compilation and uniform handling.
"""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from gi.repository import Gtk
from OpenGL import GL
from OpenGL.raw.GL.VERSION.GL_2_0 import (
    glVertexAttribPointer as _raw_glVertexAttribPointer,
)

from ..sim3d.shader.base import Shader

VERTEX_SOURCE = """
layout (location = 0) in vec3 aPos;
layout (location = 1) in vec4 aColor;
uniform mat4 uMVP;
out vec4 vColor;
void main() {
    gl_Position = uMVP * vec4(aPos, 1.0);
    vColor = aColor;
}
"""

FRAGMENT_SOURCE = """
out vec4 FragColor;
in vec4 vColor;
void main() {
    FragColor = vColor;
}
"""


def _perspective(fovy: float, aspect: float, near: float, far: float):
    f = 1.0 / math.tan(fovy / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def _look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[:3, 3] = [-np.dot(s, eye), -np.dot(u, eye), np.dot(f, eye)]
    return m


_TURBO_C = (
    (0.11408901, 0.06288341, 0.22483372),
    (6.71641950, 3.18228675, 7.57158159),
    (-66.09402360, -4.92798270, -10.09439368),
    (228.76607915, 25.04986700, -91.54105330),
    (-334.83515658, -69.31749713, 288.58588206),
    (218.76372184, 67.52150568, -305.20457722),
    (-52.88903478, -21.54527365, 110.51768410),
)


def _turbo(t: float) -> tuple[float, float, float]:
    """The turbo colormap (Google polynomial fit), RGB in [0, 1]."""
    t = min(max(t, 0.0), 1.0)
    r = g = b = 0.0
    for c in reversed(_TURBO_C):
        r, g, b = r * t + c[0], g * t + c[1], b * t + c[2]
    return (
        min(max(r, 0.0), 1.0),
        min(max(g, 0.0), 1.0),
        min(max(b, 0.0), 1.0),
    )


class BedMeshView(Gtk.GLArea):
    """Renders a probed bed height map.

    Call :meth:`update_mesh` whenever new (possibly partial) probe
    data is available; ``heights`` entries may be ``None`` while a
    probe run is in progress — incomplete quads are simply skipped.
    """

    #: Vertical exaggeration applied to Z; bed waves are a few mm
    #: over hundreds of mm of travel, invisible at 1:1.
    Z_SCALE = 25.0

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._shader: Shader | None = None
        self._surface_vao = 0
        self._surface_vbo = 0
        self._surface_count = 0
        self._grid_vao = 0
        self._grid_vbo = 0
        self._grid_count = 0
        self._dots_vao = 0
        self._dots_vbo = 0
        self._dots_count = 0
        self._pending: tuple[list, list, list] | None = None
        self._center = np.zeros(3, dtype=np.float64)
        self._radius = 1.0
        self._azimuth = math.radians(-45)
        self._elevation = math.radians(30)
        self._distance = 2.6

        drag = Gtk.GestureDrag()
        drag.connect("drag-update", self._on_drag_update)
        self.add_controller(drag)
        scroll = Gtk.EventControllerScroll()
        scroll.set_flags(Gtk.EventControllerScrollFlags.VERTICAL)
        scroll.connect("scroll", self._on_scroll)
        self.add_controller(scroll)

        self.connect("realize", self._on_realize)
        self.connect("unrealize", self._on_unrealize)
        self.connect("render", self._on_render)
        self.set_size_request(260, 200)

    # ── Public API ────────────────────────────────────────────────

    def update_mesh(
        self,
        x0: float,
        y0: float,
        dx: float,
        dy: float,
        heights: list[list[float | None]],
    ) -> None:
        """Rebuild GPU buffers from (possibly partial) probe data.

        *heights* is ``ny`` rows of ``nx`` values in machine
        coordinates; ``None`` entries are skipped (probe in progress).
        """
        ny = len(heights)
        nx = len(heights[0]) if ny else 0
        values = [z for row in heights for z in row if z is not None]
        if not values:
            return
        zmin, zmax = min(values), max(values)
        span = (zmax - zmin) or 1.0
        zlift = 0.002 * max(
            (nx - 1) * dx if nx > 1 else 1.0,
            (ny - 1) * dy if ny > 1 else 1.0,
        )

        self._center = np.array(
            [
                x0 + (nx - 1) * dx / 2.0 if nx else 0.0,
                y0 + (ny - 1) * dy / 2.0 if ny else 0.0,
                0.0,
            ]
        )
        self._radius = (
            max(
                (nx - 1) * dx if nx > 1 else 1.0,
                (ny - 1) * dy if ny > 1 else 1.0,
                span * self.Z_SCALE,
            )
            / 2.0
        )

        def vertex(i: int, j: int, lift: float = 0.0) -> list[float]:
            z = cast(float, heights[j][i])
            return [
                x0 + i * dx,
                y0 + j * dy,
                (z - zmin) * self.Z_SCALE + lift,
                *_turbo((z - zmin) / span),
                1.0,
            ]

        surface: list[float] = []
        for j in range(ny - 1):
            for i in range(nx - 1):
                corners = ((i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1))
                if any(heights[jj][ii] is None for ii, jj in corners):
                    continue
                a, b, c, d = (vertex(ii, jj) for ii, jj in corners)
                surface.extend(a)
                surface.extend(b)
                surface.extend(c)
                surface.extend(a)
                surface.extend(c)
                surface.extend(d)

        grid: list[float] = []
        dots: list[float] = []
        for j in range(ny):
            for i in range(nx):
                if heights[j][i] is None:
                    continue
                p = vertex(i, j, lift=zlift)
                dots.extend([p[0], p[1], p[2], 0.1, 0.1, 0.1, 1.0])
                if i + 1 < nx and heights[j][i + 1] is not None:
                    q = vertex(i + 1, j, lift=zlift)
                    grid.extend([p[0], p[1], p[2], 0.35, 0.35, 0.35, 1.0])
                    grid.extend([q[0], q[1], q[2], 0.35, 0.35, 0.35, 1.0])
                if j + 1 < ny and heights[j + 1][i] is not None:
                    q = vertex(i, j + 1, lift=zlift)
                    grid.extend([p[0], p[1], p[2], 0.35, 0.35, 0.35, 1.0])
                    grid.extend([q[0], q[1], q[2], 0.35, 0.35, 0.35, 1.0])

        if self._surface_vao:
            self.make_current()
            self._surface_count = self._upload(
                self._surface_vao, self._surface_vbo, surface
            )
            self._grid_count = self._upload(
                self._grid_vao, self._grid_vbo, grid
            )
            self._dots_count = self._upload(
                self._dots_vao, self._dots_vbo, dots
            )
        else:
            self._pending = (surface, grid, dots)
        self.queue_render()

    # ── GL plumbing ───────────────────────────────────────────────

    def _upload(self, vao: int, vbo: int, data: list[float]) -> int:
        count = len(data) // 7
        GL.glBindVertexArray(vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        arr = np.array(data, dtype=np.float32)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
        return count

    def _on_realize(self, *_):
        self.make_current()
        self._shader = Shader(VERTEX_SOURCE, FRAGMENT_SOURCE)
        stride = 7 * np.dtype(np.float32).itemsize
        for suffix in ("surface", "grid", "dots"):
            vao = int(GL.glGenVertexArrays(1))
            vbo = int(GL.glGenBuffers(1))
            setattr(self, f"_{suffix}_vao", vao)
            setattr(self, f"_{suffix}_vbo", vbo)
            GL.glBindVertexArray(vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
            # The wrapped glVertexAttribPointer caches the array in a
            # per-context store keyed by PyOpenGL's static platform
            # (GLX or EGL); that lookup raises when GDK's context API
            # differs from the picked platform. The raw call skips the
            # cache and GLVND dispatches to whichever context is
            # current.
            _raw_glVertexAttribPointer(
                0, 3, GL.GL_FLOAT, GL.GL_FALSE, stride, None
            )
            GL.glEnableVertexAttribArray(0)
            _raw_glVertexAttribPointer(
                1,
                4,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                stride,
                GL.ctypes.c_void_p(3 * np.dtype(np.float32).itemsize),
            )
            GL.glEnableVertexAttribArray(1)
        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        if self._pending:
            surface, grid, dots = self._pending
            self._pending = None
            self._surface_count = self._upload(
                self._surface_vao, self._surface_vbo, surface
            )
            self._grid_count = self._upload(
                self._grid_vao, self._grid_vbo, grid
            )
            self._dots_count = self._upload(
                self._dots_vao, self._dots_vbo, dots
            )

    def _on_unrealize(self, *_):
        self.make_current()
        for suffix in ("surface", "grid", "dots"):
            vao = getattr(self, f"_{suffix}_vao")
            vbo = getattr(self, f"_{suffix}_vbo")
            if vao:
                GL.glDeleteVertexArrays(1, [vao])
                GL.glDeleteBuffers(1, [vbo])
                setattr(self, f"_{suffix}_vao", 0)
        self._shader = None

    def _eye(self) -> np.ndarray:
        dist = self._radius * self._distance
        return self._center + dist * np.array(
            (
                math.cos(self._elevation) * math.cos(self._azimuth),
                math.cos(self._elevation) * math.sin(self._azimuth),
                math.sin(self._elevation),
            )
        )

    def _on_render(self, area, _context):
        if not self._shader or not self._surface_vao:
            return True
        w = self.get_width()
        h = self.get_height()
        GL.glViewport(0, 0, w, h)
        GL.glClearColor(0.06, 0.06, 0.08, 1.0)
        GL.glClear(
            GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT  # type: ignore
        )
        proj = _perspective(math.radians(45), w / max(h, 1), 0.1, 1000.0)
        view = _look_at(self._eye(), self._center, (0.0, 0.0, 1.0))
        mvp = (proj @ view).astype(np.float32)
        self._shader.use()
        self._shader.set_mat4("uMVP", mvp)

        if self._surface_count:
            GL.glBindVertexArray(self._surface_vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._surface_count)
        if self._grid_count:
            GL.glBindVertexArray(self._grid_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_count)
        if self._dots_count:
            GL.glBindVertexArray(self._dots_vao)
            GL.glPointSize(6.0)
            GL.glDrawArrays(GL.GL_POINTS, 0, self._dots_count)
        GL.glBindVertexArray(0)
        return True

    # ── Interaction ───────────────────────────────────────────────

    def _on_drag_update(self, _gesture, dx: float, dy: float):
        self._azimuth += dx * 0.01
        self._elevation = min(
            max(self._elevation + dy * 0.01, 0.05), math.pi / 2 - 0.05
        )
        self.queue_render()

    def _on_scroll(self, _ctrl, _dx, dy: float) -> bool:
        self._distance = min(max(self._distance * (1.0 + dy * 0.1), 1.2), 8.0)
        self.queue_render()
        return True
