"""Standalone 3D model preview widget for Rayforge settings."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from gi.repository import Gtk
from OpenGL import GL

from ..sim3d.canvas3d.camera import Camera
from ..sim3d.canvas3d.gl_utils import BaseRenderer, Shader
from ..sim3d.canvas3d.model_renderer import _load_mesh_data
from ..sim3d.canvas3d.shaders import (
    SIMPLE_FRAGMENT_SHADER,
    SIMPLE_VERTEX_SHADER,
)

logger = logging.getLogger(__name__)


class ModelPreviewWidget(Gtk.GLArea):
    """A minimal GLArea widget that displays a single .glb model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_has_depth_buffer(True)
        self.set_size_request(512, 288)
        self._camera: Optional[Camera] = None
        self._shader: Optional[Shader] = None
        self._renderer: Optional[_SimpleModelRenderer] = None
        self._mesh_data = None
        self._drag_start: Optional[tuple] = None
        self._last_offset: Optional[tuple] = None
        self.connect("realize", self._on_realize)
        self.connect("render", self._on_render)
        self.connect("resize", self._on_resize)
        motion = Gtk.GestureDrag()
        motion.connect("drag-begin", self._on_drag_begin)
        motion.connect("drag-update", self._on_drag_update)
        self.add_controller(motion)
        middle = Gtk.GestureDrag(button=2)
        middle.connect("drag-begin", self._on_drag_begin)
        middle.connect("drag-update", self._on_drag_update)
        self.add_controller(middle)
        scroll = Gtk.EventControllerScroll(
            flags=Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll.connect("scroll", self._on_scroll)
        self.add_controller(scroll)

    def load_model(self, resolved_path: Path):
        self._mesh_data = _load_mesh_data(resolved_path)
        if self.get_realized():
            self._init_renderer()

    def _on_realize(self, area):
        self.make_current()
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glClearColor(0.12, 0.12, 0.14, 1.0)
        try:
            self._shader = Shader(SIMPLE_VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER)
        except Exception as e:
            logger.error(f"Shader compilation failed: {e}")
            return
        if self._mesh_data:
            self._init_renderer()

    def _init_renderer(self):
        if self._shader is None or self._mesh_data is None:
            return
        self._renderer = _SimpleModelRenderer(self._mesh_data)
        self._renderer.init_gl()
        self._fit_camera()

    def _fit_camera(self):
        if self._mesh_data is None:
            return
        direction = np.array([-0.6, -0.7, 0.4])
        direction = direction / np.linalg.norm(direction)
        self._camera = Camera(
            position=direction * 2.5,
            target=np.zeros(3),
            up=np.array([0.0, 0.0, 1.0]),
            width=max(self.get_width(), 1),
            height=max(self.get_height(), 1),
        )

    def _on_render(self, area, ctx):
        if not self._camera or not self._renderer or not self._shader:
            return False
        color_bit = int(GL.GL_COLOR_BUFFER_BIT)
        depth_bit = int(GL.GL_DEPTH_BUFFER_BIT)
        GL.glClear(color_bit | depth_bit)
        proj = self._camera.get_projection_matrix()
        view = self._camera.get_view_matrix()
        mvp = (proj @ view).T
        self._renderer.render(
            self._shader, mvp, camera_position=self._camera.position
        )
        return True

    def _on_resize(self, area, w, h):
        if self._camera:
            self._camera.width = int(w)
            self._camera.height = int(h)

    def _on_drag_begin(self, gesture, start_x, start_y):
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        self._drag_start = (start_x, start_y)
        self._last_offset = (0.0, 0.0)

    def _on_drag_update(self, gesture, offset_x, offset_y):
        if self._drag_start is None or self._camera is None:
            return
        if self._last_offset is None:
            self._last_offset = (offset_x, offset_y)
            return
        dx = (offset_x - self._last_offset[0]) * 0.01
        dy = (offset_y - self._last_offset[1]) * 0.01
        self._last_offset = (offset_x, offset_y)
        target = self._camera.target
        forward = target - self._camera.position
        forward = forward / np.linalg.norm(forward)
        cam_up = self._camera.up.copy()
        cam_right = np.cross(forward, cam_up)
        norm = np.linalg.norm(cam_right)
        if norm > 1e-6:
            cam_right /= norm
            self._camera.orbit(target, cam_up, -dx)
            self._camera.orbit(target, cam_right, -dy)
        self.queue_render()

    def _on_scroll(self, controller, dx, dy):
        if self._camera:
            self._camera.dolly(dy)
            self.queue_render()


class _SimpleModelRenderer(BaseRenderer):
    """Renders pre-loaded mesh data as GL_TRIANGLES."""

    def __init__(self, mesh_data):
        super().__init__()
        self._mesh_data = mesh_data
        self._vao: int = 0
        self._vbo_pos: int = 0
        self._vbo_norm: int = 0
        self._vertex_count: int = 0
        self._positions = None
        self._normals = None

    def init_gl(self):
        flat_indices = self._mesh_data.faces.flatten()
        self._positions = self._mesh_data.positions[flat_indices].copy()
        self._normals = self._mesh_data.normals[flat_indices]
        y_up_to_z_up = np.array(
            [[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float32
        )
        self._positions = (y_up_to_z_up @ self._positions.T).T
        self._normals = (y_up_to_z_up @ self._normals.T).T
        bmin = self._positions.min(axis=0)
        bmax = self._positions.max(axis=0)
        center = (bmin + bmax) / 2.0
        extent = float((bmax - bmin).max())
        if extent > 1e-6:
            self._positions = (self._positions - center) / extent
        self._vertex_count = len(flat_indices)

        self._vao = self._create_vao()
        self._vbo_pos = self._create_vbo()
        self._vbo_norm = self._create_vbo()

        GL.glBindVertexArray(self._vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_pos)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self._positions.nbytes,
            self._positions,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo_norm)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            self._normals.nbytes,
            self._normals,
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(2, 3, GL.GL_FLOAT, GL.GL_TRUE, 0, None)
        GL.glEnableVertexAttribArray(2)

        GL.glBindVertexArray(0)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    def render(
        self,
        shader: Shader,
        mvp_matrix: np.ndarray,
        camera_position: Optional[np.ndarray] = None,
    ):
        if not self._vao:
            return
        shader.use()
        shader.set_mat4("uMVP", mvp_matrix)
        shader.set_float("uUseVertexColor", 0.0)
        shader.set_vec4("uColor", (0.55, 0.65, 0.75, 1.0))
        shader.set_float("uHasNormals", 1.0)
        shader.set_vec3("uLightDir", (0.5, 0.8, 1.0))
        cam_pos = (
            camera_position.astype(np.float32)
            if camera_position is not None
            else np.zeros(3, dtype=np.float32)
        )
        shader.set_vec3("uCameraPos", cam_pos)

        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._vertex_count)
        GL.glBindVertexArray(0)
