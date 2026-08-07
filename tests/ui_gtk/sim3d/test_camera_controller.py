"""UI tests for the CameraController interaction logic."""

# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import numpy as np
import pytest
from gi.repository import Gtk

from rayforge.ui_gtk.sim3d.camera import ViewDirection
from rayforge.ui_gtk.sim3d.camera_controller import CameraController
from rayforge.ui_gtk.sim3d.viewport import ViewportConfig


def _viewport(width_mm=100.0, depth_mm=100.0) -> ViewportConfig:
    return ViewportConfig.default(width_mm, depth_mm)


def _make_ctrl(get_viewport=None, request_render=lambda: None):
    return CameraController(
        Gtk.Box(),
        get_viewport=get_viewport or _viewport,
        request_render=request_render,
    )


@pytest.mark.ui
def test_create_camera_sets_dimensions(ui_context_initializer):
    ctrl = _make_ctrl()
    cam = ctrl.create_camera(640, 480)
    assert cam.width == 640
    assert cam.height == 480


@pytest.mark.ui
def test_on_resize_updates_camera_dimensions(ui_context_initializer):
    rendered = []
    ctrl = _make_ctrl(request_render=lambda: rendered.append(True))
    ctrl.create_camera(640, 480)
    ctrl.on_resize(None, 800, 600)
    cam = ctrl.camera
    assert cam is not None
    assert cam.width == 800
    assert cam.height == 600
    assert rendered == [True]


@pytest.mark.ui
def test_reset_view_uses_viewport_dimensions(ui_context_initializer):
    ctrl = _make_ctrl(
        get_viewport=lambda: _viewport(200.0, 150.0),
    )
    ctrl.create_camera(640, 480)
    ctrl.reset_view(ViewDirection.TOP)
    cam = ctrl.camera
    assert cam is not None
    target = cam.target
    assert target[0] == pytest.approx(100.0)
    assert target[1] == pytest.approx(75.0)


@pytest.mark.ui
def test_reset_view_without_camera_noop(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.reset_view(ViewDirection.TOP)


@pytest.mark.ui
def test_get_world_coords_on_plane_returns_point(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    pt = ctrl.get_world_coords_on_plane(320, 240)
    assert pt is not None
    assert pt.shape == (3,)
    assert pt[2] == pytest.approx(0.0, abs=1e-6)


@pytest.mark.ui
def test_on_scroll_dollies_camera(ui_context_initializer):
    rendered = []
    ctrl = _make_ctrl(request_render=lambda: rendered.append(True))
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.is_perspective = True
    before = cam.position.copy()
    ctrl.on_scroll(None, 0.0, -1.0)
    assert not np.array_equal(cam.position, before)
    assert rendered == [True]


@pytest.mark.ui
def test_drag_update_pans_with_shift(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    before = cam.position.copy()
    ctrl.on_drag_begin(_FakeGesture(shift=True), 0.0, 0.0)
    ctrl.on_drag_update(_FakeGesture(shift=True), 10.0, 5.0)
    assert not np.array_equal(cam.position, before)


class _FakeGesture:
    """Minimal GestureDrag stand-in for the drag handlers."""

    def __init__(self, shift=False, event=None):
        self._shift = shift
        self._event = event

    def set_state(self, state):
        pass

    def get_current_event_state(self):
        from gi.repository import Gdk

        if self._shift:
            return Gdk.ModifierType.SHIFT_MASK
        return 0

    def get_last_event(self):
        return self._event


@pytest.mark.ui
def test_drag_begin_sets_orbit_state(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    ctrl.on_drag_begin(_FakeGesture(shift=False), 100.0, 100.0)
    assert ctrl._is_orbiting is True
    assert ctrl._rotation_pivot is not None
