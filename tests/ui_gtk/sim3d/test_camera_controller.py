# flake8: noqa: E402
"""UI tests for the CameraController interaction logic."""

import math

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import numpy as np
import pytest
from gi.repository import Gtk

from rayforge.ui_gtk.sim3d.camera import Camera, ViewDirection
from rayforge.ui_gtk.sim3d.camera_controller import CameraController
from rayforge.ui_gtk.sim3d.viewport import ViewportConfig


def _viewport(width_mm=100.0, depth_mm=100.0) -> ViewportConfig:
    return ViewportConfig.default(width_mm, depth_mm)


def _make_ctrl(
    get_viewport=None,
    request_render=lambda: None,
    get_pick_scene=None,
):
    return CameraController(
        Gtk.Box(),
        get_viewport=get_viewport or _viewport,
        request_render=request_render,
        get_pick_scene=get_pick_scene,
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
def test_get_world_coords_on_plane_ortho_uses_parallel_ray(
    ui_context_initializer,
):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None

    cam.set_view(ViewDirection.TOP, 100.0, 100.0)
    assert cam._ortho_ref_distance is not None
    ndc_x = 2.0 * 400 / 640 - 1.0
    ndc_y = 1.0 - 2.0 * 200 / 480
    half_height = cam._ortho_ref_distance * math.tan(math.radians(45.0) / 2.0)
    right = half_height * (640 / 480)

    pt = ctrl.get_world_coords_on_plane(400, 200)
    assert pt is not None
    expected = np.array(
        [50.0 + ndc_x * right, 50.0 + ndc_y * half_height, 0.0]
    )
    assert np.allclose(pt, expected, atol=1e-6)


@pytest.mark.ui
def test_on_scroll_dollies_camera(ui_context_initializer):
    rendered = []
    ctrl = _make_ctrl(request_render=lambda: rendered.append(True))
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.is_perspective = True
    before = cam.position.copy()
    scroll = Gtk.EventControllerScroll.new(
        Gtk.EventControllerScrollFlags.VERTICAL
    )
    ctrl.on_scroll(scroll, 0.0, -1.0)
    assert not np.array_equal(cam.position, before)
    assert rendered == [True]


@pytest.mark.ui
def test_on_scroll_zooms_towards_cursor_point(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.is_perspective = True

    cursor = (400.0, 200.0)
    before = ctrl.get_world_coords_on_plane(*cursor)
    assert before is not None

    ctrl._mouse_pos = cursor
    scroll = Gtk.EventControllerScroll.new(
        Gtk.EventControllerScrollFlags.VERTICAL
    )
    ctrl.on_scroll(scroll, 0.0, -1.0)

    after = ctrl.get_world_coords_on_plane(*cursor)
    assert after is not None
    assert np.allclose(before, after, atol=1e-6)


@pytest.mark.ui
def test_zoom_towards_point_dollies_camera(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.is_perspective = True
    before = cam.position.copy()
    ctrl.zoom_towards_point(320, 240, -1.0)
    assert not np.array_equal(cam.position, before)


@pytest.mark.ui
def test_zoom_towards_point_keeps_cursor_point_fixed(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.is_perspective = True

    cursor = (400.0, 200.0)
    before = ctrl.get_world_coords_on_plane(*cursor)
    assert before is not None

    ctrl.zoom_towards_point(*cursor, -1.0)

    after = ctrl.get_world_coords_on_plane(*cursor)
    assert after is not None
    assert np.allclose(before, after, atol=1e-6)


@pytest.mark.ui
def test_zoom_towards_point_keeps_cursor_point_fixed_ortho(
    ui_context_initializer,
):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None

    cursor = (400.0, 200.0)
    before = ctrl.get_world_coords_on_plane(*cursor)
    assert before is not None
    assert not np.allclose(before, cam.target, atol=1e-3)

    ctrl.zoom_towards_point(*cursor, -1.0)

    after = ctrl.get_world_coords_on_plane(*cursor)
    assert after is not None
    assert np.allclose(before, after, atol=1e-6)


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


@pytest.mark.ui
def test_pan_tracks_cursor_point_1_to_1(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)

    start = (400.0, 200.0)
    grabbed = ctrl.get_world_coords_on_plane(*start)
    assert grabbed is not None

    ctrl.on_drag_begin(_FakeGesture(shift=True), *start)

    end = (450.0, 260.0)
    ctrl.on_drag_update(
        _FakeGesture(shift=True), end[0] - start[0], end[1] - start[1]
    )

    after = ctrl.get_world_coords_on_plane(*end)
    assert after is not None
    assert np.allclose(after, grabbed, atol=1e-6)


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


class _FakeEvent:
    """Minimal Gdk.Event stand-in providing a cursor position."""

    def __init__(self, position):
        self._position = position

    def get_position(self):
        return True, self._position[0], self._position[1]


@pytest.mark.ui
def test_drag_begin_sets_orbit_state(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    ctrl.on_drag_begin(_FakeGesture(shift=False), 100.0, 100.0)
    assert ctrl._is_orbiting is True
    assert ctrl._rotation_pivot is not None


@pytest.mark.ui
def test_drag_begin_orbits_around_plane_point(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None

    cursor = (400.0, 200.0)
    expected = ctrl.get_world_coords_on_plane(*cursor)
    assert expected is not None
    assert not np.allclose(expected, cam.target, atol=1e-3)

    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    assert ctrl._rotation_pivot is not None
    assert np.allclose(ctrl._rotation_pivot, expected, atol=1e-6)


@pytest.mark.ui
def test_drag_begin_orbits_around_plane_point_perspective(
    ui_context_initializer,
):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.is_perspective = True

    cursor = (400.0, 200.0)
    expected = ctrl.get_world_coords_on_plane(*cursor)
    assert expected is not None

    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    assert ctrl._rotation_pivot is not None
    assert np.allclose(ctrl._rotation_pivot, expected, atol=1e-6)


def _object_pick_scene(z: float = 10.0):
    """A horizontal quad covering the bed, ``z`` mm above the plane."""
    from rayforge.simulator.scene3d.picking import PickMesh, PickScene

    corners = np.array(
        [
            [0.0, 0.0, z],
            [100.0, 0.0, z],
            [100.0, 100.0, z],
            [0.0, 100.0, z],
        ],
        dtype=np.float32,
    )
    scene = PickScene()
    scene.meshes.append(
        PickMesh(
            np.vstack(
                [
                    corners[0],
                    corners[1],
                    corners[2],
                    corners[0],
                    corners[2],
                    corners[3],
                ]
            )
        )
    )
    return scene


@pytest.mark.ui
def test_drag_begin_orbits_around_object_point(ui_context_initializer):
    ctrl = _make_ctrl(get_pick_scene=lambda: _object_pick_scene(10.0))
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)

    cursor = (320.0, 240.0)
    plane_point = ctrl.get_world_coords_on_plane(*cursor)
    assert plane_point is not None
    assert plane_point[2] == pytest.approx(0.0, abs=1e-6)

    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    pivot = ctrl._rotation_pivot
    assert pivot is not None
    assert pivot[0] == pytest.approx(plane_point[0], abs=1e-5)
    assert pivot[1] == pytest.approx(plane_point[1], abs=1e-5)
    assert pivot[2] == pytest.approx(10.0, abs=1e-5)


@pytest.mark.ui
def test_drag_begin_orbits_around_object_point_perspective(
    ui_context_initializer,
):
    ctrl = _make_ctrl(get_pick_scene=lambda: _object_pick_scene(10.0))
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)
    cam.is_perspective = True

    cursor = (320.0, 240.0)
    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    pivot = ctrl._rotation_pivot
    assert pivot is not None
    assert pivot[0] == pytest.approx(50.0, abs=1e-5)
    assert pivot[1] == pytest.approx(50.0, abs=1e-5)
    assert pivot[2] == pytest.approx(10.0, abs=1e-3)


@pytest.mark.ui
def test_drag_begin_falls_back_to_plane_without_pick_scene(
    ui_context_initializer,
):
    ctrl = _make_ctrl(get_pick_scene=lambda: None)
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)

    cursor = (400.0, 200.0)
    expected = ctrl.get_world_coords_on_plane(*cursor)
    assert expected is not None

    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    assert ctrl._rotation_pivot is not None
    assert np.allclose(ctrl._rotation_pivot, expected, atol=1e-6)


def _screen_pos(camera: Camera, world: np.ndarray) -> np.ndarray:
    """Projects a world point to NDC screen coordinates."""
    view = camera.get_view_matrix() @ np.append(world, 1.0)
    clip = camera.get_projection_matrix() @ view
    return np.array([clip[0], clip[1]]) / clip[3]


@pytest.mark.ui
def test_ortho_orbit_keeps_pivot_fixed_on_screen(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)

    cursor = (400.0, 200.0)
    pivot = ctrl.get_world_coords_on_plane(*cursor)
    assert pivot is not None

    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    ctrl.on_drag_update(_FakeGesture(event=_FakeEvent(cursor)), 0.0, 0.0)
    before = _screen_pos(cam, pivot)

    ctrl.on_drag_update(
        _FakeGesture(event=_FakeEvent((420.0, 200.0))), 0.0, 0.0
    )
    after_yaw = _screen_pos(cam, pivot)

    ctrl.on_drag_update(
        _FakeGesture(event=_FakeEvent((420.0, 180.0))), 0.0, 0.0
    )
    after_pitch = _screen_pos(cam, pivot)

    assert np.allclose(before, after_yaw, atol=1e-6)
    assert np.allclose(before, after_pitch, atol=1e-6)
    assert not np.allclose(cam.target, pivot, atol=1e-3)


@pytest.mark.ui
def test_perspective_yaw_orbits_around_world_z(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.set_view(ViewDirection.ISO, 100.0, 100.0)
    cam.is_perspective = True

    cursor = (400.0, 200.0)
    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    z_before = cam.position[2]

    # A pure horizontal drag must yaw around the world Z axis, keeping
    # the camera's height above the pivot constant.
    ctrl.on_drag_update(_FakeGesture(event=_FakeEvent(cursor)), 0.0, 0.0)
    ctrl.on_drag_update(
        _FakeGesture(event=_FakeEvent((420.0, 200.0))), 0.0, 0.0
    )

    assert cam.position[2] == pytest.approx(z_before, abs=1e-6)


@pytest.mark.ui
def test_perspective_orbit_keeps_pivot_fixed_on_screen(ui_context_initializer):
    ctrl = _make_ctrl()
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.is_perspective = True

    cursor = (400.0, 200.0)
    pivot = ctrl.get_world_coords_on_plane(*cursor)
    assert pivot is not None

    ctrl.on_drag_begin(_FakeGesture(shift=False), *cursor)
    ctrl.on_drag_update(_FakeGesture(event=_FakeEvent(cursor)), 0.0, 0.0)
    before = _screen_pos(cam, pivot)

    ctrl.on_drag_update(
        _FakeGesture(event=_FakeEvent((420.0, 200.0))), 0.0, 0.0
    )

    assert np.allclose(before, _screen_pos(cam, pivot), atol=1e-6)
