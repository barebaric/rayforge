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


def _grazing_camera(ctrl: CameraController) -> Camera:
    """Positions a perspective camera near the plane looking along it."""
    cam = ctrl.create_camera(640, 480)
    cam.is_perspective = True
    cam.position = np.array([50.0, 50.0, 1.0])
    cam.target = np.array([50.0, 250.0, 0.0])
    cam.up = np.array([0.0, 0.0, 1.0])
    return cam


@pytest.mark.ui
def test_plane_pick_is_none_in_grazing_view(ui_context_initializer):
    """A ray that grazes the plane must not yield a near-infinite pick."""
    ctrl = _make_ctrl()
    cam = _grazing_camera(ctrl)
    assert cam is not None

    pt = ctrl.get_world_coords_on_plane(320, 240)
    assert pt is None


@pytest.mark.ui
def test_pan_in_grazing_view_falls_back_to_pixel_pan(
    ui_context_initializer,
):
    """Panning must not produce extreme shifts in a grazing view."""
    ctrl = _make_ctrl()
    cam = _grazing_camera(ctrl)
    assert cam is not None
    before = cam.position.copy()

    ctrl.on_drag_begin(_FakeGesture(shift=True), 320.0, 240.0)
    ctrl.on_drag_update(_FakeGesture(shift=True), 10.0, 5.0)

    shift = np.linalg.norm(cam.position - before)
    # The far-plane anchoring is disabled, so a 10px drag must stay small
    # and bounded rather than jumping to a near-infinite pick point.
    assert shift < 100.0
    assert shift > 0.0


@pytest.mark.ui
def test_zoom_towards_point_in_grazing_view_is_bounded(
    ui_context_initializer,
):
    """Zooming in a grazing view must not teleport the camera."""
    ctrl = _make_ctrl()
    cam = _grazing_camera(ctrl)
    assert cam is not None
    before = cam.position.copy()

    ctrl.zoom_towards_point(400.0, 200.0, -1.0)

    shift = np.linalg.norm(cam.position - before)
    assert shift < 100.0
    # It should still dolly in (move toward the target).
    assert np.linalg.norm(cam.target - cam.position) < np.linalg.norm(
        cam.target - before
    )


def _perspective_camera(ctrl: CameraController) -> Camera:
    """Positions a perspective camera looking down the -Z axis."""
    cam = ctrl.create_camera(640, 480)
    cam.is_perspective = True
    cam.position = np.array([0.0, 0.0, 100.0])
    cam.target = np.array([0.0, 0.0, 0.0])
    cam.up = np.array([0.0, 1.0, 0.0])
    return cam


@pytest.mark.ui
def test_dolly_zoom_in_never_passes_target(ui_context_initializer):
    """A large zoom-in scroll must not overshoot and reverse the view."""
    ctrl = _make_ctrl()
    cam = _perspective_camera(ctrl)
    assert cam is not None

    before = np.linalg.norm(cam.target - cam.position)
    ctrl.zoom_towards_point(320.0, 240.0, -15.0)
    after = np.linalg.norm(cam.target - cam.position)

    # The camera must stay on its own side of the target and keep moving
    # in, never jumping past it (which would flip the view direction).
    assert 0.0 < after < before
    assert np.dot(cam.target - cam.position, np.array([0.0, 0.0, -1.0])) > 0


@pytest.mark.ui
def test_dolly_zoom_out_increases_distance(ui_context_initializer):
    ctrl = _make_ctrl()
    cam = _perspective_camera(ctrl)
    assert cam is not None

    before = np.linalg.norm(cam.target - cam.position)
    ctrl.zoom_towards_point(320.0, 240.0, 5.0)
    after = np.linalg.norm(cam.target - cam.position)

    assert after > before


@pytest.mark.ui
def test_zoom_direction_is_independent_of_cursor_position(
    ui_context_initializer,
):
    """Zooming must zoom in/out consistently wherever the cursor points.

    The cursor over the plane and over the sky (above the plane) used to
    zoom in opposite directions because the plane-anchoring correction
    fought the dolly.
    """
    ctrl = _make_ctrl()
    cam = ctrl.create_camera(640, 480)
    cam.is_perspective = True
    cam.position = np.array([50.0, -300.0, 40.0])
    cam.target = np.array([50.0, 50.0, 0.0])
    cam.up = np.array([0.0, 0.0, 1.0])
    assert cam is not None

    for y in (80.0, 220.0, 340.0):  # above the plane and on the plane
        base = np.linalg.norm(cam.target - cam.position)

        zoom_in = ctrl.create_camera(640, 480)
        zoom_in.is_perspective = True
        zoom_in.position = cam.position.copy()
        zoom_in.target = cam.target.copy()
        zoom_in.up = cam.up.copy()
        ctrl.zoom_towards_point(320.0, y, -1.0)
        assert np.linalg.norm(zoom_in.target - zoom_in.position) < base

        zoom_out = ctrl.create_camera(640, 480)
        zoom_out.is_perspective = True
        zoom_out.position = cam.position.copy()
        zoom_out.target = cam.target.copy()
        zoom_out.up = cam.up.copy()
        ctrl.zoom_towards_point(320.0, y, 1.0)
        assert np.linalg.norm(zoom_out.target - zoom_out.position) > base


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


def _ortho_orbit_screen_motion(ctrl: CameraController, zoom: float) -> float:
    """On-screen movement of a fixed model point after an ortho yaw.

    Returns the movement in pixels so the value is meaningful regardless of
    the projection scale.
    """
    cam = ctrl.create_camera(640, 480)
    cam.set_view(ViewDirection.ISO, 100.0, 100.0)
    cam._ortho_zoom = zoom
    ref = np.linalg.norm(cam.target - cam.position)
    pivot = cam.target + np.array([2.0, 0.0, 0.0]) * ref
    fixed = cam.target.copy()
    before = _screen_pos(cam, fixed)
    ctrl._apply_orbit(cam, pivot, 10.0, 0.0)
    after = _screen_pos(cam, fixed)
    ndc = np.linalg.norm(after - before)
    return ndc * (480.0 / 2.0)


@pytest.mark.ui
def test_ortho_orbit_speed_is_zoom_independent(ui_context_initializer):
    """Ortho zoom must not amplify the on-screen rotation speed.

    Zooming an orthographic view does not move the camera, so without
    normalisation the on-screen rotation rate would scale with the zoom,
    making orbit extremely rapid when zoomed in.
    """
    ctrl = _make_ctrl()
    base = _ortho_orbit_screen_motion(ctrl, 1.0)
    assert base > 0.0
    for zoom in (3.0, 5.0, 20.0):
        moved = _ortho_orbit_screen_motion(ctrl, zoom)
        assert moved == pytest.approx(base, rel=0.05)


@pytest.mark.ui
def test_pick_pivot_clamps_far_point(ui_context_initializer):
    """A far clicked point must be pulled in so orbit stays comfortable."""
    ctrl = _make_ctrl()
    cam = ctrl.create_camera(640, 480)
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)

    far_pivot = np.array([10000.0, 10000.0, 0.0])
    clamped = ctrl._clamp_pivot(far_pivot)
    assert clamped is not None
    distance = np.linalg.norm(clamped - cam.position)
    ref = np.linalg.norm(cam.target - cam.position)
    assert distance <= 3.0 * ref + 1e-6


@pytest.mark.ui
def test_plane_pick_pivot_is_clamped_to_grid(ui_context_initializer):
    """An off-grid plane pick must be clamped onto the grid bounds.

    Without the clamp, clicking empty plane far outside the grid would
    orbit around a distant point and rotate the view extremely rapidly.
    """
    ctrl = _make_ctrl(get_viewport=lambda: _viewport(100.0, 100.0))
    ctrl.create_camera(640, 480)
    cam = ctrl.camera
    assert cam is not None
    cam.set_view(ViewDirection.TOP, 100.0, 100.0)

    # A screen point whose plane intersection is far outside the grid.
    pivot = ctrl._pick_pivot(10.0, 10.0)
    assert pivot is not None
    assert 0.0 <= pivot[0] <= 100.0
    assert 0.0 <= pivot[1] <= 100.0


def _screen_motion_for_pivot(
    ctrl: CameraController, pivot: np.ndarray, mode: str
) -> float:
    """On-screen pixel movement of the model for a 10px yaw around a pivot."""
    cam = ctrl.create_camera(640, 480)
    cam.set_view(ViewDirection.ISO, 100.0, 100.0)
    if mode == "persp":
        cam.is_perspective = True
    fixed = cam.target.copy()
    before = _screen_pos(cam, fixed)
    ctrl._apply_orbit(cam, pivot, 10.0, 0.0)
    after = _screen_pos(cam, fixed)
    ndc = np.linalg.norm(after - before)
    return ndc * (480.0 / 2.0)


@pytest.mark.ui
def test_orbit_speed_is_bounded_for_far_pivot(ui_context_initializer):
    """A far orbit pivot must not make the view race across the screen."""
    ctrl = _make_ctrl()
    cam = ctrl.create_camera(640, 480)
    cam.set_view(ViewDirection.ISO, 100.0, 100.0)
    ref = np.linalg.norm(cam.target - cam.position)

    far = cam.target + np.array([10.0, 0.0, 0.0]) * ref

    for mode in ("persp", "ortho"):
        far_motion = _screen_motion_for_pivot(ctrl, far, mode)
        # A far pivot must stay within a quarter of the screen per 10px
        # drag, rather than flying off-screen (which used to happen before
        # the sensitivity was normalised by the pivot distance).
        assert far_motion < 120.0


@pytest.mark.ui
def test_orbit_fallback_pivot_is_clamped_to_grid(ui_context_initializer):
    """A missing pick must fall back to a pivot on the grid plane.

    When no point is picked (e.g. the cursor is over the sky in a grazing
    view), the orbit used to fall back to the camera target, which can drift
    far below the plane and make the camera sweep an enormous arc while
    orbiting.  The fallback must be projected onto the grid and clamped.
    """
    ctrl = _make_ctrl(get_viewport=lambda: _viewport(100.0, 100.0))
    cam = ctrl.create_camera(640, 480)
    cam.set_view(ViewDirection.FRONT, 100.0, 100.0)
    cam.position[2] += 20.0  # raise the camera above the plane

    # In this near-plane-aligned FRONT view the pick is grazing/behind, so
    # _pick_pivot returns None and the fallback is used.
    picked = ctrl._pick_pivot(320.0, 430.0)
    assert picked is None

    ctrl.on_drag_begin(_FakeGesture(shift=False), 320.0, 430.0)
    pivot = ctrl._rotation_pivot
    assert pivot is not None
    # The fallback must lie on the grid plane and inside the grid bounds.
    assert pivot[2] == pytest.approx(0.0, abs=1e-6)
    assert 0.0 <= pivot[0] <= 100.0
    assert 0.0 <= pivot[1] <= 100.0
