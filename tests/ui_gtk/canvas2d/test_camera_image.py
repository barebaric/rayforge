"""Tests for the camera image element's optional outside view."""

from unittest.mock import MagicMock, patch

import cairo
import numpy as np
import pytest

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera
from rayforge.ui_gtk.canvas2d.elements.camera_image import (
    CameraImageElement,
)

pytestmark = pytest.mark.ui

INSIDE = (150, 150)
OUTSIDE = (90, 150)
BEYOND_COVERAGE = (50, 150)


def _controller() -> CameraController:
    """An aligned 640x480 white frame covering world x in [-25, 135]."""
    camera = Camera("Test Camera", "0")
    camera.enabled = True
    camera.image_to_world = (
        [(100, 100), (500, 100), (500, 400), (100, 400)],
        [(0, 100), (100, 100), (100, 0), (0, 0)],
    )
    camera.outside_view_margin_mm = 20
    controller = CameraController(camera)
    controller._image_data = np.full((480, 640, 3), 255, dtype=np.uint8)
    return controller


def _element(controller, supported=True) -> CameraImageElement:
    canvas = MagicMock()
    canvas.width_mm = 100.0
    canvas.height_mm = 100.0
    canvas._cam_visible = True
    canvas.get_width.return_value = 200
    canvas.get_height.return_value = 200
    element = CameraImageElement(controller, outside_view_supported=supported)
    element.canvas = canvas
    element.allocate()
    return element


def _process(element: CameraImageElement) -> None:
    key = element._current_key()
    assert key is not None
    element._process_and_update_cache(key)
    assert element._cached_key == key


def _render(element: CameraImageElement) -> np.ndarray:
    """Render at 1 px/mm with world (0, 0) at pixel (100, 200)."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 300, 300)
    ctx = cairo.Context(surface)
    ctx.translate(100, 200)
    ctx.scale(1, -1)
    with patch("rayforge.ui_gtk.canvas2d.elements.camera_image.GLib"):
        element.render(ctx)
    surface.flush()
    buf = np.frombuffer(surface.get_data(), dtype=np.uint8)
    return buf.reshape(300, 300, 4)


def _alpha(pixels: np.ndarray, point: tuple[int, int]) -> int:
    x, y = point
    return int(pixels[y, x, 3])


def test_outside_view_disabled_matches_existing_render():
    controller = _controller()
    legacy = _element(controller, supported=False)
    element = _element(controller)
    _process(legacy)
    with patch.object(
        controller,
        "get_work_surface_images",
        wraps=controller.get_work_surface_images,
    ) as paired:
        _process(element)
        paired.assert_not_called()

    np.testing.assert_array_equal(_render(element), _render(legacy))
    assert _alpha(_render(element), OUTSIDE) == 0


def test_outside_view_draws_only_outside_with_own_transparency():
    controller = _controller()
    element = _element(controller)
    _process(element)
    inside_alpha = _alpha(_render(element), INSIDE)

    controller.config.outside_view_enabled = True
    _process(element)
    pixels = _render(element)

    assert _alpha(pixels, INSIDE) == inside_alpha
    assert _alpha(pixels, OUTSIDE) == pytest.approx(0.65 * 255, abs=1)
    assert _alpha(pixels, BEYOND_COVERAGE) == 0
    # The margin ends 20 mm outside the workspace.
    assert _alpha(pixels, (75, 150)) == 0

    controller.config.outside_view_transparency = 0.3
    pixels = _render(element)
    assert _alpha(pixels, OUTSIDE) == pytest.approx(0.3 * 255, abs=1)
    assert _alpha(pixels, INSIDE) == inside_alpha


def test_outside_view_is_hidden_immediately_when_invalidated():
    controller = _controller()
    controller.config.outside_view_enabled = True
    element = _element(controller)
    _process(element)
    assert _alpha(_render(element), OUTSIDE) > 0

    controller.config.outside_view_margin_mm = 10
    assert _alpha(_render(element), OUTSIDE) == 0

    controller.config.outside_view_margin_mm = 20
    controller.config.outside_view_enabled = False
    assert _alpha(_render(element), OUTSIDE) == 0

    controller.config.outside_view_enabled = True
    controller.config.image_to_world = None
    assert _alpha(_render(element), OUTSIDE) == 0


def test_stale_job_does_not_commit_after_settings_change():
    controller = _controller()
    element = _element(controller)
    key = element._current_key()
    assert key is not None

    controller.config.outside_view_enabled = True
    element._process_and_update_cache(key)

    assert element._cached_surface is None
    assert element._outside_cache is None


def test_draw_does_not_schedule_duplicate_jobs():
    controller = _controller()
    element = _element(controller)
    ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10))

    with patch("rayforge.ui_gtk.canvas2d.elements.camera_image.GLib") as glib:
        element.draw(ctx)
        element.draw(ctx)

    assert glib.idle_add.call_count == 1


def test_element_without_outside_support_keeps_its_clip():
    controller = _controller()
    controller.config.outside_view_enabled = True

    element = _element(controller, supported=False)

    assert element.clip
    assert element._outside_margin_request() is None
