# flake8: noqa: E402

import cairo
import gi

gi.require_version("Gtk", "4.0")

import pytest
from gi.repository import Pango, PangoCairo

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera, CameraSourceType
from rayforge.ui_gtk.camera.display_widget import CameraDisplay

pytestmark = pytest.mark.ui


@pytest.fixture
def display(mocker):
    camera = Camera(
        name="Test Camera",
        source_type=CameraSourceType.LOCAL_DEVICE,
        source_config={"device_id": "/dev/video0"},
    )
    mocker.patch.object(CameraDisplay, "start")
    return CameraDisplay(CameraController(camera))


@pytest.mark.parametrize(
    "method, message",
    [
        ("_draw_disabled_message", "Camera Disabled"),
        ("_draw_no_image_message", "No Image"),
    ],
)
def test_status_message_uses_centered_pango_layout(
    display, mocker, method, message
):
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 640, 480)
    ctx = cairo.Context(surface)
    show_layout = mocker.spy(PangoCairo, "show_layout")
    getattr(display, method)(ctx, 640, 480)

    show_layout.assert_called_once()
    layout = show_layout.call_args.args[1]
    assert layout.get_text() == message
    font = layout.get_font_description()
    assert font is not None
    assert font.get_family() == "Sans"
    assert font.get_weight() == Pango.Weight.BOLD
    assert not font.get_size_is_absolute()
    assert font.get_size() == 24 * Pango.SCALE
    extents, _ = layout.get_pixel_extents()
    x, y = ctx.get_current_point()
    assert x + extents.x + extents.width / 2 == pytest.approx(320)
    assert y + extents.y + extents.height / 2 == pytest.approx(240)
    assert any(surface.get_data())
