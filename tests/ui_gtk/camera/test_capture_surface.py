# flake8: noqa: E402

from unittest.mock import Mock

import cairo
import gi

gi.require_version("Gtk", "4.0")

import pytest
from gi.repository import Gtk, Pango, PangoCairo

from rayforge.camera.controller import CameraController
from rayforge.camera.models.camera import Camera, CameraSourceType
from rayforge.ui_gtk.camera.capture_surface import CalibrationCaptureSurface

pytestmark = pytest.mark.ui


@pytest.mark.parametrize(
    "message",
    ["Waiting for camera...", "\u7b49\u5f85\u76f8\u673a..."],
)
def test_waiting_message_uses_centered_pango_layout(mocker, message):
    camera = Camera(
        name="Test Camera",
        source_type=CameraSourceType.LOCAL_DEVICE,
        source_config={"device_id": "/dev/video0"},
    )
    mocker.patch.object(CalibrationCaptureSurface, "start")
    widget = CalibrationCaptureSurface(CameraController(camera))
    mocker.patch.object(widget, "get_width", return_value=750)
    mocker.patch.object(widget, "get_height", return_value=500)
    translate = mocker.patch(
        "rayforge.ui_gtk.camera.capture_surface._", return_value=message
    )
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 750, 500)
    ctx = cairo.Context(surface)
    snapshot = Mock(spec=Gtk.Snapshot)
    snapshot.append_cairo.return_value = ctx
    show_layout = mocker.spy(PangoCairo, "show_layout")

    widget.do_snapshot(snapshot)

    translate.assert_called_once_with("Waiting for camera...")
    show_layout.assert_called_once()
    layout = show_layout.call_args.args[1]
    assert layout.get_text() == message
    font = layout.get_font_description()
    assert font is not None
    assert font.get_family() == "Sans"
    assert font.get_size_is_absolute()
    assert font.get_size() == 14 * Pango.SCALE
    extents, _ = layout.get_pixel_extents()
    x, y = ctx.get_current_point()
    assert x + extents.x + extents.width / 2 == pytest.approx(375)
    assert y + extents.y + extents.height / 2 == pytest.approx(250)
