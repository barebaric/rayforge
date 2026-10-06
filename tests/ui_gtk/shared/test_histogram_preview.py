# flake8: noqa: E402

import cairo
import gi
import numpy as np

gi.require_version("Gtk", "4.0")

import pytest
from gi.repository import Pango, PangoCairo

from rayforge.ui_gtk.shared.histogram_preview import HistogramPreview

pytestmark = pytest.mark.ui


@pytest.mark.parametrize("width, height", [(200, 100), (300, 150)])
def test_placeholder_uses_centered_pango_layout(mocker, width, height):
    preview = HistogramPreview()
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    positions = []
    original_show_layout = PangoCairo.show_layout

    def draw_layout(context, layout):
        positions.append(context.get_current_point())
        original_show_layout(context, layout)

    show_layout = mocker.patch.object(
        PangoCairo, "show_layout", side_effect=draw_layout
    )
    preview._draw_func(preview, ctx, width, height)

    show_layout.assert_called_once()
    layout = show_layout.call_args.args[1]
    assert layout.get_text() == "No image"
    font = layout.get_font_description()
    assert font is not None
    assert font.get_family() == "Sans"
    assert font.get_size_is_absolute()
    assert font.get_size() == 12 * Pango.SCALE
    extents, _ = layout.get_pixel_extents()
    x, y = positions[0]
    assert x + extents.x + extents.width / 2 == pytest.approx(width / 2)
    assert y + extents.y + extents.height / 2 == pytest.approx(height / 2)
    assert any(surface.get_data())


def test_histogram_does_not_draw_placeholder(mocker):
    preview = HistogramPreview()
    preview.histogram = np.ones(256)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 100)
    ctx = cairo.Context(surface)
    show_layout = mocker.spy(PangoCairo, "show_layout")

    preview._draw_func(preview, ctx, 200, 100)

    show_layout.assert_not_called()
    assert any(surface.get_data())
