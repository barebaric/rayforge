from types import SimpleNamespace
from unittest.mock import Mock

import cairo
import pytest
from gi.repository import Pango, PangoCairo
from raygeo.geo import Matrix
from sketcher.core.commands.dimension import DimensionData
from sketcher.ui_gtk.renderer import SketchRenderer
from sketcher.ui_gtk.sketchelement import SketchElement

pytestmark = pytest.mark.ui


@pytest.mark.parametrize("editing", [False, True])
@pytest.mark.parametrize("leader", [None, (0, 0)])
def test_preview_dimensions_use_pango(mocker, editing, leader):
    dimension = DimensionData("12.00", (100, 100), leader)
    preview = Mock()
    preview.get_dimensions.return_value = [dimension]
    buffer = Mock()
    buffer.is_active.return_value = editing
    buffer.get_display_text.return_value = "24.00"
    tool = Mock()
    tool.get_preview_state.return_value = preview
    tool._dim_input = buffer
    hittester = Mock()
    hittester.get_model_to_screen_transform.return_value = Matrix.identity()
    element = Mock(
        spec=SketchElement,
        current_tool=tool,
        sketch=SimpleNamespace(registry=Mock()),
        hittester=hittester,
    )
    renderer = SketchRenderer(element)
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 200, 200)
    ctx = cairo.Context(surface)
    show_layout = mocker.spy(PangoCairo, "show_layout")

    renderer._draw_preview_dimensions(ctx)

    show_layout.assert_called_once()
    layout = show_layout.call_args.args[1]
    assert layout.get_text() == ("24.00" if editing else "12.00")
    font = layout.get_font_description()
    assert font is not None
    assert font.get_size_is_absolute()
    assert font.get_size() == 11 * Pango.SCALE
    assert any(surface.get_data())
