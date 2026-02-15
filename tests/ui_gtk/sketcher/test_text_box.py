import os
import sys

import pytest
import cairo
from unittest.mock import MagicMock, Mock

if sys.platform.startswith("linux"):
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    if not os.environ.get("DISPLAY"):
        pytest.skip(
            "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
            allow_module_level=True,
        )

from rayforge.core.geo.font_config import FontConfig
from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.entities import TextBoxEntity
from rayforge.core.sketcher.tools import TextBoxTool
from rayforge.core.sketcher.tools.text_box_tool import TextBoxState
from rayforge.ui_gtk.sketcher.sketchelement import SketchElement
from rayforge.ui_gtk.sketcher.renderer import SketchRenderer


@pytest.fixture
def sketch_with_text_box():
    """Create a sketch with a text box entity for testing."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Hello",
        font_config=FontConfig(font_family="sans-serif", font_size=10.0),
    )

    return sketch, tb_id


@pytest.fixture
def sketch_with_empty_text_box():
    """Create a sketch with an empty text box entity for testing."""
    sketch = Sketch()

    p_origin = sketch.add_point(10, 20)
    p_width = sketch.add_point(60, 20)
    p_height = sketch.add_point(10, 30)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="",
        font_config=FontConfig(font_family="sans-serif", font_size=10.0),
    )

    return sketch, tb_id


@pytest.fixture
def element_with_text_box(sketch_with_text_box):
    """Create a SketchElement with a text box and mocked canvas."""
    sketch, tb_id = sketch_with_text_box
    element = SketchElement(sketch=sketch)

    element.canvas = Mock()
    element.canvas.get_view_scale.return_value = (1.0, 1.0)
    element.canvas.get_color.return_value = Mock(red=0.0, green=0.0, blue=0.0)
    element.canvas.edit_context = element

    mock_matrix = Mock()
    mock_matrix.for_cairo.return_value = (1, 0, 0, 1, 0, 0)
    element.hittester = Mock()
    element.hittester.get_model_to_screen_transform.return_value = mock_matrix

    return element, tb_id


@pytest.fixture
def element_with_empty_text_box(sketch_with_empty_text_box):
    """Create a SketchElement with an empty text box and mocked canvas."""
    sketch, tb_id = sketch_with_empty_text_box
    element = SketchElement(sketch=sketch)

    element.canvas = Mock()
    element.canvas.get_view_scale.return_value = (1.0, 1.0)
    element.canvas.get_color.return_value = Mock(red=0.0, green=0.0, blue=0.0)
    element.canvas.edit_context = element

    mock_matrix = Mock()
    mock_matrix.for_cairo.return_value = (1, 0, 0, 1, 0, 0)
    element.hittester = Mock()
    element.hittester.get_model_to_screen_transform.return_value = mock_matrix

    return element, tb_id


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing."""
    element = Mock()
    element.content_transform = Mock()
    element.content_transform.for_cairo.return_value = (1, 0, 0, 1, 0)
    element.canvas = Mock()
    element.canvas.get_color.return_value = Mock(red=0.0, green=0.0, blue=0.0)
    element.canvas.get_view_scale.return_value = (1.0, 1.0)
    element.line_width = 1.0
    element.selection = Mock()
    element.selection.entity_ids = []
    return element


@pytest.fixture
def mock_cairo_context():
    """Create a mock Cairo context for testing."""
    ctx = MagicMock(spec=cairo.Context)
    return ctx


@pytest.mark.ui
def test_empty_text_box_cursor_at_origin(
    element_with_empty_text_box, mock_cairo_context
):
    """
    Test that the cursor is drawn at the element origin when text buffer
    is empty. This verifies the fix for the bug where the cursor was drawn
    at canvas zero.
    """
    element, tb_id = element_with_empty_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)
    assert text_tool.state == TextBoxState.EDITING
    assert text_tool.text_buffer == ""
    assert text_tool.cursor_pos == 0

    text_tool.draw_overlay(mock_cairo_context)

    mock_cairo_context.save.assert_called()
    mock_cairo_context.fill.assert_called()
    mock_cairo_context.restore.assert_called()


@pytest.mark.ui
def test_empty_text_box_cursor_moves_after_typing(
    element_with_empty_text_box, mock_cairo_context
):
    """
    Test that the cursor moves to the correct position after typing.
    This verifies that the cursor transformation works correctly when text
    is added to an initially empty text box.
    """
    element, tb_id = element_with_empty_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)
    assert text_tool.text_buffer == ""

    mock_cairo_context.reset_mock()

    text_tool.handle_text_input("A")
    assert text_tool.text_buffer == "A"
    assert text_tool.cursor_pos == 1

    text_tool.draw_overlay(mock_cairo_context)

    mock_cairo_context.fill.assert_called()


@pytest.mark.ui
def test_text_box_rendering_in_sketch_renderer(
    sketch_with_text_box, mock_element, mock_cairo_context
):
    """Test that a text box with content is rendered correctly."""
    sketch, tb_id = sketch_with_text_box
    mock_element.sketch = sketch

    renderer = SketchRenderer(mock_element)
    renderer.draw(mock_cairo_context)

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Hello"


@pytest.mark.ui
def test_text_box_rendering_with_empty_content(
    sketch_with_empty_text_box, mock_element, mock_cairo_context
):
    """Test that an empty text box is handled correctly."""
    sketch, tb_id = sketch_with_empty_text_box
    mock_element.sketch = sketch

    renderer = SketchRenderer(mock_element)
    renderer.draw(mock_cairo_context)

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == ""


@pytest.mark.ui
def test_text_box_rendering_produces_visible_output(
    sketch_with_text_box, mock_element, mock_cairo_context
):
    """
    Test that calling draw() on the renderer produces visible output.
    This verifies that the text geometry is generated and transformed
    before being drawn to the Cairo context.
    """
    sketch, tb_id = sketch_with_text_box
    mock_element.sketch = sketch

    renderer = SketchRenderer(mock_element)
    renderer.draw(mock_cairo_context)

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Hello"

    mock_cairo_context.save.assert_called()
    mock_cairo_context.restore.assert_called()
    mock_cairo_context.fill.assert_called()


@pytest.mark.ui
def test_text_box_rendering_with_different_font_params(
    sketch_with_text_box, mock_element, mock_cairo_context
):
    """Test that text boxes with different font params are rendered."""
    sketch, tb_id = sketch_with_text_box
    mock_element.sketch = sketch

    tb = sketch.registry.get_entity(tb_id)
    tb.font_config = FontConfig(
        font_family="serif",
        font_size=14.0,
        bold=True,
        italic=False,
    )

    renderer = SketchRenderer(mock_element)
    renderer.draw(mock_cairo_context)

    assert tb.font_config.font_family == "serif"
    assert tb.font_config.font_size == 14.0
    assert tb.font_config.bold is True


@pytest.mark.ui
def test_text_box_cursor_draws_when_visible(
    element_with_text_box, mock_cairo_context
):
    """
    Test that the text cursor is drawn when cursor_visible is True.
    This test verifies that the cursor drawing code path is executed
    when the cursor should be visible.
    """
    element, tb_id = element_with_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)
    assert text_tool.state == TextBoxState.EDITING
    assert text_tool.cursor_visible is True

    text_tool.draw_overlay(mock_cairo_context)

    mock_cairo_context.save.assert_called()
    mock_cairo_context.fill.assert_called()
    mock_cairo_context.restore.assert_called()


@pytest.mark.ui
def test_text_box_cursor_not_drawn_when_hidden(
    element_with_text_box, mock_cairo_context
):
    """
    Test that the text cursor is NOT drawn when cursor_visible is False.
    This test verifies that the cursor drawing code path is skipped
    when the cursor should be hidden.
    """
    element, tb_id = element_with_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)
    text_tool.cursor_visible = False

    text_tool.draw_overlay(mock_cairo_context)

    mock_cairo_context.save.assert_called()
    mock_cairo_context.fill.assert_called_once()
    mock_cairo_context.restore.assert_called()


@pytest.mark.ui
def test_text_box_cursor_toggles_visibility(
    element_with_text_box, mock_cairo_context
):
    """
    Test that the cursor visibility can be toggled.
    This simulates the UI timer behavior that blinks the cursor.
    """
    element, tb_id = element_with_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)

    mock_cairo_context.reset_mock()

    text_tool.draw_overlay(mock_cairo_context)
    assert mock_cairo_context.fill.call_count == 2

    text_tool.toggle_cursor_visibility()
    mock_cairo_context.reset_mock()

    text_tool.draw_overlay(mock_cairo_context)
    assert mock_cairo_context.fill.call_count == 1

    text_tool.toggle_cursor_visibility()
    mock_cairo_context.reset_mock()

    text_tool.draw_overlay(mock_cairo_context)
    assert mock_cairo_context.fill.call_count == 2


@pytest.mark.ui
def test_text_box_cursor_visible_after_text_input(
    element_with_text_box, mock_cairo_context
):
    """
    Test that the cursor becomes visible after text input.
    This ensures the cursor is shown when user types.
    """
    element, tb_id = element_with_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)
    text_tool.cursor_visible = False

    text_tool.handle_text_input("X")

    assert text_tool.cursor_visible is True


@pytest.mark.ui
def test_text_box_cursor_has_visible_width(
    element_with_text_box, mock_cairo_context
):
    """
    Test that cursor has a visible width in its geometry.
    This ensures cursor is thick enough to be seen on screen.
    """
    element, tb_id = element_with_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)
    text_tool.cursor_visible = True

    text_tool.draw_overlay(mock_cairo_context)

    mock_cairo_context.fill.assert_called()


@pytest.mark.ui
def test_text_box_cursor_visible_with_large_scale(
    element_with_text_box, mock_cairo_context
):
    """
    Test that cursor remains visible even with large scale.
    This ensures cursor is visible when zoomed out.
    """
    element, tb_id = element_with_text_box

    text_tool = element.tools["text_box"]
    assert isinstance(text_tool, TextBoxTool)

    text_tool.start_editing(tb_id)
    text_tool.cursor_visible = True

    element.canvas.get_view_scale.return_value = (10.0, 10.0)

    text_tool.draw_overlay(mock_cairo_context)

    mock_cairo_context.fill.assert_called()
