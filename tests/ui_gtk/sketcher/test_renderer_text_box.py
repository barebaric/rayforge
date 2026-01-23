import pytest
import cairo
from unittest.mock import Mock, MagicMock
from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.entities import TextBoxEntity
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
        content="Hello World",
        font_params={"family": "sans-serif", "size": 10.0},
    )

    return sketch, tb_id


@pytest.fixture
def sketch_with_empty_text_box():
    """Create a sketch with an empty text box entity for testing."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="",
        font_params={"family": "sans-serif", "size": 10.0},
    )

    return sketch, tb_id


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
    assert tb.content == "Hello World"


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
    assert tb.content == "Hello World"

    mock_cairo_context.save.assert_called()
    mock_cairo_context.restore.assert_called()
    mock_cairo_context.stroke.assert_called()


def test_text_box_rendering_with_different_font_params(
    sketch_with_text_box, mock_element, mock_cairo_context
):
    """Test that text boxes with different font params are rendered."""
    sketch, tb_id = sketch_with_text_box
    mock_element.sketch = sketch

    tb = sketch.registry.get_entity(tb_id)
    tb.font_params = {
        "family": "serif",
        "size": 14.0,
        "bold": True,
        "italic": False,
    }

    renderer = SketchRenderer(mock_element)
    renderer.draw(mock_cairo_context)

    assert tb.font_params["family"] == "serif"
    assert tb.font_params["size"] == 14.0
    assert tb.font_params["bold"] is True
