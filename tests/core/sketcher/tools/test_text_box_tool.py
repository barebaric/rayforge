from unittest.mock import MagicMock, Mock
import pytest

from rayforge.core.sketcher.tools import TextBoxTool
from rayforge.core.sketcher.tools.base import SketcherKey
from rayforge.core.sketcher.tools.text_box_tool import TextBoxState
from rayforge.core.sketcher.entities import TextBoxEntity


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing tools."""
    element = Mock()
    element.sketch = Mock()
    element.sketch.registry._id_counter = 0
    element.sketch.registry.entities = []
    element.sketch.is_fully_constrained = False
    element.hittester.get_hit_data.return_value = (None, None)
    element.execute_command = MagicMock()
    element.mark_dirty = MagicMock()
    element.sketch.registry.get_entity = Mock(
        side_effect=lambda x: Mock() if x == 5 else None
    )
    element.content_transform = Mock()
    element.content_transform.transform_point = Mock(return_value=(100, 200))
    return element


@pytest.fixture
def text_box_tool(mock_element):
    """Create a TextBoxTool instance with a mocked element."""
    return TextBoxTool(mock_element)


def test_text_box_tool_initialization(text_box_tool):
    """Test the tool's initial state."""
    assert text_box_tool.state == TextBoxState.IDLE
    assert text_box_tool.editing_entity_id is None
    assert text_box_tool.text_buffer == ""
    assert text_box_tool.cursor_pos == 0
    assert text_box_tool.cursor_visible is True


def test_text_box_tool_on_press_creates_box(text_box_tool, mock_element):
    """Test that first press creates a text box and enters EDITING state."""
    mock_element.hittester.screen_to_model.return_value = (10, 20)
    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="", construction_line_ids=[]
    )
    mock_element.sketch.registry.entities = [mock_entity]
    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    result = text_box_tool.on_press(100, 200, 1)

    assert result is True
    assert text_box_tool.state == TextBoxState.EDITING
    assert text_box_tool.editing_entity_id is not None
    assert text_box_tool.text_buffer == ""
    assert mock_element.execute_command.call_count == 1
    mock_element.mark_dirty.assert_called()


def test_text_box_tool_on_press_outside_box_finalizes(
    text_box_tool, mock_element
):
    """Test that clicking outside the box finalizes the edit."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.editing_entity_id = 5
    text_box_tool.text_buffer = "Test Text"

    mock_element.hittester.screen_to_model.return_value = (1000, 1000)
    mock_element.content_transform.transform_point.return_value = (1000, 1000)
    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="Test Text", construction_line_ids=[]
    )
    mock_entity.font_params = {"family": "sans-serif", "size": 10.0}
    mock_element.sketch.registry.get_entity.return_value = mock_entity
    mock_element.sketch.registry.get_point.side_effect = [
        Mock(x=0, y=0),
        Mock(x=50, y=0),
        Mock(x=0, y=10),
    ]

    result = text_box_tool.on_press(100, 200, 1)

    assert result is False
    assert text_box_tool.state == TextBoxState.IDLE
    assert text_box_tool.editing_entity_id is None
    assert text_box_tool.text_buffer == ""
    mock_element.execute_command.assert_called_once()


def test_text_box_tool_handle_text_input_appends_character(
    text_box_tool, mock_element
):
    """Test that text input appends character to buffer."""
    text_box_tool.state = TextBoxState.EDITING

    result = text_box_tool.handle_text_input("A")

    assert result is True
    assert text_box_tool.text_buffer == "A"
    assert text_box_tool.cursor_pos == 1


def test_text_box_tool_handle_text_input_inserts_at_cursor(
    text_box_tool, mock_element
):
    """Test that text input inserts at cursor position."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.text_buffer = "AB"
    text_box_tool.cursor_pos = 1

    result = text_box_tool.handle_text_input("X")

    assert result is True
    assert text_box_tool.text_buffer == "AXB"
    assert text_box_tool.cursor_pos == 2


def test_text_box_tool_handle_key_event_backspace(text_box_tool, mock_element):
    """Test that backspace key is handled."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 4

    result = text_box_tool.handle_key_event(SketcherKey.BACKSPACE)

    assert result is True
    assert text_box_tool.text_buffer == "Tes"
    assert text_box_tool.cursor_pos == 3


def test_text_box_tool_handle_key_event_delete(text_box_tool, mock_element):
    """Test that delete key is handled."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 1

    result = text_box_tool.handle_key_event(SketcherKey.DELETE)

    assert result is True
    assert text_box_tool.text_buffer == "Tst"
    assert text_box_tool.cursor_pos == 1


def test_text_box_tool_handle_key_event_arrow_left(
    text_box_tool, mock_element
):
    """Test that arrow left key moves cursor."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 4

    result = text_box_tool.handle_key_event(SketcherKey.ARROW_LEFT)

    assert result is True
    assert text_box_tool.cursor_pos == 3
    mock_element.mark_dirty.assert_called()


def test_text_box_tool_handle_key_event_arrow_right(
    text_box_tool, mock_element
):
    """Test that arrow right key moves cursor."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 1

    result = text_box_tool.handle_key_event(SketcherKey.ARROW_RIGHT)

    assert result is True
    assert text_box_tool.cursor_pos == 2
    mock_element.mark_dirty.assert_called()


def test_text_box_tool_handle_key_event_return(text_box_tool, mock_element):
    """Test that return key finalizes edit."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.editing_entity_id = 5
    text_box_tool.text_buffer = "Test"

    mock_entity = Mock()
    mock_entity.id = 5
    mock_entity.font_params = {"family": "sans-serif", "size": 10.0}
    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    result = text_box_tool.handle_key_event(SketcherKey.RETURN)

    assert result is True
    assert text_box_tool.state == TextBoxState.IDLE
    assert text_box_tool.editing_entity_id is None
    assert text_box_tool.text_buffer == ""
    mock_element.execute_command.assert_called_once()


def test_text_box_tool_handle_key_event_escape(text_box_tool, mock_element):
    """Test that escape key cancels edit."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.editing_entity_id = 5
    text_box_tool.text_buffer = "Test"

    mock_entity = Mock()
    mock_entity.id = 5
    mock_entity.font_params = {"family": "sans-serif", "size": 10.0}
    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    result = text_box_tool.handle_key_event(SketcherKey.ESCAPE)

    assert result is True
    assert text_box_tool.state == TextBoxState.IDLE
    assert text_box_tool.editing_entity_id is None
    assert text_box_tool.text_buffer == ""


def test_text_box_tool_handle_key_event_idle_state(
    text_box_tool, mock_element
):
    """Test that key event is ignored in IDLE state."""
    text_box_tool.state = TextBoxState.IDLE

    result = text_box_tool.handle_key_event(SketcherKey.BACKSPACE)

    assert result is False
    mock_element.mark_dirty.assert_not_called()


def test_text_box_tool_handle_text_input_idle_state(
    text_box_tool, mock_element
):
    """Test that text input is ignored in IDLE state."""
    text_box_tool.state = TextBoxState.IDLE

    result = text_box_tool.handle_text_input("A")

    assert result is False
    mock_element.mark_dirty.assert_not_called()


def test_text_box_tool_start_editing(text_box_tool, mock_element):
    """Test starting to edit an existing text box."""
    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="Existing text", construction_line_ids=[]
    )
    mock_entity.font_params = {"family": "sans-serif", "size": 10.0}
    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    text_box_tool.start_editing(5)

    assert text_box_tool.state == TextBoxState.EDITING
    assert text_box_tool.editing_entity_id == 5
    assert text_box_tool.text_buffer == "Existing text"
    assert text_box_tool.cursor_pos == 13
    assert text_box_tool.cursor_visible is True
    mock_element.mark_dirty.assert_called()


def test_text_box_tool_on_deactivate(text_box_tool, mock_element):
    """Test that deactivating cleans up state."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.editing_entity_id = 5
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 4

    text_box_tool.on_deactivate()

    assert text_box_tool.state == TextBoxState.IDLE
    assert text_box_tool.editing_entity_id is None
    assert text_box_tool.text_buffer == ""
    assert text_box_tool.cursor_pos == 0


def test_text_box_tool_toggle_cursor_visibility(text_box_tool, mock_element):
    """Test toggling cursor visibility."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.cursor_visible = True

    text_box_tool.toggle_cursor_visibility()

    assert text_box_tool.cursor_visible is False
    mock_element.mark_dirty.assert_called()

    text_box_tool.toggle_cursor_visibility()

    assert text_box_tool.cursor_visible is True


def test_text_box_tool_on_drag_does_nothing(text_box_tool, mock_element):
    """Test that drag does nothing."""
    result = text_box_tool.on_drag(10, 20)

    assert result is None


def test_text_box_tool_on_release_does_nothing(text_box_tool, mock_element):
    """Test that release does nothing."""
    result = text_box_tool.on_release(10, 20)

    assert result is None


def test_text_box_tool_is_click_outside_box(text_box_tool, mock_element):
    """Test checking if click is outside box bounds."""
    text_box_tool.editing_entity_id = 5

    mock_entity = Mock()
    mock_entity.origin_id = 0
    mock_entity.width_id = 1
    mock_entity.height_id = 2

    mock_element.sketch.registry.get_entity.return_value = mock_entity
    mock_element.sketch.registry.get_point.side_effect = [
        Mock(x=0, y=0),
        Mock(x=50, y=0),
        Mock(x=0, y=10),
    ]
    mock_element.hittester.screen_to_model.return_value = (100, 100)

    result = text_box_tool._is_click_outside_box(100, 200)

    assert result is True


def test_text_box_tool_is_click_inside_box(text_box_tool, mock_element):
    """Test checking if click is inside box bounds."""
    text_box_tool.editing_entity_id = 5

    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="", construction_line_ids=[]
    )

    mock_element.sketch.registry.get_entity.return_value = mock_entity
    mock_element.sketch.registry.get_point.side_effect = [
        Mock(x=0, y=0),
        Mock(x=50, y=0),
        Mock(x=0, y=10),
    ]
    mock_element.hittester.screen_to_model.return_value = (25, 5)

    result = text_box_tool._is_click_outside_box(25, 5)

    assert result is False


def test_text_box_tool_draw_overlay_idle_state(text_box_tool, mock_element):
    """Test that draw_overlay does nothing in IDLE state."""
    ctx = MagicMock()

    text_box_tool.draw_overlay(ctx)

    ctx.save.assert_not_called()


def test_text_box_tool_draw_overlay_editing_state(text_box_tool, mock_element):
    """Test that draw_overlay draws in EDITING state."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.editing_entity_id = 5
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 4
    text_box_tool.cursor_visible = True

    mock_entity = Mock()
    mock_entity.origin_id = 0
    mock_entity.width_id = 1
    mock_entity.height_id = 2
    mock_entity.font_params = {
        "family": "sans-serif",
        "size": 10.0,
        "bold": False,
        "italic": False,
    }

    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)
    mock_element.sketch.registry.get_point.side_effect = [
        Mock(x=0, y=0),
        Mock(x=50, y=0),
        Mock(x=0, y=10),
    ]
    mock_matrix = Mock()
    mock_matrix.for_cairo.return_value = (1, 0, 0, 1, 0, 0)
    mock_element.hittester.get_model_to_screen_transform.return_value = (
        mock_matrix
    )
    mock_element.canvas = Mock()
    mock_element.canvas.get_view_scale.return_value = (1.0, 1.0)

    ctx = MagicMock()

    text_box_tool.draw_overlay(ctx)

    ctx.save.assert_called()
    ctx.restore.assert_called()
