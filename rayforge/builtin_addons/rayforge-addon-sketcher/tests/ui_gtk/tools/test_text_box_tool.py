from typing import cast
from unittest.mock import MagicMock, Mock, patch

import pytest
from raygeo.geo.shape.text import FontConfig
from sketcher.core import Sketch
from sketcher.core.commands import TextBoxCommand
from sketcher.core.commands.live_text_edit import LiveTextEditCommand
from sketcher.core.commands.text_property import ModifyTextPropertyCommand
from sketcher.core.entities import TextBoxEntity
from sketcher.ui_gtk.tools import TextBoxTool
from sketcher.ui_gtk.tools.base import SketcherKey
from sketcher.ui_gtk.tools.text_box_tool import TextBoxState


@pytest.fixture
def mock_element():
    """Create a mock SketchElement for testing tools."""
    element = Mock()
    element.sketch = Mock()
    element.sketch.registry._id_counter = 0
    element.sketch.registry.entities = []
    element.sketch.registry.points = []
    element.sketch.constraints = []
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


@pytest.mark.ui
def test_text_box_tool_initialization(text_box_tool):
    """Test tool's initial state."""
    assert text_box_tool.state == TextBoxState.IDLE
    assert text_box_tool.editing_entity_id is None
    assert text_box_tool.text_buffer == ""
    assert text_box_tool.cursor_pos == 0
    assert text_box_tool.cursor_visible is True


@pytest.mark.ui
def test_text_box_tool_on_press_creates_box(text_box_tool, mock_element):
    """Test that first press creates a text box and enters EDITING state."""
    mock_element.hittester.screen_to_model.return_value = (10, 20)
    mock_element.sketch.registry.entities = []

    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="", construction_line_ids=[]
    )

    def mock_execute(cmd):
        cmd.text_box_id = 5

    mock_element.execute_command.side_effect = mock_execute

    def get_entity_side_effect(eid):
        if eid == 5:
            return mock_entity
        return None

    mock_element.sketch.registry.get_entity.side_effect = (
        get_entity_side_effect
    )

    result = text_box_tool.on_press(100, 200, 1)

    assert result is True
    assert text_box_tool.state == TextBoxState.EDITING
    assert text_box_tool.editing_entity_id == 5
    assert text_box_tool.text_buffer == ""
    assert mock_element.execute_command.call_count == 1
    mock_element.mark_dirty.assert_called()


@pytest.mark.ui
def test_text_box_tool_on_press_outside_box_creates_new(
    text_box_tool, mock_element
):
    """Test that clicking outside box creates a new text box."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.editing_entity_id = 5
    text_box_tool.text_buffer = "Test Text"

    mock_element.hittester.screen_to_model.return_value = (1000, 1000)
    mock_element.content_transform.transform_point.return_value = (1000, 1000)
    mock_element.sketch.registry.entities = []

    mock_entity = MagicMock(spec=TextBoxEntity)
    mock_entity.origin_id = 0
    mock_entity.width_id = 1
    mock_entity.height_id = 2
    mock_entity.content = "Test Text"
    mock_entity.font_config = FontConfig(family="sans-serif", size=10.0)

    new_entity = TextBoxEntity(
        6, 0, 1, 2, content="", construction_line_ids=[]
    )

    def get_entity_side_effect(eid):
        if eid == 5:
            return mock_entity
        if eid == 6:
            return new_entity
        return None

    mock_element.sketch.registry.get_entity.side_effect = (
        get_entity_side_effect
    )

    def get_point_side_effect(pid):
        vals = {0: (0, 0), 1: (50, 0), 2: (0, 10)}
        if pid in vals:
            return Mock(x=vals[pid][0], y=vals[pid][1])
        return Mock(x=0, y=0)

    mock_element.sketch.registry.get_point.side_effect = get_point_side_effect

    def mock_execute(cmd):
        cmd.text_box_id = 6

    mock_element.execute_command.side_effect = mock_execute

    result = text_box_tool.on_press(100, 200, 1)

    assert result is True
    assert text_box_tool.state == TextBoxState.EDITING
    assert text_box_tool.editing_entity_id == 6
    assert text_box_tool.text_buffer == ""
    # Only the TextBoxCommand is executed: the previous session had no
    # finalize command (it was not started via start_editing) and the
    # session-scoped LiveTextEditCommand is never pushed.
    assert mock_element.execute_command.call_count == 1


@pytest.mark.ui
def test_text_box_tool_handle_text_input_appends_character(
    text_box_tool, mock_element
):
    """Test that text input appends character to buffer."""
    text_box_tool.state = TextBoxState.EDITING

    result = text_box_tool.handle_text_input("A")

    assert result is True
    assert text_box_tool.text_buffer == "A"
    assert text_box_tool.cursor_pos == 1


@pytest.mark.ui
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


@pytest.mark.ui
def test_text_box_tool_handle_key_event_backspace(text_box_tool, mock_element):
    """Test that backspace key is handled."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 4

    result = text_box_tool.handle_key_event(SketcherKey.BACKSPACE)

    assert result is True
    assert text_box_tool.text_buffer == "Tes"
    assert text_box_tool.cursor_pos == 3


@pytest.mark.ui
def test_text_box_tool_handle_key_event_delete(text_box_tool, mock_element):
    """Test that delete key is handled."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.text_buffer = "Test"
    text_box_tool.cursor_pos = 1

    result = text_box_tool.handle_key_event(SketcherKey.DELETE)

    assert result is True
    assert text_box_tool.text_buffer == "Tst"
    assert text_box_tool.cursor_pos == 1


@pytest.mark.ui
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


@pytest.mark.ui
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


@pytest.mark.ui
def test_text_box_tool_handle_key_event_return(text_box_tool, mock_element):
    """Test that return key finalizes edit."""
    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="Test", construction_line_ids=[]
    )
    mock_entity.font_config = FontConfig(family="sans-serif", size=10.0)
    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    text_box_tool.start_editing(5)

    result = text_box_tool.handle_key_event(SketcherKey.RETURN)

    assert result is True
    assert text_box_tool.state == TextBoxState.IDLE
    assert text_box_tool.editing_entity_id is None
    assert text_box_tool.text_buffer == ""
    mock_element.execute_command.assert_called_once()
    executed_cmd = mock_element.execute_command.call_args.args[0]
    assert executed_cmd.new_content == "Test"


@pytest.mark.ui
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


@pytest.mark.ui
def test_text_box_tool_handle_key_event_idle_state(
    text_box_tool, mock_element
):
    """Test that key event is ignored in IDLE state."""
    text_box_tool.state = TextBoxState.IDLE

    result = text_box_tool.handle_key_event(SketcherKey.BACKSPACE)

    assert result is False
    mock_element.mark_dirty.assert_not_called()


@pytest.mark.ui
def test_text_box_tool_handle_text_input_idle_state(
    text_box_tool, mock_element
):
    """Test that text input is ignored in IDLE state."""
    text_box_tool.state = TextBoxState.IDLE

    result = text_box_tool.handle_text_input("A")

    assert result is False
    mock_element.mark_dirty.assert_not_called()


@pytest.mark.ui
def test_text_box_tool_start_editing(text_box_tool, mock_element):
    """Test starting to edit an existing text box."""
    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="Existing text", construction_line_ids=[]
    )
    mock_entity.font_config = FontConfig(family="sans-serif", size=10.0)
    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    text_box_tool.start_editing(5)

    assert text_box_tool.state == TextBoxState.EDITING
    assert text_box_tool.editing_entity_id == 5
    assert text_box_tool.text_buffer == "Existing text"
    assert text_box_tool.cursor_pos == 13
    assert text_box_tool.cursor_visible is True
    mock_element.mark_dirty.assert_called()


@pytest.mark.ui
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


@pytest.mark.ui
def test_text_box_tool_deactivate_pushes_only_finalize_command(
    text_box_tool, mock_element
):
    """Ending an edit session must push exactly one command for the edit
    (ModifyTextPropertyCommand). The session-scoped LiveTextEditCommand
    must never become a global history entry."""
    entity = MagicMock(spec=TextBoxEntity)
    entity.content = "before"
    entity.font_config = FontConfig(family="sans-serif", size=10.0)
    entity.origin_id = 0
    entity.width_id = 1
    entity.height_id = 2
    entity.get_natural_size.return_value = (10.0, 5.0)
    entity.construction_line_ids = []
    mock_element.sketch.registry.get_entity.side_effect = lambda eid: (
        entity if eid == 5 else None
    )
    mock_element.sketch.registry.get_point.side_effect = lambda pid: Mock(
        x=0.0, y=0.0
    )
    mock_element.sketch.constraints = []

    text_box_tool.start_editing(5)
    assert text_box_tool.live_edit_cmd is not None

    text_box_tool.handle_text_input("!")
    text_box_tool.on_deactivate()

    executed = [
        call.args[0] for call in mock_element.execute_command.call_args_list
    ]
    assert not any(isinstance(cmd, LiveTextEditCommand) for cmd in executed)
    finalize_cmds = [
        cmd for cmd in executed if isinstance(cmd, ModifyTextPropertyCommand)
    ]
    assert len(finalize_cmds) == 1
    # old_content is captured from the entity when the command executes,
    # which the mocked execute_command does not do.
    assert finalize_cmds[0].new_content == "before!"
    assert text_box_tool.live_edit_cmd is None


@pytest.mark.ui
def test_text_box_tool_undo_restores_cursor_position(
    text_box_tool, mock_element
):
    """Undoing a mid-line edit must restore the cursor to the position
    the edit was made at, not to the stale session-start position."""

    class MockTime:
        current_time = 1000.0

        @classmethod
        def time(cls):
            return cls.current_time

    with patch("time.time", MockTime.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        mock_element.sketch = sketch

        assert box_cmd.text_box_id is not None
        entity_id = box_cmd.text_box_id
        text_box_tool.state = TextBoxState.EDITING
        text_box_tool.editing_entity_id = entity_id
        text_box_tool.text_buffer = "hello world"
        text_box_tool.cursor_pos = len(text_box_tool.text_buffer)
        text_box_tool.live_edit_cmd = LiveTextEditCommand(sketch, entity_id)
        text_box_tool.live_edit_cmd.capture_state(
            text_box_tool.text_buffer, text_box_tool.cursor_pos
        )

        # Move the cursor into the middle of the line (pure cursor move,
        # not an undoable action), then type there.
        text_box_tool.handle_key_event(SketcherKey.ARROW_LEFT)
        text_box_tool.handle_key_event(SketcherKey.ARROW_LEFT)
        assert text_box_tool.cursor_pos == 9

        # Advance time so the keystroke is not coalesced into the
        # pre-edit history entry.
        MockTime.current_time += 1.0
        text_box_tool.handle_text_input("X")
        assert text_box_tool.text_buffer == "hello worXld"

    text_box_tool.handle_key_event(SketcherKey.UNDO)

    assert text_box_tool.text_buffer == "hello world"
    assert text_box_tool.cursor_pos == 9


@pytest.mark.ui
def test_text_box_tool_undo_targets_session_start_geometry(
    text_box_tool, mock_element
):
    """The finalize command must record the box geometry as it was when
    the edit session started, not after the live resize that typing
    performs (which caused a non-uniform resize on undo)."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None
    mock_element.sketch = sketch

    entity_id = box_cmd.text_box_id
    entity = cast(TextBoxEntity, sketch.registry.get_entity(entity_id))

    text_box_tool.start_editing(entity_id)
    assert text_box_tool._edit_cmd is not None

    p_width = sketch.registry.get_point(entity.width_id)
    orig_width = (p_width.x, p_width.y)

    # Simulate the live resize while typing: the width point moves
    # before any command is executed.
    p_width.x += 30.0

    text_box_tool.text_buffer = "something"
    text_box_tool.on_deactivate()

    executed = [
        call.args[0] for call in mock_element.execute_command.call_args_list
    ]
    finalize = [
        cmd for cmd in executed if isinstance(cmd, ModifyTextPropertyCommand)
    ]
    assert len(finalize) == 1
    assert finalize[0].new_content == "something"
    assert finalize[0].old_point_positions[entity.width_id] == pytest.approx(
        orig_width
    )


@pytest.mark.ui
def test_text_box_tool_apply_font_change_returns_false_when_idle(
    text_box_tool, mock_element
):
    """apply_font_change must report it did not handle the change when no
    edit session is active so the caller commits its own command."""
    result = text_box_tool.apply_font_change(
        FontConfig(family="serif", size=12.0)
    )
    assert result is False


@pytest.mark.ui
def test_text_box_tool_apply_font_change_folds_into_finalize_command(
    text_box_tool, mock_element
):
    """A font change during an active edit session must not push a
    separate command: the live resize mutates geometry outside the
    command system, so a standalone command would snapshot the mutated
    state as its pre-state. The change is folded into the session's
    finalize command, whose pre-edit state was captured at session
    start."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None
    mock_element.sketch = sketch

    entity_id = box_cmd.text_box_id
    entity = cast(TextBoxEntity, sketch.registry.get_entity(entity_id))

    text_box_tool.start_editing(entity_id)
    assert text_box_tool._edit_cmd is not None

    # Simulate the live resize that typing performs: the width point has
    # moved away from its session-start position.
    p_width = sketch.registry.get_point(entity.width_id)
    orig_width = (p_width.x, p_width.y)
    p_width.x += 25.0

    new_font = FontConfig(family="serif", size=14.0, bold=True)
    handled = text_box_tool.apply_font_change(new_font)

    assert handled is True
    # No command was pushed for the font change.
    assert mock_element.execute_command.call_count == 0
    # The font change is staged on the finalize command and entity live.
    assert text_box_tool._edit_cmd.new_font_config == new_font
    assert entity.font_config == new_font

    # Finalize: the single pushed command carries the session-start
    # geometry as its pre-state, not the mutated one.
    text_box_tool.text_buffer = "hello"
    text_box_tool.on_deactivate()

    executed = [
        call.args[0] for call in mock_element.execute_command.call_args_list
    ]
    finalize = [
        cmd for cmd in executed if isinstance(cmd, ModifyTextPropertyCommand)
    ]
    assert len(finalize) == 1
    assert finalize[0].new_content == "hello"
    assert finalize[0].new_font_config == new_font
    assert finalize[0].old_point_positions[entity.width_id] == (
        pytest.approx(orig_width)
    )


@pytest.mark.ui
def test_text_box_tool_toggle_cursor_visibility(text_box_tool, mock_element):
    """Test toggling cursor visibility."""
    text_box_tool.state = TextBoxState.EDITING
    text_box_tool.cursor_visible = True

    text_box_tool.toggle_cursor_visibility()

    assert text_box_tool.cursor_visible is False
    mock_element.mark_dirty.assert_called()

    text_box_tool.toggle_cursor_visibility()

    assert text_box_tool.cursor_visible is True


@pytest.mark.ui
def test_text_box_tool_on_drag_does_nothing(text_box_tool, mock_element):
    """Test that drag does nothing."""
    result = text_box_tool.on_drag(10, 20)

    assert result is None


@pytest.mark.ui
def test_text_box_tool_on_release_does_nothing(text_box_tool, mock_element):
    """Test that release does nothing."""
    result = text_box_tool.on_release(10, 20)

    assert result is None


@pytest.mark.ui
def test_text_box_tool_is_click_outside_box(text_box_tool, mock_element):
    """Test checking if click is outside box bounds."""
    text_box_tool.editing_entity_id = 5

    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="", construction_line_ids=[]
    )
    mock_entity.font_config = FontConfig(family="sans-serif", size=10.0)

    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    def get_point_side_effect(pid):
        vals = {0: (0, 0), 1: (50, 0), 2: (0, 10)}
        if pid in vals:
            return Mock(x=vals[pid][0], y=vals[pid][1])
        return Mock(x=0, y=0)

    mock_element.sketch.registry.get_point.side_effect = get_point_side_effect

    mock_element.hittester.screen_to_model.return_value = (100, 100)

    result = text_box_tool._is_point_inside_box(100, 200)

    assert result is False


@pytest.mark.ui
def test_text_box_tool_is_click_inside_box(text_box_tool, mock_element):
    """Test checking if click is inside box bounds."""
    text_box_tool.editing_entity_id = 5

    mock_entity = TextBoxEntity(
        5, 0, 1, 2, content="", construction_line_ids=[]
    )
    mock_entity.font_config = FontConfig(family="sans-serif", size=10.0)

    mock_element.sketch.registry.get_entity = Mock(return_value=mock_entity)

    def get_point_side_effect(pid):
        vals = {0: (0, 0), 1: (50, 0), 2: (0, 10)}
        if pid in vals:
            return Mock(x=vals[pid][0], y=vals[pid][1])
        return Mock(x=0, y=0)

    mock_element.sketch.registry.get_point.side_effect = get_point_side_effect
    mock_element.hittester.screen_to_model.return_value = (25, 5)

    result = text_box_tool._is_point_inside_box(25, 5)

    assert result is True


@pytest.mark.ui
def test_text_box_tool_draw_overlay_idle_state(text_box_tool, mock_element):
    """Test that draw_overlay does nothing in IDLE state."""
    ctx = MagicMock()

    text_box_tool.draw_overlay(ctx)

    ctx.save.assert_not_called()


@pytest.mark.ui
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
    mock_entity.font_config = FontConfig(
        family="sans-serif",
        size=10.0,
        bold=False,
        italic=False,
    )
    mock_entity.get_font_metrics = Mock(return_value=(10.0, -2.0, 12.0))

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
