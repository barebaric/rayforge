from typing import cast
from unittest.mock import patch

from sketcher.core import Sketch
from sketcher.core.commands import TextBoxCommand
from sketcher.core.commands.live_text_edit import LiveTextEditCommand
from sketcher.core.entities import TextBoxEntity

from rayforge.core.undo.history import COALESCE_THRESHOLD


class MockTime:
    """Helper class to mock time.time() for testing coalescing."""

    def __init__(self):
        self.current_time = 1000.0

    def time(self):
        return self.current_time

    def advance(self, seconds):
        self.current_time += seconds


def test_live_text_edit_command_initialization():
    """Test command initialization."""
    sketch = Sketch()
    cmd = LiveTextEditCommand(sketch, 1)

    assert cmd.text_entity_id == 1
    assert cmd._sketch is sketch
    assert cmd.history == []
    assert cmd.current_index == -1
    assert cmd.cursor_pos == 0


def test_live_text_edit_command_execute():
    """Test command execution captures initial state."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "initial"

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        assert len(cmd.history) == 1
        assert cmd.current_index == 0
        assert cmd.get_current_content() == "initial"


def test_live_text_edit_execute_preserves_session_history():
    """Executing an already-active session must not wipe its history,
    otherwise undo would lose the pre-edit state."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None
        text_box_id = box_cmd.text_box_id

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.capture_state("before", 6)

        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "before edited"

        cmd.execute()

        assert len(cmd.history) == 1
        assert cmd.current_index == 0
        assert cmd.get_current_content() == "before"


def test_live_text_edit_should_skip_undo():
    """The session-scoped command is never a valid global-history entry."""
    sketch = Sketch()
    cmd = LiveTextEditCommand(sketch, 1)

    assert cmd.should_skip_undo() is True


def test_live_text_edit_update_cursor_syncs_tip():
    """Cursor movements must update the current tip without creating
    history entries, so undo restores the expected cursor position."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None
        text_box_id = box_cmd.text_box_id

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.capture_state("hello", 5)

        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.update_cursor(2)
        assert len(cmd.history) == 1
        assert cmd.get_current_content() == "hello"
        assert cmd.get_current_cursor_pos() == 2

        cmd.capture_state("hello world", 11)
        assert len(cmd.history) == 2
        assert cmd.history[0] == ("hello", 2, cmd.history[0][2])

        cmd.undo()
        assert cmd.get_current_content() == "hello"
        assert cmd.get_current_cursor_pos() == 2


def test_live_text_edit_update_cursor_empty_history():
    """update_cursor on an uninitialized session must be a no-op."""
    sketch = Sketch()
    cmd = LiveTextEditCommand(sketch, 1)

    cmd.update_cursor(3)

    assert cmd.history == []
    assert cmd.current_index == -1


def test_live_text_edit_capture_state():
    """Test capturing state updates history."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.capture_state("hello", 5)
        assert len(cmd.history) == 2
        assert cmd.current_index == 1
        assert cmd.get_current_content() == "hello"
        assert cmd.get_current_cursor_pos() == 5


def test_live_text_edit_undo():
    """Test undo functionality."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "initial"

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.capture_state("hello", 5)
        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.capture_state("hello world", 11)

        assert cmd.get_current_content() == "hello world"

        cmd.undo()
        assert cmd.get_current_content() == "hello"

        cmd.undo()
        assert cmd.get_current_content() == "initial"


def test_live_text_edit_redo():
    """Test redo functionality."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "initial"

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.capture_state("hello", 5)
        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.capture_state("hello world", 11)

        cmd.undo()
        assert cmd.get_current_content() == "hello"

        cmd.redo()
        assert cmd.get_current_content() == "hello world"


def test_live_text_edit_coalesce_rapid_keystrokes():
    """Test that rapid keystrokes are coalesced into one history entry,
    while the pre-edit baseline stays separately undoable."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        initial_len = len(cmd.history)
        baseline_content = cmd.get_current_content()

        cmd.capture_state("h", 1)
        mock_time.advance(0.05)
        cmd.capture_state("he", 2)
        mock_time.advance(0.05)
        cmd.capture_state("hel", 3)
        mock_time.advance(0.05)
        cmd.capture_state("hell", 4)
        mock_time.advance(0.05)
        cmd.capture_state("hello", 5)

        assert len(cmd.history) == initial_len + 1
        assert cmd.get_current_content() == "hello"

        cmd.undo()
        assert cmd.get_current_content() == baseline_content


def test_live_text_edit_coalesce_after_pause():
    """Test that a pause creates a new history entry."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        initial_len = len(cmd.history)

        cmd.capture_state("h", 1)
        mock_time.advance(0.05)
        cmd.capture_state("he", 2)
        mock_time.advance(0.05)
        cmd.capture_state("hel", 3)

        mock_time.advance(COALESCE_THRESHOLD + 0.1)

        cmd.capture_state("hell", 4)
        mock_time.advance(0.05)
        cmd.capture_state("hello", 5)

        assert len(cmd.history) == initial_len + 2
        assert cmd.get_current_content() == "hello"


def test_live_text_edit_first_keystroke_keeps_baseline_undoable():
    """Typing immediately (< COALESCE_THRESHOLD) after session start must
    not overwrite the pre-edit baseline: the first undo restores it."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "initial"

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        cmd.capture_state("initial h", 9)
        mock_time.advance(0.05)
        cmd.capture_state("initial he", 10)

        assert cmd.get_current_content() == "initial he"

        cmd.undo()
        assert cmd.get_current_content() == "initial"
        assert cmd.current_index == 0


def test_live_text_edit_coalesce_undo_through_coalesced():
    """Test undo works correctly with coalesced states."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "initial"

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.capture_state("h", 1)
        mock_time.advance(0.05)
        cmd.capture_state("he", 2)
        mock_time.advance(0.05)
        cmd.capture_state("hel", 3)

        mock_time.advance(COALESCE_THRESHOLD + 0.1)

        cmd.capture_state("hell", 4)
        mock_time.advance(0.05)
        cmd.capture_state("hello", 5)

        assert cmd.get_current_content() == "hello"

        cmd.undo()
        assert cmd.get_current_content() == "hel"

        cmd.undo()
        assert cmd.get_current_content() == "initial"


def test_live_text_edit_coalesce_redo_through_coalesced():
    """Test redo works correctly with coalesced states."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "initial"

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        cmd.capture_state("h", 1)
        mock_time.advance(0.05)
        cmd.capture_state("he", 2)
        mock_time.advance(0.05)
        cmd.capture_state("hel", 3)

        mock_time.advance(COALESCE_THRESHOLD + 0.1)

        cmd.capture_state("hell", 4)
        mock_time.advance(0.05)
        cmd.capture_state("hello", 5)

        cmd.undo()
        assert cmd.get_current_content() == "hel"

        cmd.redo()
        assert cmd.get_current_content() == "hello"


def test_live_text_edit_restore_state():
    """Test that _restore_state correctly updates entity content."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
        text_box.content = "initial"

        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        mock_time.advance(COALESCE_THRESHOLD + 0.1)
        cmd.capture_state("hello", 5)

        text_box.content = "modified"

        cmd._restore_state(1)

        assert text_box.content == "hello"


def test_live_text_edit_get_current_content_empty():
    """Test get_current_content returns empty string when no history."""
    sketch = Sketch()
    cmd = LiveTextEditCommand(sketch, 1)

    assert cmd.get_current_content() == ""


def test_live_text_edit_get_current_cursor_pos_zero():
    """Test get_current_cursor_pos returns 0 when no history."""
    sketch = Sketch()
    cmd = LiveTextEditCommand(sketch, 1)

    assert cmd.get_current_cursor_pos() == 0


def test_live_text_edit_execute_with_invalid_entity():
    """Test execute handles invalid entity gracefully."""
    sketch = Sketch()
    cmd = LiveTextEditCommand(sketch, 999)

    cmd.execute()

    assert len(cmd.history) == 0


def test_live_text_edit_restore_state_with_invalid_entity():
    """Test _restore_state handles invalid entity gracefully."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        cmd = LiveTextEditCommand(sketch, 999)
        cmd.history = [("test", 4, mock_time.time())]

        cmd._restore_state(0)

        assert cmd.history[0] == ("test", 4, cmd.history[0][2])


def test_live_text_edit_undo_at_start():
    """Test undo does nothing when at start of history."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        initial_content = cmd.get_current_content()
        cmd.undo()

        assert cmd.get_current_content() == initial_content


def test_live_text_edit_redo_at_end():
    """Test redo does nothing when at end of history."""
    mock_time = MockTime()
    with patch("time.time", mock_time.time):
        sketch = Sketch()
        box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
        box_cmd.execute()
        assert box_cmd.text_box_id is not None

        text_box_id = box_cmd.text_box_id
        cmd = LiveTextEditCommand(sketch, text_box_id)
        cmd.execute()

        cmd.capture_state("hello", 5)

        initial_content = cmd.get_current_content()
        cmd.redo()

        assert cmd.get_current_content() == initial_content
