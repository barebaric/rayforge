import time
from typing import cast
from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import TextBoxCommand
from rayforge.core.sketcher.commands.live_text_edit import LiveTextEditCommand
from rayforge.core.sketcher.entities import TextBoxEntity
from rayforge.core.undo.history import COALESCE_THRESHOLD


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


def test_live_text_edit_capture_state():
    """Test capturing state updates history."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None

    text_box_id = box_cmd.text_box_id
    cmd = LiveTextEditCommand(sketch, text_box_id)
    cmd.execute()

    time.sleep(COALESCE_THRESHOLD + 0.1)
    cmd.capture_state("hello", 5)
    assert len(cmd.history) == 2
    assert cmd.current_index == 1
    assert cmd.get_current_content() == "hello"
    assert cmd.get_current_cursor_pos() == 5


def test_live_text_edit_undo():
    """Test undo functionality."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None

    text_box_id = box_cmd.text_box_id
    text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
    text_box.content = "initial"

    cmd = LiveTextEditCommand(sketch, text_box_id)
    cmd.execute()

    time.sleep(COALESCE_THRESHOLD + 0.1)
    cmd.capture_state("hello", 5)
    time.sleep(COALESCE_THRESHOLD + 0.1)
    cmd.capture_state("hello world", 11)

    assert cmd.get_current_content() == "hello world"

    cmd.undo()
    assert cmd.get_current_content() == "hello"

    cmd.undo()
    assert cmd.get_current_content() == "initial"


def test_live_text_edit_redo():
    """Test redo functionality."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None

    text_box_id = box_cmd.text_box_id
    text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
    text_box.content = "initial"

    cmd = LiveTextEditCommand(sketch, text_box_id)
    cmd.execute()

    time.sleep(COALESCE_THRESHOLD + 0.1)
    cmd.capture_state("hello", 5)
    time.sleep(COALESCE_THRESHOLD + 0.1)
    cmd.capture_state("hello world", 11)

    cmd.undo()
    assert cmd.get_current_content() == "hello"

    cmd.redo()
    assert cmd.get_current_content() == "hello world"


def test_live_text_edit_coalesce_rapid_keystrokes():
    """Test that rapid keystrokes are coalesced into one history entry."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None

    text_box_id = box_cmd.text_box_id
    cmd = LiveTextEditCommand(sketch, text_box_id)
    cmd.execute()

    initial_len = len(cmd.history)

    cmd.capture_state("h", 1)
    time.sleep(0.05)
    cmd.capture_state("he", 2)
    time.sleep(0.05)
    cmd.capture_state("hel", 3)
    time.sleep(0.05)
    cmd.capture_state("hell", 4)
    time.sleep(0.05)
    cmd.capture_state("hello", 5)

    assert len(cmd.history) == initial_len
    assert cmd.get_current_content() == "hello"


def test_live_text_edit_coalesce_after_pause():
    """Test that a pause creates a new history entry."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None

    text_box_id = box_cmd.text_box_id
    cmd = LiveTextEditCommand(sketch, text_box_id)
    cmd.execute()

    initial_len = len(cmd.history)

    cmd.capture_state("h", 1)
    time.sleep(0.05)
    cmd.capture_state("he", 2)
    time.sleep(0.05)
    cmd.capture_state("hel", 3)

    time.sleep(COALESCE_THRESHOLD + 0.1)

    cmd.capture_state("hell", 4)
    time.sleep(0.05)
    cmd.capture_state("hello", 5)

    assert len(cmd.history) == initial_len + 1
    assert cmd.get_current_content() == "hello"


def test_live_text_edit_coalesce_undo_through_coalesced():
    """Test undo works correctly with coalesced states."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None

    text_box_id = box_cmd.text_box_id
    text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
    text_box.content = "initial"

    cmd = LiveTextEditCommand(sketch, text_box_id)
    cmd.execute()

    time.sleep(COALESCE_THRESHOLD + 0.1)
    cmd.capture_state("h", 1)
    time.sleep(0.05)
    cmd.capture_state("he", 2)
    time.sleep(0.05)
    cmd.capture_state("hel", 3)

    time.sleep(COALESCE_THRESHOLD + 0.1)

    cmd.capture_state("hell", 4)
    time.sleep(0.05)
    cmd.capture_state("hello", 5)

    assert cmd.get_current_content() == "hello"

    cmd.undo()
    assert cmd.get_current_content() == "hel"

    cmd.undo()
    assert cmd.get_current_content() == "initial"


def test_live_text_edit_coalesce_redo_through_coalesced():
    """Test redo works correctly with coalesced states."""
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
    time.sleep(0.05)
    cmd.capture_state("he", 2)
    time.sleep(0.05)
    cmd.capture_state("hel", 3)

    time.sleep(COALESCE_THRESHOLD + 0.1)

    cmd.capture_state("hell", 4)
    time.sleep(0.05)
    cmd.capture_state("hello", 5)

    cmd.undo()
    assert cmd.get_current_content() == "hel"

    cmd.redo()
    assert cmd.get_current_content() == "hello"


def test_live_text_edit_restore_state():
    """Test that _restore_state correctly updates entity content."""
    sketch = Sketch()
    box_cmd = TextBoxCommand(sketch, (0, 0), 10.0, 10.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None

    text_box_id = box_cmd.text_box_id
    text_box = cast(TextBoxEntity, sketch.registry.get_entity(text_box_id))
    text_box.content = "initial"

    cmd = LiveTextEditCommand(sketch, text_box_id)
    cmd.execute()

    time.sleep(COALESCE_THRESHOLD + 0.1)
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
    sketch = Sketch()
    cmd = LiveTextEditCommand(sketch, 999)
    cmd.history = [("test", 4, time.time())]

    cmd._restore_state(0)

    assert cmd.history[0] == ("test", 4, cmd.history[0][2])


def test_live_text_edit_undo_at_start():
    """Test undo does nothing when at start of history."""
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
