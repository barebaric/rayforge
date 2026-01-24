import pytest
from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import ModifyTextPropertyCommand
from rayforge.core.sketcher.entities.text_box import TextBoxEntity
from rayforge.core.undo import HistoryManager


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
        content="Original Text",
        font_params={
            "font_family": "sans-serif",
            "font_size": 10.0,
            "bold": False,
            "italic": False,
        },
    )

    return sketch, tb_id


def test_modify_text_property_command_initialization(
    sketch_with_text_box,
):
    """Test that ModifyTextPropertyCommand initializes correctly."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_params = {
        "font_family": "serif",
        "font_size": 14.0,
        "bold": True,
        "italic": False,
    }

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_params
    )

    assert cmd.sketch is sketch
    assert cmd.text_entity_id == tb_id
    assert cmd.new_content == new_content
    assert cmd.new_font_params == new_font_params
    assert cmd.old_content == ""
    assert cmd.old_font_params == {}


def test_modify_text_property_command_execute(sketch_with_text_box):
    """Test that execute updates the text entity properties."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_params = {
        "font_family": "serif",
        "font_size": 14.0,
        "bold": True,
        "italic": False,
    }

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_params
    )

    tb = sketch.registry.get_entity(tb_id)
    assert tb.content == "Original Text"
    assert tb.font_params["font_family"] == "sans-serif"

    cmd.execute()

    assert tb.content == new_content
    assert tb.font_params == new_font_params
    assert cmd.old_content == "Original Text"
    assert cmd.old_font_params["font_family"] == "sans-serif"


def test_modify_text_property_command_undo(sketch_with_text_box):
    """Test that undo restores the original text entity properties."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_params = {
        "font_family": "serif",
        "font_size": 14.0,
        "bold": True,
        "italic": False,
    }

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_params
    )

    cmd.execute()

    tb = sketch.registry.get_entity(tb_id)
    assert tb.content == new_content
    assert tb.font_params == new_font_params

    cmd.undo()

    assert tb.content == "Original Text"
    assert tb.font_params["font_family"] == "sans-serif"
    assert tb.font_params["font_size"] == 10.0


def test_modify_text_property_command_execute_undo_cycle(sketch_with_text_box):
    """Test that execute and undo can be called multiple times."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_params = {
        "font_family": "serif",
        "font_size": 14.0,
        "bold": True,
        "italic": False,
    }

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_params
    )

    for _ in range(3):
        cmd.execute()
        tb = sketch.registry.get_entity(tb_id)
        assert tb.content == new_content
        assert tb.font_params == new_font_params

        cmd.undo()
        tb = sketch.registry.get_entity(tb_id)
        assert tb.content == "Original Text"
        assert tb.font_params["font_family"] == "sans-serif"


def test_modify_text_property_command_with_missing_entity(
    sketch_with_text_box,
):
    """Test that execute handles missing entity gracefully."""
    sketch, _ = sketch_with_text_box

    cmd = ModifyTextPropertyCommand(sketch, 9999, "New Text", {})

    cmd.execute()
    assert cmd.old_content == ""
    assert cmd.old_font_params == {}

    cmd.undo()
    assert cmd.old_content == ""
    assert cmd.old_font_params == {}


def test_modify_text_property_command_full_font_update(
    sketch_with_text_box,
):
    """Test that command replaces entire font_params."""
    sketch, tb_id = sketch_with_text_box

    new_font_params = {
        "font_family": "serif",
        "font_size": 14.0,
        "bold": True,
        "italic": True,
    }

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "New Text", new_font_params)

    cmd.execute()

    tb = sketch.registry.get_entity(tb_id)
    assert tb.font_params["font_family"] == "serif"
    assert tb.font_params["font_size"] == 14.0
    assert tb.font_params["bold"] is True
    assert tb.font_params["italic"] is True


def test_text_property_command_undo_with_history_manager():
    """
    Test that text property command works correctly with history manager.
    This is an integration test that verifies the full undo flow.
    """
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_params={
            "font_family": "sans-serif",
            "font_size": 10.0,
            "bold": False,
            "italic": False,
        },
    )

    history = HistoryManager()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Original Text"
    assert tb.font_params["font_size"] == 10.0

    new_content = "New Text"
    new_font_params = {
        "font_family": "serif",
        "font_size": 14.0,
        "bold": True,
        "italic": False,
    }

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_params
    )

    history.execute(cmd)

    assert history.can_undo()
    assert not history.can_redo()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == new_content
    assert tb.font_params == new_font_params

    history.undo()

    assert not history.can_undo()
    assert history.can_redo()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Original Text"
    assert tb.font_params["font_family"] == "sans-serif"
    assert tb.font_params["font_size"] == 10.0
    assert tb.font_params["bold"] is False


def test_text_property_command_redo_after_undo():
    """
    Test that text property command can be redone after undo.
    """
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    history = HistoryManager()

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, "Modified", {"font_family": "serif", "font_size": 12.0}
    )

    history.execute(cmd)
    history.undo()
    history.redo()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Modified"
    assert tb.font_params["font_family"] == "serif"
    assert tb.font_params["font_size"] == 12.0


def test_text_property_command_multiple_edits_with_history():
    """
    Test that multiple text edits can be undone and redone correctly.
    """
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Initial",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    history = HistoryManager()

    cmd1 = ModifyTextPropertyCommand(
        sketch,
        tb_id,
        "First Edit",
        {"font_family": "serif", "font_size": 12.0},
    )
    cmd2 = ModifyTextPropertyCommand(
        sketch,
        tb_id,
        "Second Edit",
        {"font_family": "monospace", "font_size": 14.0},
    )

    history.execute(cmd1)
    history.execute(cmd2)

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Second Edit"
    assert tb.font_params["font_family"] == "monospace"

    history.undo()
    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "First Edit"
    assert tb.font_params["font_family"] == "serif"

    history.undo()
    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Initial"
    assert tb.font_params["font_family"] == "sans-serif"

    history.redo()
    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "First Edit"
    assert tb.font_params["font_family"] == "serif"


def test_should_skip_undo_both_empty():
    """Test should_skip_undo returns True when both contents empty."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})

    assert cmd.should_skip_undo() is True


def test_should_skip_undo_old_empty_new_not_empty():
    """Test should_skip_undo returns False when old empty, new not empty."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "New Text", {})

    assert cmd.should_skip_undo() is False


def test_should_skip_undo_old_not_empty_new_empty():
    """Test should_skip_undo returns False when old not empty, new empty."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})
    cmd.execute()

    assert cmd.should_skip_undo() is False


def test_should_skip_undo_both_not_empty():
    """Test should_skip_undo returns False when both contents not empty."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "New Text", {})

    assert cmd.should_skip_undo() is False


def test_empty_text_box_removed_on_execute():
    """Test that text box is removed when content becomes empty."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})
    cmd.execute()

    assert cmd._entity_was_removed is True
    assert cmd._removed_entity is not None
    assert cmd._removed_entity.id == tb_id

    tb = sketch.registry.get_entity(tb_id)
    assert tb is None


def test_empty_text_box_points_removed():
    """Test that associated points are removed when text box is removed."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})
    cmd.execute()

    assert len(cmd._removed_points) == 3

    point_ids = {pt.id for pt in cmd._removed_points}
    for pt in sketch.registry.points:
        assert pt.id not in point_ids


def test_empty_text_box_constraints_removed():
    """Test that constraints depending on removed points are removed."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})
    cmd.execute()

    point_ids = {pt.id for pt in cmd._removed_points}

    for constr in sketch.constraints or []:
        assert not constr.depends_on_points(point_ids)


def test_empty_text_box_undo_restores_entity():
    """Test that undo restores the removed text entity."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})
    cmd.execute()

    assert sketch.registry.get_entity(tb_id) is None

    cmd.undo()

    tb = sketch.registry.get_entity(tb_id)
    assert tb is not None
    assert isinstance(tb, TextBoxEntity)


def test_empty_text_box_undo_restores_points():
    """Test that undo restores the removed points."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})
    cmd.execute()

    point_ids = {pt.id for pt in cmd._removed_points}
    for pt in sketch.registry.points:
        assert pt.id not in point_ids

    cmd.undo()

    for pt in cmd._removed_points:
        restored = sketch.registry.get_point(pt.id)
        assert restored is not None
        assert restored.id == pt.id


def test_empty_text_box_with_history_manager():
    """Test empty text box removal works with history manager."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_params={"font_family": "sans-serif", "font_size": 10.0},
    )

    history = HistoryManager()

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", {})
    history.execute(cmd)

    assert history.can_undo() is True
    assert sketch.registry.get_entity(tb_id) is None

    history.undo()

    assert history.can_redo() is True
    tb = sketch.registry.get_entity(tb_id)
    assert tb is not None
