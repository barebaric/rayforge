import pytest

from rayforge.core.sketcher import Sketch
from rayforge.core.sketcher.commands import ModifyTextPropertyCommand


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
