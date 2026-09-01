import pytest
from raygeo.geo.shape.text import FontConfig
from sketcher.core import Sketch
from sketcher.core.commands import ModifyTextPropertyCommand, TextBoxCommand
from sketcher.core.entities.text_box import TextBoxEntity

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
        font_config=FontConfig(
            family="sans-serif",
            size=10.0,
            bold=False,
            italic=False,
        ),
    )

    return sketch, tb_id


def test_modify_text_property_command_initialization(
    sketch_with_text_box,
):
    """Test that ModifyTextPropertyCommand initializes correctly."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_config = FontConfig(
        family="serif",
        size=14.0,
        bold=True,
        italic=False,
    )

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_config
    )

    assert cmd.sketch is sketch
    assert cmd.text_entity_id == tb_id
    assert cmd.new_content == new_content
    assert cmd.new_font_config == new_font_config
    assert cmd.old_content == ""
    assert cmd.old_font_config is None


def test_modify_text_property_command_execute(sketch_with_text_box):
    """Test that execute updates the text entity properties."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_config = FontConfig(
        family="serif",
        size=14.0,
        bold=True,
        italic=False,
    )

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_config
    )

    tb = sketch.registry.get_entity(tb_id)
    assert tb.content == "Original Text"
    assert tb.font_config.family == "sans-serif"

    cmd.execute()

    assert tb.content == new_content
    assert tb.font_config == new_font_config
    assert cmd.old_content == "Original Text"
    assert cmd.old_font_config is not None
    assert cmd.old_font_config.family == "sans-serif"
    assert cmd.old_font_config.size == 10.0


def test_modify_text_property_command_undo(sketch_with_text_box):
    """Test that undo restores the original text entity properties."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_config = FontConfig(
        family="serif",
        size=14.0,
        bold=True,
        italic=False,
    )

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_config
    )

    cmd.execute()

    tb = sketch.registry.get_entity(tb_id)
    assert tb.content == new_content
    assert tb.font_config == new_font_config

    cmd.undo()

    assert tb.content == "Original Text"
    assert tb.font_config.family == "sans-serif"
    assert tb.font_config.size == 10.0
    assert tb.font_config.bold is False


def test_modify_text_property_command_execute_undo_cycle(sketch_with_text_box):
    """Test that execute and undo can be called multiple times."""
    sketch, tb_id = sketch_with_text_box

    new_content = "New Text"
    new_font_config = FontConfig(
        family="serif",
        size=14.0,
        bold=True,
        italic=False,
    )

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_config
    )

    for _ in range(3):
        cmd.execute()
        tb = sketch.registry.get_entity(tb_id)
        assert tb.content == new_content
        assert tb.font_config == new_font_config

        cmd.undo()
        tb = sketch.registry.get_entity(tb_id)
        assert tb.content == "Original Text"
        assert tb.font_config.family == "sans-serif"
        assert tb.font_config.size == 10.0
        assert tb.font_config.bold is False
        assert tb.font_config.size == 10.0
        assert tb.font_config.bold is False


def test_modify_text_property_undo_restores_pre_session_geometry(
    sketch_with_text_box,
):
    """Box geometry mutated live during a typing session (before the
    command executes) must not become the undo target: undo restores
    the geometry captured at session start."""
    sketch, tb_id = sketch_with_text_box
    tb = sketch.registry.get_entity(tb_id)
    p_width = sketch.registry.get_point(tb.width_id)
    p_height = sketch.registry.get_point(tb.height_id)
    orig_width = (p_width.x, p_width.y)
    orig_height = (p_height.x, p_height.y)

    cmd = ModifyTextPropertyCommand(
        sketch,
        tb_id,
        "Much Longer Text",
        FontConfig(family="sans-serif", size=10.0),
    )

    # Session start: freeze pre-edit state, then simulate the live
    # resize that typing performs outside the command system.
    cmd.capture_undo_state()
    cmd.capture_pre_edit_state()

    p_width.x += 40.0
    p_height.y += 15.0

    cmd.execute()
    assert cmd.old_content == "Original Text"
    assert cmd.old_point_positions[tb.width_id] == pytest.approx(orig_width)
    assert cmd.old_point_positions[tb.height_id] == pytest.approx(orig_height)

    cmd.undo()

    assert (p_width.x, p_width.y) == pytest.approx(orig_width)
    assert (p_height.x, p_height.y) == pytest.approx(orig_height)


def test_modify_text_property_command_with_missing_entity(
    sketch_with_text_box,
):
    """Test that execute handles missing entity gracefully."""
    sketch, _ = sketch_with_text_box

    cmd = ModifyTextPropertyCommand(sketch, 9999, "New Text", FontConfig())

    cmd.execute()
    assert cmd.old_content == ""
    assert cmd.old_font_config is None

    cmd.undo()
    assert cmd.old_content == ""
    assert cmd.old_font_config is None


def test_modify_text_property_command_full_font_update(
    sketch_with_text_box,
):
    """Test that command replaces entire font_config."""
    sketch, tb_id = sketch_with_text_box

    new_font_config = FontConfig(
        family="serif",
        size=14.0,
        bold=True,
        italic=True,
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "New Text", new_font_config)

    cmd.execute()

    tb = sketch.registry.get_entity(tb_id)
    assert tb.font_config.family == "serif"
    assert tb.font_config.size == 14.0
    assert tb.font_config.bold is True
    assert tb.font_config.italic is True


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
        font_config=FontConfig(
            family="sans-serif",
            size=10.0,
            bold=False,
            italic=False,
        ),
    )

    history = HistoryManager()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Original Text"
    assert tb.font_config.size == 10.0

    new_content = "New Text"
    new_font_config = FontConfig(
        family="serif",
        size=14.0,
        bold=True,
        italic=False,
    )

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, new_content, new_font_config
    )

    history.execute(cmd)

    assert history.can_undo()
    assert not history.can_redo()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == new_content
    assert tb.font_config == new_font_config

    history.undo()

    assert not history.can_undo()
    assert history.can_redo()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Original Text"
    assert tb.font_config.family == "sans-serif"
    assert tb.font_config.size == 10.0
    assert tb.font_config.bold is False
    assert tb.font_config.size == 10.0
    assert tb.font_config.bold is False
    assert tb.font_config.size == 10.0
    assert tb.font_config.bold is False


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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    history = HistoryManager()

    cmd = ModifyTextPropertyCommand(
        sketch,
        tb_id,
        "Modified",
        FontConfig(family="serif", size=12.0),
    )

    history.execute(cmd)
    history.undo()
    history.redo()

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Modified"
    assert tb.font_config.family == "serif"
    assert tb.font_config.size == 12.0


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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    history = HistoryManager()

    cmd1 = ModifyTextPropertyCommand(
        sketch,
        tb_id,
        "First Edit",
        FontConfig(family="serif", size=12.0),
    )
    cmd2 = ModifyTextPropertyCommand(
        sketch,
        tb_id,
        "Second Edit",
        FontConfig(family="monospace", size=14.0),
    )

    history.execute(cmd1)
    history.execute(cmd2)

    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Second Edit"
    assert tb.font_config.family == "monospace"

    history.undo()
    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "First Edit"
    assert tb.font_config.family == "serif"

    history.undo()
    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "Initial"
    assert tb.font_config.family == "sans-serif"
    assert tb.font_config.size == 10.0
    assert tb.font_config.bold is False
    assert tb.font_config.size == 10.0
    assert tb.font_config.bold is False
    assert tb.font_config.size == 10.0
    assert tb.font_config.bold is False

    history.redo()
    tb = sketch.registry.get_entity(tb_id)
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "First Edit"
    assert tb.font_config.family == "serif"


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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "New Text", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())
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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "New Text", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

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
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    history = HistoryManager()

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

    history.execute(cmd)

    assert history.can_undo() is True
    assert sketch.registry.get_entity(tb_id) is None

    history.undo()

    assert history.can_redo() is True
    tb = sketch.registry.get_entity(tb_id)
    assert tb is not None
    assert isinstance(tb, TextBoxEntity)


def test_empty_text_box_undo_restores_constraints():
    """Test that undo restores the removed constraints."""
    sketch = Sketch()

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="Original Text",
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(sketch, tb_id, "", FontConfig())

    cmd.execute()

    point_ids = {pt.id for pt in cmd._removed_points}

    for constr in sketch.constraints or []:
        assert not constr.depends_on_points(point_ids)

    cmd.undo()

    for pt in cmd._removed_points:
        restored = sketch.registry.get_point(pt.id)
        assert restored is not None
        assert restored.id == pt.id


def test_undo_removes_text_box_when_reverting_to_empty():
    """
    Test that undo removes the text box when reverting to empty content.
    This handles the case where user types text into an empty box, then
    undoes - the box should be removed, not left empty.
    """
    sketch = Sketch()

    initial_point_count = len(sketch.registry.points)

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="",
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, "New Text", FontConfig(family="sans-serif")
    )

    cmd.execute()

    tb = sketch.registry.get_entity(tb_id)
    assert tb is not None
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "New Text"

    cmd.undo()

    assert sketch.registry.get_entity(tb_id) is None
    assert len(sketch.registry.points) == initial_point_count


def test_redo_restores_text_box_after_undo_to_empty():
    """
    Test that redo restores the text box after undoing to empty content.
    Uses HistoryManager to properly test redo behavior.
    """
    sketch = Sketch()

    initial_point_count = len(sketch.registry.points)

    p_origin = sketch.add_point(0, 0)
    p_width = sketch.add_point(50, 0)
    p_height = sketch.add_point(0, 10)

    tb_id = sketch.registry.add_text_box(
        p_origin,
        p_width,
        p_height,
        content="",
        font_config=FontConfig(family="sans-serif", size=10.0),
    )

    history = HistoryManager()

    cmd = ModifyTextPropertyCommand(
        sketch, tb_id, "New Text", FontConfig(family="sans-serif")
    )

    history.execute(cmd)

    tb = sketch.registry.get_entity(tb_id)
    assert tb is not None
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "New Text"

    history.undo()

    assert sketch.registry.get_entity(tb_id) is None
    assert len(sketch.registry.points) == initial_point_count

    history.redo()

    tb = sketch.registry.get_entity(tb_id)
    assert tb is not None
    assert isinstance(tb, TextBoxEntity)
    assert tb.content == "New Text"


def test_create_box_transaction_undo_redo_no_duplicate_ids():
    """Undoing and redoing a create-text-box transaction must not
    duplicate registry entries. Both commands in the transaction own
    the same points/entities; a non-idempotent restore produced two
    objects per ID, which corrupted sketches after save/reload."""
    sketch = Sketch()
    history = HistoryManager()

    with history.transaction("Add Text Box") as t:
        box_cmd = TextBoxCommand(sketch, origin=(0, 0), width=30.0)
        t.execute(box_cmd)
        assert box_cmd.text_box_id is not None
        fin_cmd = ModifyTextPropertyCommand(
            sketch,
            box_cmd.text_box_id,
            "abc",
            FontConfig(family="sans-serif", size=10.0),
        )
        t.execute(fin_cmd)

    def assert_no_duplicates():
        pids = [p.id for p in sketch.registry.points]
        eids = [e.id for e in sketch.registry.entities]
        assert len(pids) == len(set(pids))
        assert len(eids) == len(set(eids))

    assert_no_duplicates()
    history.undo()
    assert_no_duplicates()
    history.redo()
    assert_no_duplicates()

    text_boxes = [
        e for e in sketch.registry.entities if isinstance(e, TextBoxEntity)
    ]
    assert len(text_boxes) == 1
    assert text_boxes[0].content == "abc"

    # A second cycle must stay consistent as well.
    history.undo()
    history.redo()
    assert_no_duplicates()
    assert sketch.solve() is True
