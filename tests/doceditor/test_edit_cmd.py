import pytest
from unittest.mock import MagicMock

from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.edit_cmd import EditCmd
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.group import Group
from rayforge.core.item import DocItem
from rayforge.core.layer import Layer
from rayforge.shared.tasker.manager import TaskManager


@pytest.fixture
def mock_editor(context_initializer):
    """Provides a DocEditor instance with a clean document."""
    task_manager = MagicMock(spec=TaskManager)
    # Using a real Doc and HistoryManager is better for these tests
    doc = Doc()
    return DocEditor(task_manager, context_initializer, doc)


@pytest.fixture
def edit_cmd(mock_editor):
    """Provides an EditCmd instance linked to the mock_editor."""
    return EditCmd(mock_editor)


@pytest.fixture
def items_on_layer(mock_editor):
    """
    Creates a set of nested items on the default layer for testing.
    Structure:
    - Layer
      - wp1 (WorkPiece)
      - group1 (Group)
        - wp2 (WorkPiece)
        - group2 (Group)
          - wp3 (WorkPiece)
      - wp4 (WorkPiece)
    Returns a dictionary of these items.
    """
    layer = mock_editor.doc.active_layer
    wp1 = WorkPiece(name="wp1")
    wp1.pos = (10, 10)
    group1 = Group(name="group1")
    group1.pos = (100, 100)
    wp2 = WorkPiece(name="wp2")
    wp2.pos = (20, 20)  # Local to group1
    group2 = Group(name="group2")
    group2.pos = (200, 200)  # Local to group1
    wp3 = WorkPiece(name="wp3")
    wp3.pos = (30, 30)  # Local to group2
    wp4 = WorkPiece(name="wp4")
    wp4.pos = (50, 50)

    layer.add_child(wp1)
    layer.add_child(group1)
    group1.add_child(wp2)
    group1.add_child(group2)
    group2.add_child(wp3)
    layer.add_child(wp4)

    return {
        "layer": layer,
        "wp1": wp1,
        "group1": group1,
        "wp2": wp2,
        "group2": group2,
        "wp3": wp3,
        "wp4": wp4,
    }


def test_initial_state(edit_cmd: EditCmd):
    """Test the initial state of the EditCmd handler."""
    assert not edit_cmd.can_paste()
    assert edit_cmd._clipboard_snapshot == []
    assert edit_cmd._paste_counter == 0


def test_get_top_level_items(edit_cmd: EditCmd, items_on_layer):
    """Test the logic for identifying top-level items in a selection."""
    items = items_on_layer
    wp1, group1, wp2, group2, wp3, wp4 = (
        items["wp1"],
        items["group1"],
        items["wp2"],
        items["group2"],
        items["wp3"],
        items["wp4"],
    )

    # Test case 1: Siblings
    selection1 = [wp1, group1, wp4]
    top_level1 = edit_cmd._get_top_level_items(selection1)
    assert set(top_level1) == set(selection1)

    # Test case 2: Parent and child
    selection2 = [group1, wp2]
    top_level2 = edit_cmd._get_top_level_items(selection2)
    assert top_level2 == [group1]

    # Test case 3: Parent and grandchild
    selection3 = [group1, wp3]
    top_level3 = edit_cmd._get_top_level_items(selection3)
    assert top_level3 == [group1]

    # Test case 4: Grandparent, parent, and child
    selection4 = [group1, group2, wp3]
    top_level4 = edit_cmd._get_top_level_items(selection4)
    assert top_level4 == [group1]

    # Test case 5: Sibling and child of another sibling
    selection5 = [wp1, wp2]
    top_level5 = edit_cmd._get_top_level_items(selection5)
    assert set(top_level5) == set(selection5)

    # Test case 6: Empty list
    assert edit_cmd._get_top_level_items([]) == []


def test_copy_paste(edit_cmd: EditCmd, mock_editor: DocEditor):
    """Test basic copy and multiple paste operations."""
    layer = mock_editor.doc.active_layer
    wp1 = WorkPiece(name="wp1")
    wp1.pos = (10, 20)
    edit_cmd.add_items([wp1])

    assert len(layer.get_content_items()) == 1

    # 1. Copy
    edit_cmd.copy_items([wp1])
    assert edit_cmd.can_paste()
    assert edit_cmd._paste_counter == 1  # For a copy, next paste is offset.

    # 2. First Paste
    pasted1_list = edit_cmd.paste_items()
    assert len(pasted1_list) == 1
    pasted1 = pasted1_list[0]
    assert len(layer.get_content_items()) == 2
    assert pasted1 in layer.get_content_items()
    assert pasted1.uid != wp1.uid
    assert pasted1.name == wp1.name
    # Check offset position
    dx, dy = edit_cmd._paste_increment_mm
    assert pasted1.pos[0] == pytest.approx(wp1.pos[0] + dx)
    assert pasted1.pos[1] == pytest.approx(wp1.pos[1] + dy)
    assert edit_cmd._paste_counter == 2  # Incremented for next paste

    # 3. Second Paste
    pasted2_list = edit_cmd.paste_items()
    assert len(pasted2_list) == 1
    pasted2 = pasted2_list[0]
    assert len(layer.get_content_items()) == 3
    # Check offset position (cumulative)
    assert pasted2.pos[0] == pytest.approx(wp1.pos[0] + 2 * dx)
    assert pasted2.pos[1] == pytest.approx(wp1.pos[1] + 2 * dy)
    assert edit_cmd._paste_counter == 3


def test_copy_paste_nested(edit_cmd: EditCmd, items_on_layer):
    """Test copying and pasting a group with children."""
    layer = items_on_layer["layer"]
    group1 = items_on_layer["group1"]
    wp2_original = items_on_layer["wp2"]  # child
    wp3_original = items_on_layer["wp3"]  # grandchild

    edit_cmd.copy_items([group1])
    pasted_list = edit_cmd.paste_items()

    assert len(pasted_list) == 1
    pasted_group = pasted_list[0]
    assert isinstance(pasted_group, Group)
    assert pasted_group.uid != group1.uid
    assert pasted_group.parent == layer

    # Check children and grandchildren UIDs
    assert len(pasted_group.children) == 2
    assert pasted_group.find_descendant_by_uid(wp2_original.uid) is None
    assert pasted_group.find_descendant_by_uid(wp3_original.uid) is None

    # Verify new UIDs were assigned down the tree
    all_descendants = [pasted_group] + pasted_group.get_descendants()
    original_uids = {
        group1.uid,
        wp2_original.uid,
        items_on_layer["group2"].uid,
        wp3_original.uid,
    }
    assert all(item.uid not in original_uids for item in all_descendants)
    assert all(isinstance(item, DocItem) for item in all_descendants)


def test_cut_paste_with_undo(edit_cmd: EditCmd, mock_editor: DocEditor):
    """Test cut, paste, and undo cycle."""
    layer = mock_editor.doc.active_layer
    wp1 = WorkPiece(name="wp1")
    wp1.pos = (100, 200)
    edit_cmd.add_items([wp1])

    original_count = len(layer.get_content_items())
    original_item_uid = wp1.uid

    # 1. Cut
    edit_cmd.cut_items([wp1])
    assert len(layer.get_content_items()) == original_count - 1
    assert wp1 not in layer.children
    assert edit_cmd.can_paste()
    assert edit_cmd._paste_counter == 0  # Counter reset for in-place paste

    # 2. First Paste
    pasted1_list = edit_cmd.paste_items()
    assert len(pasted1_list) == 1
    pasted1 = pasted1_list[0]
    assert len(layer.get_content_items()) == original_count
    assert pasted1.pos == pytest.approx((100, 200))
    assert edit_cmd._paste_counter == 1  # Incremented for next paste

    # 3. Undo Paste
    mock_editor.history_manager.undo()
    assert len(layer.get_content_items()) == original_count - 1
    assert pasted1 not in layer.children

    # 4. Undo Cut
    mock_editor.history_manager.undo()
    assert len(layer.get_content_items()) == original_count
    restored_item = layer.children[-1]
    assert restored_item.uid == original_item_uid
    assert restored_item.pos == pytest.approx((100, 200))


def test_duplicate(edit_cmd: EditCmd, mock_editor: DocEditor):
    """Test duplicating items, which should create copies in the same place."""
    layer = mock_editor.doc.active_layer
    wp1 = WorkPiece(name="wp1")
    wp1.pos = (50, -50)
    edit_cmd.add_items([wp1])

    assert len(layer.get_content_items()) == 1

    duplicated_list = edit_cmd.duplicate_items([wp1])

    assert len(duplicated_list) == 1
    duplicated_wp = duplicated_list[0]

    assert len(layer.get_content_items()) == 2
    assert duplicated_wp in layer.children
    assert duplicated_wp.uid != wp1.uid
    assert duplicated_wp.name == wp1.name
    # Position must be identical to the original
    assert duplicated_wp.pos == wp1.pos


def test_remove_items_with_undo(
    edit_cmd: EditCmd, items_on_layer, mock_editor: DocEditor
):
    """Test removing a top-level item and ensure its children are detached."""
    layer = items_on_layer["layer"]
    wp1 = items_on_layer["wp1"]
    group1 = items_on_layer["group1"]
    wp2 = items_on_layer["wp2"]  # Child of group1

    initial_content = list(layer.get_content_items())
    initial_count = len(initial_content)

    # Remove group1 and wp1, which are siblings at the top level of the layer
    edit_cmd.remove_items([group1, wp1])

    assert len(layer.get_content_items()) == initial_count - 2
    assert group1 not in layer.children
    assert wp1 not in layer.children
    # The removed items' parent pointers are now None.
    assert group1.parent is None
    assert wp1.parent is None
    # The internal structure of the removed group remains intact.
    assert wp2.parent == group1

    # Test undo
    mock_editor.history_manager.undo()
    assert len(layer.get_content_items()) == initial_count
    assert set(layer.get_content_items()) == set(initial_content)
    # The parent pointers are correctly restored.
    assert group1.parent == layer
    assert wp1.parent == layer
    assert wp2.parent == group1  # Child's parent is still the group


def test_clear_all_items(edit_cmd: EditCmd, mock_editor: DocEditor):
    """Test removing all items from all layers."""
    # Setup: two layers with items
    layer1 = mock_editor.doc.layers[0]
    layer2 = Layer(name="Layer 2")
    mock_editor.doc.add_layer(layer2)

    wp1 = WorkPiece(name="L1 WP1")
    g1 = Group(name="L1 G1")
    layer1.add_child(wp1)
    layer1.add_child(g1)

    wp2 = WorkPiece(name="L2 WP2")
    layer2.add_child(wp2)

    assert len(layer1.get_content_items()) == 2
    assert len(layer2.get_content_items()) == 1

    # Action
    edit_cmd.clear_all_items()

    # Assert
    assert len(layer1.get_content_items()) == 0
    assert len(layer2.get_content_items()) == 0
    # Make sure workflows are preserved
    assert layer1.workflow is not None
    assert len(layer1.children) == 1
    assert layer2.workflow is not None
    assert len(layer2.children) == 1

    # Test undo
    mock_editor.history_manager.undo()
    assert len(layer1.get_content_items()) == 2
    assert len(layer2.get_content_items()) == 1
    assert wp1 in layer1.children
    assert g1 in layer1.children
    assert wp2 in layer2.children


def test_reset_paste_counter(edit_cmd: EditCmd):
    """Test that the paste counter can be reset."""
    edit_cmd.copy_items([WorkPiece(name="dummy")])
    assert edit_cmd._paste_counter == 1

    edit_cmd.paste_items()
    assert edit_cmd._paste_counter == 2

    edit_cmd.reset_paste_counter()
    assert edit_cmd._paste_counter == 0

    # After a reset, the next paste should be "in place"
    # (i.e., offset by 0 * increment).
    # paste_items then increments the counter for the *next* paste.
    edit_cmd.paste_items()
    assert edit_cmd._paste_counter == 1
