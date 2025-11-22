import pytest

from rayforge.doceditor.editor import DocEditor
from rayforge.core.workpiece import WorkPiece
from rayforge.core.matrix import Matrix
from rayforge.core.group import Group
from rayforge.core.item import DocItem


@pytest.fixture
def doc_editor(task_mgr, context_initializer):
    """
    Provides a DocEditor instance using the standard, fully-initialized
    test context provided by the `context_initializer` fixture.
    """
    editor = DocEditor(task_manager=task_mgr, context=context_initializer)
    yield editor
    # The editor's cleanup handles its internal resources like the pipeline.
    # The fixtures will handle their own teardown (context, task manager).
    editor.cleanup()


@pytest.mark.asyncio
async def test_group_and_ungroup_preserves_transform(doc_editor: DocEditor):
    """
    Tests that grouping and then ungrouping items restores their original
    local transformations relative to their parent layer. This is the core
    test for the user-reported bug.
    """
    # 1. Setup: Create items with known transforms in the active layer
    layer = doc_editor.doc.active_layer

    wp1 = WorkPiece(name="wp1.svg")
    # A simple translation and scale
    wp1.matrix = Matrix.translation(10, 20) @ Matrix.scale(5, 5)
    layer.add_child(wp1)

    wp2 = WorkPiece(name="wp2.svg")
    # A more complex transform with rotation
    wp2.matrix = (
        Matrix.translation(50, 60) @ Matrix.rotation(45) @ Matrix.scale(8, 2)
    )
    layer.add_child(wp2)

    # Capture original state (local matrices and world transforms for
    # good measure)
    original_wp1_matrix = wp1.matrix.copy()
    original_wp2_matrix = wp2.matrix.copy()
    original_wp1_world = wp1.get_world_transform()
    original_wp2_world = wp2.get_world_transform()

    # 2. Group the items using the command handler
    # Explicitly type hint the list to satisfy type checkers (List[Sub] is not
    # a subtype of List[Base]).
    items_to_group: list[DocItem] = [wp1, wp2]
    doc_editor.group.group_items(layer, items_to_group)
    await doc_editor.wait_until_settled(timeout=2)

    # Assert that grouping worked as expected
    assert len(layer.get_content_items()) == 1
    group = layer.get_content_items()[0]
    assert isinstance(group, Group)
    assert len(group.children) == 2
    assert wp1 in group.children
    assert wp2 in group.children

    # World transforms must be preserved after grouping
    assert wp1.get_world_transform() == original_wp1_world
    assert wp2.get_world_transform() == original_wp2_world

    # 3. Ungroup the items using the command handler
    doc_editor.group.ungroup_items([group])
    await doc_editor.wait_until_settled()

    # 4. Assert: Check if items are back in the layer with original transforms
    content_items = layer.get_content_items()
    assert len(content_items) == 2
    assert wp1 in content_items
    assert wp2 in content_items
    assert group.parent is None  # Group should be detached

    # The critical check: are the local matrices restored?
    assert wp1.matrix == original_wp1_matrix
    assert wp2.matrix == original_wp2_matrix

    # Double check world transforms as well
    assert wp1.get_world_transform() == original_wp1_world
    assert wp2.get_world_transform() == original_wp2_world
