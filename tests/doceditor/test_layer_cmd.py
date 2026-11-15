import pytest

from rayforge.core.layer import Layer
from rayforge.core.layer_types import (
    LayerType,
    layer_type_registry,
    locked_layer_notice,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.layer_cmd import (
    AddLayerAndSetActiveCommand,
    LayerCmd,
)

LOCKED_LAYER_TEST_ADDON = "layer_cmd_test_addon"


@pytest.fixture
def layer_cmd(doc_editor):
    """Provides a LayerCmd instance."""
    return LayerCmd(doc_editor)


@pytest.fixture
def sample_layer(doc_editor):
    """Provides a sample Layer instance."""
    return Layer(name="Test Layer")


def test_add_layer_and_set_active_command_execute(layer_cmd, sample_layer):
    """Test that the command adds a layer and sets it as active."""
    initial_layer_count = len(layer_cmd._editor.doc.layers)
    cmd = AddLayerAndSetActiveCommand(
        layer_cmd._editor, sample_layer, name="Add layer"
    )
    cmd.execute()

    assert len(layer_cmd._editor.doc.layers) == initial_layer_count + 1
    assert layer_cmd._editor.doc.layers[-1] is sample_layer
    assert layer_cmd._editor.doc.active_layer is sample_layer


def test_add_layer_and_set_active_command_undo(layer_cmd, sample_layer):
    """Test that the command correctly undoes adding a layer."""
    initial_layer_count = len(layer_cmd._editor.doc.layers)
    initial_active_layer = layer_cmd._editor.doc.active_layer
    cmd = AddLayerAndSetActiveCommand(
        layer_cmd._editor, sample_layer, name="Add layer"
    )
    cmd.execute()
    cmd.undo()

    assert len(layer_cmd._editor.doc.layers) == initial_layer_count
    assert sample_layer not in layer_cmd._editor.doc.layers
    assert layer_cmd._editor.doc.active_layer is initial_active_layer


def test_layer_cmd_add_layer_and_set_active(layer_cmd, sample_layer):
    """Test that the LayerCmd wrapper correctly executes the command."""
    initial_layer_count = len(layer_cmd._editor.doc.layers)
    layer_cmd.add_layer_and_set_active(sample_layer)

    assert len(layer_cmd._editor.doc.layers) == initial_layer_count + 1
    assert layer_cmd._editor.doc.layers[-1] is sample_layer
    assert layer_cmd._editor.doc.active_layer is sample_layer
    assert len(layer_cmd._editor.history_manager.undo_stack) == 1


def test_layer_cmd_add_default_layer_and_set_active(layer_cmd):
    """
    Test that the LayerCmd wrapper creates a default layer if none is
    provided.
    """
    initial_layer_count = len(layer_cmd._editor.doc.layers)
    layer_cmd.add_layer_and_set_active()

    assert len(layer_cmd._editor.doc.layers) == initial_layer_count + 1
    assert layer_cmd._editor.doc.active_layer.name.startswith("Layer")
    assert len(layer_cmd._editor.history_manager.undo_stack) == 1


def test_layer_cmd_set_active_layer(layer_cmd):
    """Test setting the active layer."""
    layer1 = Layer(name="Layer 1")
    layer2 = Layer(name="Layer 2")
    layer_cmd._editor.doc.add_child(layer1)
    layer_cmd._editor.doc.add_child(layer2)
    layer_cmd._editor.doc.active_layer = layer1

    layer_cmd.set_active_layer(layer2)

    assert layer_cmd._editor.doc.active_layer is layer2
    assert len(layer_cmd._editor.history_manager.undo_stack) == 1


def test_layer_cmd_set_active_layer_no_change(layer_cmd):
    """Test that setting the same active layer does nothing."""
    layer1 = Layer(name="Layer 1")
    layer_cmd._editor.doc.add_child(layer1)
    layer_cmd._editor.doc.active_layer = layer1

    hm = layer_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)
    layer_cmd.set_active_layer(layer1)

    assert layer_cmd._editor.doc.active_layer is layer1
    assert len(hm.undo_stack) == initial_history_len


def test_layer_cmd_delete_layer(layer_cmd):
    """Test deleting a layer."""
    layer1 = Layer(name="Layer 1")
    layer2 = Layer(name="Layer 2")
    layer_cmd._editor.doc.add_child(layer1)
    layer_cmd._editor.doc.add_child(layer2)
    initial_count = len(layer_cmd._editor.doc.layers)

    layer_cmd.delete_layer(layer1)

    assert len(layer_cmd._editor.doc.layers) == initial_count - 1
    assert layer1 not in layer_cmd._editor.doc.layers
    assert len(layer_cmd._editor.history_manager.undo_stack) == 1


def test_layer_cmd_reorder_layers(layer_cmd):
    """Test reordering layers."""
    layer1 = Layer(name="Layer 1")
    layer2 = Layer(name="Layer 2")
    layer_cmd._editor.doc.add_child(layer1)
    layer_cmd._editor.doc.add_child(layer2)

    new_order = [layer2, layer1]
    layer_cmd.reorder_layers(new_order)

    assert layer_cmd._editor.doc.layers == new_order


def test_layer_cmd_set_layer_stock_material(layer_cmd, sample_layer):
    """Test setting a layer's stock material is undoable and dirties."""
    hm = layer_cmd._editor.history_manager
    layer_cmd.set_layer_stock_material(sample_layer, "mat-1")

    assert sample_layer.stock_material_uid == "mat-1"
    assert len(hm.undo_stack) == 1
    assert not layer_cmd._editor.is_saved

    hm.undo()
    assert sample_layer.stock_material_uid is None


def test_layer_cmd_set_layer_stock_material_no_change(layer_cmd, sample_layer):
    """Test that setting the same material does nothing."""
    hm = layer_cmd._editor.history_manager
    layer_cmd.set_layer_stock_material(sample_layer, None)

    assert len(hm.undo_stack) == 0


@pytest.fixture
def locked_layer(doc_editor):
    """Provides a layer whose type refuses other workpieces."""
    layer_type_registry.register(
        LayerType(id="locked_test_kind", title="Test Kind", locked=True),
        LOCKED_LAYER_TEST_ADDON,
    )
    layer = Layer(name="Test Kind Layer")
    layer.layer_type = "locked_test_kind"
    doc_editor.doc.add_child(layer)
    yield layer
    layer_type_registry.unregister_all_from_addon(LOCKED_LAYER_TEST_ADDON)


def _track_notifications(editor) -> list:
    messages = []
    editor.notification_requested.connect(
        lambda sender, **kwargs: messages.append(kwargs.get("message")),
        weak=False,
    )
    return messages


def test_move_items_to_locked_layer_is_refused(layer_cmd, locked_layer):
    """Test that a locked layer does not take workpieces from others."""
    editor = layer_cmd._editor
    source = editor.doc.active_layer
    wp = WorkPiece(name="wp")
    source.add_child(wp)
    hm = editor.history_manager
    initial_history_len = len(hm.undo_stack)

    layer_cmd.move_items_to_layer([wp], locked_layer)

    assert wp.parent is source
    assert len(hm.undo_stack) == initial_history_len


def test_move_items_to_locked_layer_notifies(layer_cmd, locked_layer):
    """Test that the refusal is explained to the user."""
    editor = layer_cmd._editor
    source = editor.doc.active_layer
    wp = WorkPiece(name="wp")
    source.add_child(wp)
    messages = _track_notifications(editor)

    layer_cmd.move_items_to_layer([wp], locked_layer)

    assert messages == [locked_layer_notice(locked_layer)]


def test_move_items_to_unlocked_layer_still_works(layer_cmd, locked_layer):
    """Test that ordinary layers keep accepting workpieces."""
    editor = layer_cmd._editor
    source = editor.doc.active_layer
    wp = WorkPiece(name="wp")
    source.add_child(wp)

    other = Layer(name="Other Layer")
    editor.doc.add_child(other)
    layer_cmd.move_items_to_layer([wp], other)

    assert wp.parent is other
