import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.layer_cmd import LayerCmd, AddLayerAndSetActiveCommand
from rayforge.shared.tasker.manager import TaskManager
from rayforge.config import ConfigManager


@pytest.fixture
def mock_editor():
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    config_manager = MagicMock(spec=ConfigManager)
    doc = Doc()
    return DocEditor(task_manager, config_manager, doc)


@pytest.fixture
def layer_cmd(mock_editor):
    """Provides a LayerCmd instance."""
    return LayerCmd(mock_editor)


@pytest.fixture
def sample_layer(mock_editor):
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
    assert len(layer_cmd._editor.history_manager.undo_stack) == 1
