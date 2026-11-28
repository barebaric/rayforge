import pytest

from rayforge.core.setup import Setup
from rayforge.doceditor.setup_cmd import AddSetupAndSetActiveCommand, SetupCmd


@pytest.fixture
def setup_cmd(doc_editor):
    """Provides a SetupCmd instance."""
    return SetupCmd(doc_editor)


@pytest.fixture
def sample_setup(doc_editor):
    """Provides a sample Setup instance."""
    return Setup(name="Test Setup")


def test_add_setup_and_set_active_command_execute(setup_cmd, sample_setup):
    """Test that the command adds a layer and sets it as active."""
    initial_layer_count = len(setup_cmd._editor.doc.setups)
    cmd = AddSetupAndSetActiveCommand(
        setup_cmd._editor, sample_setup, name="Add layer"
    )
    cmd.execute()

    assert len(setup_cmd._editor.doc.setups) == initial_layer_count + 1
    assert setup_cmd._editor.doc.setups[-1] is sample_setup
    assert setup_cmd._editor.doc.active_setup is sample_setup


def test_add_setup_and_set_active_command_undo(setup_cmd, sample_setup):
    """Test that the command correctly undoes adding a layer."""
    initial_layer_count = len(setup_cmd._editor.doc.setups)
    initial_active_setup = setup_cmd._editor.doc.active_setup
    cmd = AddSetupAndSetActiveCommand(
        setup_cmd._editor, sample_setup, name="Add layer"
    )
    cmd.execute()
    cmd.undo()

    assert len(setup_cmd._editor.doc.setups) == initial_layer_count
    assert sample_setup not in setup_cmd._editor.doc.setups
    assert setup_cmd._editor.doc.active_setup is initial_active_setup


def test_setup_cmd_add_setup_and_set_active(setup_cmd, sample_setup):
    """Test that the SetupCmd wrapper correctly executes the command."""
    initial_layer_count = len(setup_cmd._editor.doc.setups)
    setup_cmd.add_setup_and_set_active(sample_setup)

    assert len(setup_cmd._editor.doc.setups) == initial_layer_count + 1
    assert setup_cmd._editor.doc.setups[-1] is sample_setup
    assert setup_cmd._editor.doc.active_setup is sample_setup
    assert len(setup_cmd._editor.history_manager.undo_stack) == 1


def test_setup_cmd_add_default_layer_and_set_active(setup_cmd):
    """
    Test that the SetupCmd wrapper creates a default layer if none is
    provided.
    """
    initial_layer_count = len(setup_cmd._editor.doc.setups)
    setup_cmd.add_setup_and_set_active()

    assert len(setup_cmd._editor.doc.setups) == initial_layer_count + 1
    assert setup_cmd._editor.doc.active_setup.name.startswith("Setup")
    assert len(setup_cmd._editor.history_manager.undo_stack) == 1


def test_setup_cmd_set_active_setup(setup_cmd):
    """Test setting the active layer."""
    layer1 = Setup(name="Layer 1")
    layer2 = Setup(name="Layer 2")
    setup_cmd._editor.doc.add_child(layer1)
    setup_cmd._editor.doc.add_child(layer2)
    setup_cmd._editor.doc.active_setup = layer1

    setup_cmd.set_active_setup(layer2)

    assert setup_cmd._editor.doc.active_setup is layer2
    assert len(setup_cmd._editor.history_manager.undo_stack) == 1


def test_setup_cmd_set_active_setup_no_change(setup_cmd):
    """Test that setting the same active layer does nothing."""
    layer1 = Setup(name="Layer 1")
    setup_cmd._editor.doc.add_child(layer1)
    setup_cmd._editor.doc.active_setup = layer1

    hm = setup_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)
    setup_cmd.set_active_setup(layer1)

    assert setup_cmd._editor.doc.active_setup is layer1
    assert len(hm.undo_stack) == initial_history_len


def test_setup_cmd_delete_setup(setup_cmd):
    """Test deleting a layer."""
    layer1 = Setup(name="Layer 1")
    layer2 = Setup(name="Layer 2")
    setup_cmd._editor.doc.add_child(layer1)
    setup_cmd._editor.doc.add_child(layer2)
    initial_count = len(setup_cmd._editor.doc.setups)

    setup_cmd.delete_setup(layer1)

    assert len(setup_cmd._editor.doc.setups) == initial_count - 1
    assert layer1 not in setup_cmd._editor.doc.setups
    assert len(setup_cmd._editor.history_manager.undo_stack) == 1


def test_setup_cmd_reorder_setups(setup_cmd):
    """Test reordering layers."""
    layer1 = Setup(name="Layer 1")
    layer2 = Setup(name="Layer 2")
    setup_cmd._editor.doc.add_child(layer1)
    setup_cmd._editor.doc.add_child(layer2)

    new_order = [layer2, layer1]
    setup_cmd.reorder_setups(new_order)

    assert setup_cmd._editor.doc.setups == new_order


def test_setup_cmd_set_setup_stock_material(setup_cmd, sample_setup):
    """Test setting a layer's stock material is undoable and dirties."""
    hm = setup_cmd._editor.history_manager
    setup_cmd.set_setup_stock_material(sample_setup, "mat-1")

    assert sample_setup.stock_material_uid == "mat-1"
    assert len(hm.undo_stack) == 1
    assert not setup_cmd._editor.is_saved

    hm.undo()
    assert sample_setup.stock_material_uid is None


def test_setup_cmd_set_setup_stock_material_no_change(setup_cmd, sample_setup):
    """Test that setting the same material does nothing."""
    hm = setup_cmd._editor.history_manager
    setup_cmd.set_setup_stock_material(sample_setup, None)

    assert len(hm.undo_stack) == 0
