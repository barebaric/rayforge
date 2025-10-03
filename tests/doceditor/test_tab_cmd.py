import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.layer import Layer
from rayforge.core.tab import Tab
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.tab_cmd import TabCmd
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
def tab_cmd(mock_editor):
    """Provides a TabCmd instance."""
    return TabCmd(mock_editor)


@pytest.fixture
def sample_workpiece(mock_editor):
    """Provides a sample WorkPiece instance."""
    layer = Layer(name="Test Layer")
    mock_editor.doc.add_child(layer)
    workpiece = WorkPiece(name="Test WP")
    layer.add_child(workpiece)
    return workpiece


def test_set_workpiece_tab_width(tab_cmd, sample_workpiece):
    """Test setting the tab width for a workpiece."""
    # Add some initial tabs
    initial_tabs = [Tab(width=1.0, segment_index=0, pos=0.5)]
    sample_workpiece.tabs = initial_tabs

    initial_history_len = len(tab_cmd._editor.history_manager.undo_stack)

    tab_cmd.set_workpiece_tab_width(sample_workpiece, 2.5)

    assert len(sample_workpiece.tabs) == 1
    assert sample_workpiece.tabs[0].width == 2.5
    hm = tab_cmd._editor.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1

    # Test undo
    hm.undo()
    assert sample_workpiece.tabs[0].width == 1.0


def test_set_workpiece_tab_width_no_change(tab_cmd, sample_workpiece):
    """Test that setting the same tab width does not create a command."""
    initial_tabs = [Tab(width=1.0, segment_index=0, pos=0.5)]
    sample_workpiece.tabs = initial_tabs

    initial_history_len = len(tab_cmd._editor.history_manager.undo_stack)

    tab_cmd.set_workpiece_tab_width(sample_workpiece, 1.0)

    hm = tab_cmd._editor.history_manager
    assert len(hm.undo_stack) == initial_history_len


def test_set_workpiece_tab_width_no_tabs(tab_cmd, sample_workpiece):
    """Test that setting width on a workpiece with no tabs does nothing."""
    initial_history_len = len(tab_cmd._editor.history_manager.undo_stack)

    tab_cmd.set_workpiece_tab_width(sample_workpiece, 2.5)

    hm = tab_cmd._editor.history_manager
    assert len(hm.undo_stack) == initial_history_len


def test_set_workpiece_tabs_enabled(tab_cmd, sample_workpiece):
    """Test enabling/disabling tabs for a workpiece."""
    initial_history_len = len(tab_cmd._editor.history_manager.undo_stack)
    initial_state = sample_workpiece.tabs_enabled

    # Toggle the state
    tab_cmd.set_workpiece_tabs_enabled(sample_workpiece, not initial_state)

    assert sample_workpiece.tabs_enabled is not initial_state
    hm = tab_cmd._editor.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1

    # Test undo
    hm.undo()
    assert sample_workpiece.tabs_enabled is initial_state


def test_set_workpiece_tabs_enabled_no_change(tab_cmd, sample_workpiece):
    """Test that setting the same enabled state does not create a command."""
    initial_state = sample_workpiece.tabs_enabled
    initial_history_len = len(tab_cmd._editor.history_manager.undo_stack)

    tab_cmd.set_workpiece_tabs_enabled(sample_workpiece, initial_state)

    hm = tab_cmd._editor.history_manager
    assert len(hm.undo_stack) == initial_history_len


def test_clear_tabs(tab_cmd, sample_workpiece):
    """Test clearing all tabs from a workpiece."""
    # Add some initial tabs
    initial_tabs = [Tab(width=1.0, segment_index=0, pos=0.5)]
    sample_workpiece.tabs = initial_tabs

    initial_history_len = len(tab_cmd._editor.history_manager.undo_stack)

    tab_cmd.clear_tabs(sample_workpiece)

    assert len(sample_workpiece.tabs) == 0
    hm = tab_cmd._editor.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1

    # Test undo
    hm.undo()
    assert len(sample_workpiece.tabs) == 1
