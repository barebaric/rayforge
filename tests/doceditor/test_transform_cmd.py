import pytest
from unittest.mock import MagicMock, patch

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.layer import Layer
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.transform_cmd import TransformCmd
from rayforge.shared.tasker.manager import TaskManager


@pytest.fixture
def mock_editor(context_initializer):
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    doc = Doc()
    return DocEditor(task_manager, context_initializer, doc)


@pytest.fixture
def transform_cmd(mock_editor):
    """Provides a TransformCmd instance."""
    return TransformCmd(mock_editor)


@pytest.fixture
def sample_items(mock_editor):
    """Provides sample DocItem instances."""
    layer = Layer(name="Test Layer")
    mock_editor.doc.add_child(layer)
    item1 = WorkPiece(name="Item 1")
    item2 = WorkPiece(name="Item 2")
    item1.set_size(10, 10)
    item2.set_size(20, 20)
    item1.pos = (5, 5)
    item2.pos = (10, 10)
    layer.add_child(item1)
    layer.add_child(item2)
    return [item1, item2]


def test_set_position(transform_cmd, sample_items):
    """Test setting the position of items."""
    with patch(
        "rayforge.doceditor.transform_cmd.get_context"
    ) as mock_get_context:
        mock_get_context.return_value.machine = None
        hm = transform_cmd._editor.history_manager
        initial_history_len = len(hm.undo_stack)

        transform_cmd.set_position(sample_items, 15.0, 20.0)

        assert sample_items[0].pos == (15.0, 20.0)
        assert sample_items[1].pos == (15.0, 20.0)
        assert len(hm.undo_stack) == initial_history_len + 1
        hm.undo()
        assert sample_items[0].pos == (5.0, 5.0)


def test_set_position_with_y_axis_down(transform_cmd, sample_items):
    """Test setting position when Y-axis is down."""
    with patch(
        "rayforge.doceditor.transform_cmd.get_context"
    ) as mock_get_context:
        mock_machine = MagicMock()
        mock_machine.x_axis_right = False
        mock_machine.y_axis_down = True
        mock_machine.dimensions = (100, 100)
        mock_get_context.return_value.machine = mock_machine
        hm = transform_cmd._editor.history_manager
        initial_history_len = len(hm.undo_stack)

        # Y=0 in machine coords means top of the bed.
        # For item2 (size 20), bottom-left should be at (15, 80).
        transform_cmd.set_position([sample_items[1]], 15.0, 0.0)

        assert sample_items[1].pos == (15.0, 80.0)
        assert len(hm.undo_stack) == initial_history_len + 1


def test_set_size(transform_cmd, sample_items):
    """Test setting the size of items."""
    hm = transform_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)

    transform_cmd.set_size(sample_items, 30.0, 40.0)

    assert sample_items[0].size == (30.0, 40.0)
    assert sample_items[1].size == (30.0, 40.0)
    assert len(hm.undo_stack) == initial_history_len + 1
    hm.undo()
    assert sample_items[0].size == (10.0, 10.0)


def test_set_size_with_fixed_ratio(transform_cmd, sample_items):
    """Test setting size with fixed aspect ratio."""
    hm = transform_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)

    # item1 has ratio 1.0, item2 has ratio 1.0
    transform_cmd.set_size(sample_items, 30.0, None, fixed_ratio=True)

    assert sample_items[0].size == (30.0, 30.0)
    assert sample_items[1].size == (30.0, 30.0)
    assert len(hm.undo_stack) == initial_history_len + 1


def test_set_size_with_individual_sizes(transform_cmd, sample_items):
    """Test setting size with a list of individual sizes."""
    hm = transform_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)

    sizes = [(30.0, 40.0), (50.0, 60.0)]
    transform_cmd.set_size(sample_items, sizes=sizes)

    assert sample_items[0].size == (30.0, 40.0)
    assert sample_items[1].size == (50.0, 60.0)
    assert len(hm.undo_stack) == initial_history_len + 1


def test_set_size_with_mismatched_sizes(transform_cmd, sample_items, caplog):
    """Test that providing a mismatched sizes list logs an error."""
    transform_cmd.set_size(sample_items, sizes=[(10.0, 20.0)])
    assert "Length of sizes list must match" in caplog.text


def test_set_angle(transform_cmd, sample_items):
    """Test setting the angle of items."""
    hm = transform_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)
    initial_angle = sample_items[0].angle

    transform_cmd.set_angle(sample_items, 45.0)

    assert sample_items[0].angle == 45.0
    assert sample_items[1].angle == 45.0
    assert len(hm.undo_stack) == initial_history_len + 1
    hm.undo()
    assert sample_items[0].angle == initial_angle


def test_set_shear(transform_cmd, sample_items):
    """Test setting the shear of items."""
    hm = transform_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)
    initial_shear = sample_items[0].shear

    transform_cmd.set_shear(sample_items, 15.0)

    assert sample_items[0].shear == pytest.approx(15.0)
    assert sample_items[1].shear == pytest.approx(15.0)
    assert len(hm.undo_stack) == initial_history_len + 1
    hm.undo()
    assert sample_items[0].shear == initial_shear


def test_set_position_no_items(transform_cmd):
    """Test that set_position does nothing if no items are provided."""
    hm = transform_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)
    transform_cmd.set_position([], 10, 10)
    assert len(hm.undo_stack) == initial_history_len


def test_set_size_no_items(transform_cmd):
    """Test that set_size does nothing if no items are provided."""
    hm = transform_cmd._editor.history_manager
    initial_history_len = len(hm.undo_stack)
    transform_cmd.set_size([], 10, 10)
    assert len(hm.undo_stack) == initial_history_len
