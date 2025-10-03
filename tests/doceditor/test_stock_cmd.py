import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.stock import StockItem
from rayforge.core.geo import Geometry
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.stock_cmd import StockCmd
from rayforge.shared.tasker.manager import TaskManager


@pytest.fixture
def mock_editor():
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    config_manager = MagicMock()
    config_manager.config = MagicMock()
    config_manager.config.machine = MagicMock()
    config_manager.config.machine.dimensions = (200.0, 200.0)
    doc = Doc()
    return DocEditor(task_manager, config_manager, doc)


@pytest.fixture
def stock_cmd(mock_editor):
    """Provides a StockCmd instance."""
    return StockCmd(mock_editor)


@pytest.fixture
def sample_stock_item(mock_editor):
    """Provides a sample StockItem instance."""
    geometry = Geometry()
    geometry.move_to(0, 0)
    geometry.line_to(10, 0)
    geometry.line_to(10, 10)
    geometry.line_to(0, 10)
    geometry.close_path()
    return StockItem(geometry=geometry, name="Test Stock")


def test_rename_stock_item(stock_cmd, sample_stock_item):
    """Test renaming a stock item."""
    new_name = "New Test Stock"
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.rename_stock_item(sample_stock_item, new_name)

    assert sample_stock_item.name == new_name
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_rename_stock_item_no_change(stock_cmd, sample_stock_item):
    """Test that renaming to the same name does nothing."""
    initial_name = sample_stock_item.name
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.rename_stock_item(sample_stock_item, initial_name)

    assert sample_stock_item.name == initial_name
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len


def test_add_stock_item(stock_cmd):
    """Test adding a new stock item."""
    initial_count = len(stock_cmd._editor.doc.stock_items)
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.add_stock_item()

    assert len(stock_cmd._editor.doc.stock_items) == initial_count + 1
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_delete_stock_item(stock_cmd, sample_stock_item):
    """Test deleting a stock item."""
    stock_cmd._editor.doc.add_child(sample_stock_item)
    initial_count = len(stock_cmd._editor.doc.stock_items)
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.delete_stock_item(sample_stock_item)

    assert len(stock_cmd._editor.doc.stock_items) == initial_count - 1
    assert sample_stock_item not in stock_cmd._editor.doc.stock_items
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_toggle_stock_visibility(stock_cmd, sample_stock_item):
    """Test toggling stock item visibility."""
    initial_visibility = sample_stock_item.visible
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.toggle_stock_visibility(sample_stock_item)

    assert sample_stock_item.visible is not initial_visibility
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_reorder_stock_items(stock_cmd):
    """Test reordering stock items."""
    stock1 = StockItem(name="Stock 1")
    stock2 = StockItem(name="Stock 2")
    stock_cmd._editor.doc.add_child(stock1)
    stock_cmd._editor.doc.add_child(stock2)
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    new_order = [stock2, stock1]
    stock_cmd.reorder_stock_items(new_order)

    assert stock_cmd._editor.doc.stock_items == new_order
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_set_stock_thickness(stock_cmd, sample_stock_item):
    """Test setting the stock item thickness."""
    new_thickness = 15.0
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.set_stock_thickness(sample_stock_item, new_thickness)

    assert sample_stock_item.thickness == new_thickness
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_set_stock_thickness_no_change(stock_cmd, sample_stock_item):
    """Test that setting the same thickness does nothing."""
    initial_thickness = sample_stock_item.thickness
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.set_stock_thickness(sample_stock_item, initial_thickness)

    assert sample_stock_item.thickness == initial_thickness
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len
