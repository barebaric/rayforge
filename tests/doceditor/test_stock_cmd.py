import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.geo import Geometry
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.stock_cmd import StockCmd
from rayforge.shared.tasker.manager import TaskManager


@pytest.fixture
def mock_editor(context_initializer):
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    doc = Doc()
    # Mock machine for add_stock, which uses machine dimensions
    mock_machine = MagicMock()
    mock_machine.dimensions = (200.0, 200.0)
    context_initializer.config.machine = mock_machine
    return DocEditor(task_manager, context_initializer, doc)


@pytest.fixture
def stock_cmd(mock_editor):
    """Provides a StockCmd instance."""
    return StockCmd(mock_editor)


@pytest.fixture
def sample_stock_item_and_asset(mock_editor):
    """
    Adds a sample StockAsset and StockItem to the doc and returns both.
    """
    doc = mock_editor.doc
    geometry = Geometry()
    geometry.move_to(0, 0)
    geometry.line_to(10, 10)

    asset = StockAsset(geometry=geometry, name="Test Stock Asset")
    item = StockItem(stock_asset_uid=asset.uid, name="Test Stock Asset")

    doc.add_asset(asset)
    doc.add_child(item)

    return item, asset


def test_add_stock(stock_cmd: StockCmd):
    """Test adding a new stock item and its asset."""
    doc = stock_cmd._editor.doc
    initial_item_count = len(doc.stock_items)
    initial_asset_count = len(doc.stock_assets)
    initial_history_len = len(doc.history_manager.undo_stack)

    stock_cmd.add_stock()

    assert len(doc.stock_items) == initial_item_count + 1
    assert len(doc.stock_assets) == initial_asset_count + 1
    hm = doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_toggle_stock_visibility(stock_cmd, sample_stock_item_and_asset):
    """Test toggling stock item visibility."""
    item, _ = sample_stock_item_and_asset
    initial_visibility = item.visible
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.toggle_stock_visibility(item)

    assert item.visible is not initial_visibility
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_set_stock_thickness(stock_cmd, sample_stock_item_and_asset):
    """Test setting the stock item thickness via its asset."""
    item, asset = sample_stock_item_and_asset
    new_thickness = 15.0
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.set_stock_thickness(item, new_thickness)

    assert item.thickness == new_thickness
    assert asset.thickness == new_thickness
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len + 1


def test_set_stock_thickness_no_change(stock_cmd, sample_stock_item_and_asset):
    """Test that setting the same thickness does nothing."""
    item, asset = sample_stock_item_and_asset
    asset.thickness = 5.0  # Ensure a starting value
    initial_thickness = item.thickness
    initial_history_len = len(stock_cmd._editor.doc.history_manager.undo_stack)

    stock_cmd.set_stock_thickness(item, initial_thickness)

    assert item.thickness == initial_thickness
    hm = stock_cmd._editor.doc.history_manager
    assert len(hm.undo_stack) == initial_history_len
