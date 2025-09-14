import pytest
from unittest.mock import MagicMock
from blinker import Signal
from rayforge.core.stocklayer import StockLayer
from rayforge.core.stock import StockItem
from rayforge.core.workpiece import WorkPiece
from rayforge.core.step import Step


@pytest.fixture
def stock_layer():
    return StockLayer()


@pytest.fixture
def mock_stock_item():
    """Provides a MagicMock of a StockItem with real Signal objects."""
    mock_item = MagicMock(spec=StockItem)
    # DocItem base class expects these signals to exist for connection
    mock_item.updated = Signal()
    mock_item.transform_changed = Signal()
    mock_item.descendant_added = Signal()
    mock_item.descendant_removed = Signal()
    mock_item.descendant_updated = Signal()
    mock_item.descendant_transform_changed = Signal()
    mock_item.parent = None
    return mock_item


def test_stocklayer_init_no_workflow(stock_layer):
    """Verify that a new StockLayer has no workflow."""
    assert stock_layer.workflow is None
    assert stock_layer.children == []


def test_stocklayer_get_renderable_items_is_empty(stock_layer):
    """Verify get_renderable_items is always empty for a StockLayer."""
    assert stock_layer.get_renderable_items() == []


def test_stocklayer_add_stockitem_succeeds(stock_layer, mock_stock_item):
    """Test that adding a StockItem to a StockLayer is allowed."""
    stock_layer.add_child(mock_stock_item)
    assert mock_stock_item in stock_layer.children
    assert mock_stock_item.parent is stock_layer


def test_stocklayer_add_workpiece_fails(stock_layer):
    """Test that adding a WorkPiece to a StockLayer raises a TypeError."""
    mock_workpiece = MagicMock(spec=WorkPiece)
    with pytest.raises(TypeError):
        stock_layer.add_child(mock_workpiece)


def test_stocklayer_add_step_fails(stock_layer):
    """Test that adding a Step to a StockLayer raises a TypeError."""
    mock_step = MagicMock(spec=Step)
    with pytest.raises(TypeError):
        stock_layer.add_child(mock_step)
