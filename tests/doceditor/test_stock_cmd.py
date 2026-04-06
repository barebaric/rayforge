import pytest
from unittest.mock import MagicMock

from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.core.geo import Geometry
from rayforge.core.geo.constants import CMD_TYPE_ARC
from rayforge.doceditor.stock_cmd import StockCmd
from rayforge.pipeline.coordspace import (
    MachineSpace,
    OriginCorner,
    AxisDirection,
)


@pytest.fixture
def stock_cmd(doc_editor, context_initializer):
    mock_machine = MagicMock()
    mock_machine.axis_extents = (200.0, 200.0)
    mock_machine.work_area = (0.0, 0.0, 200.0, 200.0)
    mock_machine.get_reference_position_world.return_value = (0.0, 0.0)
    mock_machine.get_coordinate_space.return_value = MachineSpace(
        origin=OriginCorner.BOTTOM_LEFT,
        x_positive_direction=AxisDirection.POSITIVE_RIGHT,
        y_positive_direction=AxisDirection.POSITIVE_UP,
        extents=(200.0, 200.0),
    )
    context_initializer.config.machine = mock_machine
    return StockCmd(doc_editor)


@pytest.fixture
def sample_stock_item_and_asset(doc_editor):
    """
    Adds a sample StockAsset and StockItem to the doc and returns both.
    """
    doc = doc_editor.doc
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


def test_convert_to_stock_preserves_arcs(stock_cmd, doc_editor):
    """Test that converting a workpiece to stock preserves arcs."""
    doc = doc_editor.doc

    geometry = Geometry()
    geometry.move_to(0, 0)
    geometry.line_to(10, 0)
    geometry.arc_to(10, 10, 0, 5, clockwise=True)
    geometry.line_to(0, 10)
    geometry.arc_to(0, 0, 0, -5, clockwise=True)
    geometry.close_path()

    workpiece = WorkPiece(name="Test Workpiece")
    workpiece._edited_boundaries = geometry
    doc.add_child(workpiece)

    arc_count_before = sum(
        1 for cmd in geometry.iter_commands() if cmd[0] == CMD_TYPE_ARC
    )
    assert arc_count_before == 2, "Test setup: geometry should have 2 arcs"

    stock_item = stock_cmd.convert_to_stock(workpiece)

    assert stock_item is not None
    assert stock_item.name == "Test Workpiece"
    assert workpiece not in doc.children

    stock_geo = stock_item.geometry
    arc_count_after = sum(
        1 for cmd in stock_geo.iter_commands() if cmd[0] == CMD_TYPE_ARC
    )

    assert arc_count_after == arc_count_before, (
        f"Expected {arc_count_before} arcs but got {arc_count_after}"
    )
