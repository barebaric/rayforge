import pytest
from unittest.mock import MagicMock

from rayforge.core.doc import Doc
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.sketcher.sketch import Sketch
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.editor import DocEditor
from rayforge.doceditor.asset_cmd import AssetCmd
from rayforge.shared.tasker.manager import TaskManager


@pytest.fixture
def mock_editor(context_initializer):
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    doc = Doc()
    return DocEditor(task_manager, context_initializer, doc)


@pytest.fixture
def asset_cmd(mock_editor):
    """Provides an AssetCmd instance."""
    return AssetCmd(mock_editor)


def test_rename_stock_asset(asset_cmd: AssetCmd):
    """Test renaming a StockAsset also renames its dependent StockItem."""
    doc = asset_cmd.doc
    asset = StockAsset(name="Old Stock Name")
    item = StockItem(stock_asset_uid=asset.uid, name="Old Stock Name")
    doc.add_asset(asset)
    doc.add_child(item)

    new_name = "New Stock Name"
    asset_cmd.rename_asset(asset, new_name)

    assert asset.name == new_name
    assert item.name == new_name
    assert len(doc.history_manager.undo_stack) == 1

    # Test undo
    doc.history_manager.undo()
    assert asset.name == "Old Stock Name"
    assert item.name == "Old Stock Name"


def test_rename_sketch_asset(asset_cmd: AssetCmd):
    """Test renaming a Sketch asset also renames its dependent WorkPiece."""
    doc = asset_cmd.doc
    sketch = Sketch(name="Old Sketch Name")
    workpiece = WorkPiece.from_sketch(sketch)
    workpiece.name = "Old Sketch Name"  # Match the asset name
    doc.add_asset(sketch)
    doc.add_workpiece(workpiece)

    new_name = "New Sketch Name"
    asset_cmd.rename_asset(sketch, new_name)

    assert sketch.name == new_name
    assert workpiece.name == new_name
    assert len(doc.history_manager.undo_stack) == 1

    # Test undo
    doc.history_manager.undo()
    assert sketch.name == "Old Sketch Name"
    assert workpiece.name == "Old Sketch Name"


def test_delete_stock_asset_and_item(asset_cmd: AssetCmd):
    """Test deleting a StockAsset also removes its dependent StockItem."""
    doc = asset_cmd.doc
    asset = StockAsset(name="Stock To Delete")
    item = StockItem(stock_asset_uid=asset.uid, name="Stock To Delete")
    doc.add_asset(asset)
    doc.add_child(item)

    assert len(doc.stock_assets) == 1
    assert len(doc.stock_items) == 1

    asset_cmd.delete_asset(asset)

    assert len(doc.stock_assets) == 0
    assert len(doc.stock_items) == 0
    assert len(doc.history_manager.undo_stack) == 1

    # Test undo
    doc.history_manager.undo()
    assert len(doc.stock_assets) == 1
    assert len(doc.stock_items) == 1
    restored_asset = next(iter(doc.stock_assets.values()))
    restored_item = doc.stock_items[0]
    assert restored_asset.uid == asset.uid
    assert restored_item.uid == item.uid


def test_delete_sketch_and_workpiece(asset_cmd: AssetCmd):
    """Test deleting a Sketch also removes its dependent WorkPiece."""
    doc = asset_cmd.doc
    sketch = Sketch(name="Sketch To Delete")
    workpiece = WorkPiece.from_sketch(sketch)
    doc.add_asset(sketch)
    doc.add_workpiece(workpiece)

    assert len(doc.sketches) == 1
    assert len(doc.all_workpieces) == 1

    asset_cmd.delete_asset(sketch)

    assert len(doc.sketches) == 0
    assert len(doc.all_workpieces) == 0
    assert len(doc.history_manager.undo_stack) == 1

    # Test undo
    doc.history_manager.undo()
    assert len(doc.sketches) == 1
    assert len(doc.all_workpieces) == 1
    restored_sketch = next(iter(doc.sketches.values()))
    restored_wp = doc.all_workpieces[0]
    assert restored_sketch.uid == sketch.uid
    assert restored_wp.uid == workpiece.uid
