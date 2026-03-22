import pytest

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.asset_cmd import AssetCmd
from sketcher.core import Sketch


@pytest.fixture
def doc():
    """Provides a Doc instance."""
    return Doc()


@pytest.fixture
def asset_cmd(doc):
    """Provides an AssetCmd instance."""
    return AssetCmd(doc)


def test_rename_sketch_asset(asset_cmd: AssetCmd):
    """Test renaming a Sketch asset also renames its dependent WorkPiece."""
    doc = asset_cmd.doc
    sketch = Sketch(name="Old Sketch Name")
    workpiece = WorkPiece.from_geometry_provider(sketch)
    workpiece.name = "Old Sketch Name"
    doc.add_asset(sketch)
    doc.add_workpiece(workpiece)

    new_name = "New Sketch Name"
    asset_cmd.rename_asset(sketch, new_name)

    assert sketch.name == new_name
    assert workpiece.name == new_name
    assert len(doc.history_manager.undo_stack) == 1

    doc.history_manager.undo()
    assert sketch.name == "Old Sketch Name"
    assert workpiece.name == "Old Sketch Name"


def test_delete_sketch_and_workpiece(asset_cmd: AssetCmd):
    """Test deleting a Sketch also removes its dependent WorkPiece."""
    doc = asset_cmd.doc
    sketch = Sketch(name="Sketch To Delete")
    workpiece = WorkPiece.from_geometry_provider(sketch)
    doc.add_asset(sketch)
    doc.add_workpiece(workpiece)

    assert len(doc.get_assets_by_type("sketch")) == 1
    assert len(doc.all_workpieces) == 1

    asset_cmd.delete_asset(sketch)

    assert len(doc.get_assets_by_type("sketch")) == 0
    assert len(doc.all_workpieces) == 0
    assert len(doc.history_manager.undo_stack) == 1

    doc.history_manager.undo()
    assert len(doc.get_assets_by_type("sketch")) == 1
    assert len(doc.all_workpieces) == 1
    restored_sketch = next(iter(doc.get_assets_by_type("sketch").values()))
    restored_wp = doc.all_workpieces[0]
    assert restored_sketch.uid == sketch.uid
    assert restored_wp.uid == workpiece.uid
