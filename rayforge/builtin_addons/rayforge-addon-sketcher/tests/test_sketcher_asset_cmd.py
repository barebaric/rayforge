from typing import cast

import pytest
from sketcher.core import Sketch
from sketcher.core.commands import TextBoxCommand
from sketcher.core.entities import TextBoxEntity

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.asset_cmd import AssetCmd, UpdateAssetCommand


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


def test_update_asset_undo_restores_workpiece_size(doc):
    """Undoing a sketcher session must revert the workpiece size.

    The sketcher mutates the live asset object in place, so the undo
    target must be the state captured when the sketcher was entered -
    not the state captured when the update command is created."""
    sketch = Sketch(name="s")
    box_cmd = TextBoxCommand(sketch, origin=(0, 0), width=30.0)
    box_cmd.execute()
    assert box_cmd.text_box_id is not None
    text_box = cast(
        TextBoxEntity, sketch.registry.get_entity(box_cmd.text_box_id)
    )
    text_box.content = "ab"
    sketch.solve()
    doc.add_asset(sketch)

    workpiece = WorkPiece.from_geometry_provider(sketch)
    doc.add_workpiece(workpiece)
    initial_size = workpiece.size

    # Enter the sketcher: snapshot the pre-edit state.
    entry_data = sketch.to_dict()

    # The session mutates the live asset object...
    text_box.content = "abcdef longer"
    sketch.solve()

    # ...and finishing builds the command against the mutated object.
    cmd = UpdateAssetCommand(
        doc=doc,
        asset_uid=sketch.uid,
        new_data=sketch.to_dict(),
        old_data=entry_data,
    )
    doc.history_manager.execute(cmd)

    grown_size = workpiece.size
    assert grown_size[0] > initial_size[0]

    doc.history_manager.undo()

    assert workpiece.size == pytest.approx(initial_size)
    assert workpiece.natural_width_mm == pytest.approx(initial_size[0])
