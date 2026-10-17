"""
Undoable barcode edits: adding a barcode and changing its settings.
"""

from gettext import gettext as _
from typing import Any, Protocol

from rayforge.core.doc import Doc
from rayforge.core.item import DocItem
from rayforge.core.undo import HistoryManager, ListItemCommand
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.asset_cmd import UpdateAssetCommand

from .asset import BarcodeAsset

DEFAULT_CONTENT = "Rayforge"
DEFAULT_POSITION_MM = (50.0, 50.0)


class _Editor(Protocol):
    doc: Doc
    history_manager: HistoryManager


def barcode_of(item: DocItem | None) -> BarcodeAsset | None:
    """The barcode asset behind a workpiece, if it is a barcode."""
    if not isinstance(item, WorkPiece):
        return None
    provider = item.get_geometry_provider()
    return provider if isinstance(provider, BarcodeAsset) else None


def default_position(editor: Any) -> tuple[float, float]:
    """The centre of the machine's work area in world coordinates."""
    machine = editor.context.machine
    if machine is None:
        return DEFAULT_POSITION_MM
    return machine.panel.work_area_center()


def add_barcode(
    editor: _Editor, position_mm: tuple[float, float]
) -> WorkPiece:
    """Adds a QR code centred on *position_mm*, as one undo step."""
    doc = editor.doc
    asset = BarcodeAsset(content=DEFAULT_CONTENT)
    with editor.history_manager.transaction(_("Add Barcode")) as t:
        t.execute(
            ListItemCommand(
                owner_obj=doc,
                item=asset,
                undo_command="remove_asset",
                redo_command="add_asset",
            )
        )
        workpiece = WorkPiece.from_geometry_provider(asset)
        width, height = workpiece.natural_size
        workpiece.pos = (
            position_mm[0] - width / 2.0,
            position_mm[1] - height / 2.0,
        )
        t.execute(
            ListItemCommand(
                owner_obj=doc.active_layer,
                item=workpiece,
                undo_command="remove_child",
                redo_command="add_child",
            )
        )
    return workpiece


def set_barcode_setting(
    editor: _Editor, asset: BarcodeAsset, setting: str, value: Any
) -> None:
    """
    Changes one barcode setting as an undo step. The asset is replaced
    and its workpieces are resized by the generic asset update.
    """
    if getattr(asset, setting) == value:
        return
    changed = BarcodeAsset.from_dict(asset.to_dict())
    setattr(changed, setting, value)
    editor.history_manager.execute(
        UpdateAssetCommand(
            doc=editor.doc,
            asset_uid=asset.uid,
            new_data=changed.to_dict(),
            name=_("Change Barcode"),
        )
    )
