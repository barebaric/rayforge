"""
Undoable barcode edits: adding a barcode and changing its settings.
"""

from gettext import gettext as _
from typing import Any, Protocol

from rayforge.core.doc import Doc
from rayforge.core.item import DocItem
from rayforge.core.undo import (
    ChangePropertyCommand,
    HistoryManager,
    ListItemCommand,
)
from rayforge.core.workpiece import WorkPiece

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
    """The centre of the machine's work area, or a fixed fallback."""
    machine = getattr(getattr(editor, "context", None), "machine", None)
    if machine is None:
        return DEFAULT_POSITION_MM
    x, y, w, h = machine.work_area
    return (x + w / 2.0, y + h / 2.0)


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


def _resize_instances(doc: Doc, asset: BarcodeAsset) -> None:
    """Gives every workpiece of *asset* the symbol's true size."""
    for workpiece in doc.all_workpieces:
        if workpiece.geometry_provider_uid == asset.uid:
            workpiece.regenerate_from_geometry_provider()


def set_barcode_setting(
    editor: _Editor, asset: BarcodeAsset, setting: str, value: Any
) -> None:
    """Changes one barcode setting as an undo step."""
    if getattr(asset, setting) == value:
        return
    doc = editor.doc
    editor.history_manager.execute(
        ChangePropertyCommand(
            target=asset,
            property_name=setting,
            new_value=value,
            on_change_callback=lambda: _resize_instances(doc, asset),
            name=_("Change Barcode"),
        )
    )
