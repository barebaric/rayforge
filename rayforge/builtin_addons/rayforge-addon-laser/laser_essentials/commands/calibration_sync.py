"""
Keeps state that follows from the document in step with it, without
undo entries of its own: undoing the change that caused it brings the
document back and this follows again.

- The preview of a Material Test Grid follows its step, so a recipe
  applied to the step resizes the grid.
- The record and the hidden source assets of a test leave the document
  with the test's layer and come back when the layer does.
"""

import json
import logging
from typing import TYPE_CHECKING, Any, cast

from rayforge.core.asset import IAsset
from rayforge.core.item import DocItem
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece

from ..calibration.material_test_grid import (
    MaterialTestGridGenerator,
    params_from_step,
)
from ..calibration.record import CalibrationRecord
from ..material_test_helpers import get_material_test_proportional_size

if TYPE_CHECKING:
    from rayforge.core.doc import Doc
    from rayforge.doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class CalibrationSync:
    """
    Registered with the command registry so it lives as long as the
    editor and keeps its signal connections.
    """

    def __init__(self, editor: "DocEditor"):
        self._editor = editor
        self._doc: Doc = editor.doc
        self._parked: dict[str, list[IAsset]] = {}
        self._connect()
        editor.document_changed.connect(self._on_document_changed)

    def _connect(self):
        self._doc.descendant_updated.connect(self._on_updated)
        self._doc.descendant_removed.connect(self._on_removed)
        self._doc.descendant_added.connect(self._on_added)

    def _on_document_changed(self, sender):
        self._doc.descendant_updated.disconnect(self._on_updated)
        self._doc.descendant_removed.disconnect(self._on_removed)
        self._doc.descendant_added.disconnect(self._on_added)
        self._doc = self._editor.doc
        self._parked.clear()
        self._connect()

    def _on_updated(self, sender, *, origin: DocItem, **kwargs):
        if (
            isinstance(origin, Step)
            and origin.ASSEMBLER_NAME == "material_test_grid"
        ):
            sync_grid_preview(self._doc, origin)

    def _on_removed(self, sender, *, origin: DocItem, **kwargs):
        if not isinstance(origin, Layer):
            return
        assets = [
            asset
            for record in _records(self._doc, origin.uid)
            for asset in [record, *_assets(self._doc, record)]
        ]
        if not assets:
            return
        self._parked[origin.uid] = assets
        for asset in assets:
            self._doc.remove_asset_by_uid(asset.uid, silent=True)
        self._doc.updated.send(self._doc)

    def _on_added(self, sender, *, origin: DocItem, **kwargs):
        if not isinstance(origin, Layer):
            return
        assets = self._parked.pop(origin.uid, [])
        for asset in assets:
            self._doc.add_asset(asset, silent=True)
        if assets:
            self._doc.updated.send(self._doc)


def _records(doc: "Doc", layer_uid: str) -> list[CalibrationRecord]:
    return [
        cast(CalibrationRecord, asset)
        for asset in doc.get_all_assets()
        if asset.asset_type_name == CalibrationRecord.asset_type_name
        and cast(CalibrationRecord, asset).layer_uid == layer_uid
    ]


def _assets(doc: "Doc", record: CalibrationRecord) -> list[IAsset]:
    found = (doc.get_asset_by_uid(uid) for uid in record.state.asset_uids)
    return [asset for asset in found if asset is not None]


def sync_grid_preview(doc: "Doc", step: Step):
    """
    Redraws and resizes the grid preview when its step's grid settings
    differ from the ones it was drawn with.
    """
    if not step.workpiece_uids:
        return
    wp = doc.find_descendant_by_uid(step.workpiece_uids[0])
    source = wp.source if isinstance(wp, WorkPiece) else None
    if source is None or source.original_data is None:
        return
    try:
        recipe: dict[str, Any] = json.loads(source.original_data)
        drawn = MaterialTestGridGenerator.grid_params(recipe["params"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        logger.warning("Material Test Grid preview has no grid settings.")
        return
    wanted = MaterialTestGridGenerator.grid_params(params_from_step(step))
    if wanted == drawn:
        return
    recipe["params"] = wanted
    source.original_data = json.dumps(recipe).encode("utf-8")
    assert isinstance(wp, WorkPiece)
    wp.clear_render_cache()
    wp.set_size(*get_material_test_proportional_size(wanted))
    wp.updated.send(wp)
