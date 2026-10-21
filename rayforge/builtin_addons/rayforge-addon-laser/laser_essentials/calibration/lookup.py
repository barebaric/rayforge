"""Finds the calibration test that selected workpieces belong to."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

from rayforge.core.item import DocItem
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece

from .material_test_grid import (
    NO_PRESET,
    MaterialTestGridGenerator,
    params_from_step,
)
from .record import CalibrationRecord, CalibrationState

if TYPE_CHECKING:
    from rayforge.core.asset import IAsset
    from rayforge.core.doc import Doc


def record_for_items(items: list[DocItem]) -> CalibrationRecord | None:
    """
    The record of the test all *items* belong to, or None.

    A Material Test Grid made by an older version has no record. For it
    a record is made from its step; it is not in the document yet and
    gets added there on the first change of a parameter.
    """
    if not items or not all(isinstance(i, WorkPiece) for i in items):
        return None
    first = items[0]
    doc = first.doc
    if doc is None:
        return None
    record = find_record(doc, first.uid) or _legacy_record(first)
    if record is None:
        return None
    if all(record.owns(item.uid) for item in items):
        return record
    return None


def find_record(doc: "Doc", workpiece_uid: str) -> CalibrationRecord | None:
    """
    The record that owns *workpiece_uid*. Records are matched by asset
    type name, so records loaded before the addon was reloaded match.
    """
    for asset in doc.get_all_assets():
        if asset.asset_type_name != CalibrationRecord.asset_type_name:
            continue
        record = cast(CalibrationRecord, asset)
        if record.owns(workpiece_uid):
            return record
    return None


def _legacy_record(wp: DocItem) -> CalibrationRecord | None:
    assert isinstance(wp, WorkPiece)
    layer = wp.layer
    if layer is None or layer.workflow is None:
        return None
    for step in layer.workflow.steps:
        if (
            step.ASSEMBLER_NAME == "material_test_grid"
            and step.workpiece_uids == [wp.uid]
        ):
            return _record_from_step(step, wp, layer.uid)
    return None


def _record_from_step(
    step: Step, wp: WorkPiece, layer_uid: str
) -> CalibrationRecord:
    params = params_from_step(step)
    params["preset"] = NO_PRESET
    source = wp.source
    state = CalibrationState(
        params=params,
        workpiece_uids=(wp.uid,),
        step_uids=(step.uid,),
        asset_uids=(source.uid,) if source is not None else (),
    )
    return CalibrationRecord(
        generator_id=MaterialTestGridGenerator.id,
        layer_uid=layer_uid,
        state=state,
        name=step.name,
    )


def unused_assets(
    doc: "Doc", record: CalibrationRecord, removed: Iterable[str] = ()
) -> list["IAsset"]:
    """
    The record's assets that no workpiece in the document uses, leaving
    out the workpieces with uids in *removed*. A cell the user moved to
    another layer keeps its asset.
    """
    gone = set(removed)
    in_use = {
        wp.source_segment.source_asset_uid
        for wp in doc.all_workpieces
        if wp.uid not in gone and wp.source_segment is not None
    }
    found = (
        doc.get_asset_by_uid(uid)
        for uid in record.state.asset_uids
        if uid not in in_use
    )
    return [asset for asset in found if asset is not None]
