from types import SimpleNamespace
from typing import Any

import pytest
from barcodes.asset import BarcodeAsset
from barcodes.commands import (
    DEFAULT_CONTENT,
    DEFAULT_POSITION_MM,
    add_barcode,
    barcode_of,
    default_position,
    new_barcode_asset,
    rename_barcode_asset,
    set_barcode_setting,
)
from raygeo.barcode import Symbology
from raygeo.template import CounterSettings

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.asset_cmd import AssetCmd


def _editor() -> Any:
    doc = Doc()
    editor: Any = SimpleNamespace(
        doc=doc,
        history_manager=doc.history_manager,
        context=SimpleNamespace(machine=None),
    )
    editor.asset = AssetCmd(editor)
    return editor


def test_add_barcode_creates_asset_and_workpiece():
    editor = _editor()
    wp = add_barcode(editor, (40.0, 30.0))
    asset = barcode_of(wp)
    assert isinstance(asset, BarcodeAsset)
    assert asset.content == DEFAULT_CONTENT
    assert editor.doc.get_asset_by_uid(asset.uid) is asset
    assert wp in editor.doc.all_workpieces
    x, y = wp.pos
    w, h = wp.size
    assert (x + w / 2, y + h / 2) == pytest.approx((40.0, 30.0))


def test_add_barcode_is_one_undo_step():
    editor = _editor()
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    editor.history_manager.undo()
    assert editor.doc.all_workpieces == []
    assert editor.doc.get_asset_by_uid(asset.uid) is None


def test_new_barcode_asset_adds_no_workpiece():
    editor = _editor()
    asset = new_barcode_asset(editor)
    assert editor.doc.get_asset_by_uid(asset.uid) is asset
    assert editor.doc.all_workpieces == []
    editor.history_manager.undo()
    assert editor.doc.get_asset_by_uid(asset.uid) is None


def test_setting_resizes_the_workpiece_and_undoes():
    editor = _editor()
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    width = wp.size[0]
    set_barcode_setting(editor, asset, "module_mm", asset.module_mm * 2)
    assert wp.size[0] == pytest.approx(width * 2)
    editor.history_manager.undo()
    assert wp.size[0] == pytest.approx(width)


def test_switching_symbology():
    editor = _editor()
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    set_barcode_setting(editor, asset, "content", "590123412345")
    asset = barcode_of(wp)
    assert asset is not None
    set_barcode_setting(editor, asset, "symbology", Symbology.EAN13)
    asset = barcode_of(wp)
    assert asset is not None
    assert asset.error is None
    width, height = wp.size
    assert width == pytest.approx(95 * asset.module_mm)
    assert height == pytest.approx(asset.bar_height_mm)


def test_unchanged_setting_adds_no_undo_step():
    editor = _editor()
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    set_barcode_setting(editor, asset, "content", asset.content)
    editor.history_manager.undo()
    assert editor.doc.all_workpieces == []


def test_setting_keeps_the_resolved_counter_value():
    editor = _editor()
    editor.doc.job_counter.counter = CounterSettings(current=99)
    asset = BarcodeAsset(content="{counter}", symbology=Symbology.CODE128)
    editor.doc.add_asset(asset)
    wp = WorkPiece.from_geometry_provider(asset)
    editor.doc.active_setup.add_child(wp)
    assert asset.error is None

    width = wp.size[0]
    set_barcode_setting(editor, asset, "module_mm", 1.0)
    asset = barcode_of(wp)
    assert asset is not None
    assert wp.size[0] == pytest.approx(width * 2)
    assert asset.error is None


def test_rename_barcode_asset():
    editor = _editor()
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    rename_barcode_asset(editor, asset, "Serial label")
    assert asset.name == "Serial label"
    assert wp.name == "Serial label"


def test_barcode_of_other_items():
    assert barcode_of(WorkPiece("plain")) is None
    assert barcode_of(None) is None


def test_default_position_without_machine():
    assert default_position(_editor()) == DEFAULT_POSITION_MM


def test_default_position_is_the_panel_work_area_center():
    editor = _editor()
    panel = SimpleNamespace(work_area_center=lambda: (120.0, 80.0))
    editor.context.machine = SimpleNamespace(panel=panel)
    assert default_position(editor) == (120.0, 80.0)
