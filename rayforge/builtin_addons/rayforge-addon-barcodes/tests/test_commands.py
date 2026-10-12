from types import SimpleNamespace

import pytest
from barcodes.asset import BarcodeAsset
from barcodes.commands import (
    DEFAULT_CONTENT,
    add_barcode,
    barcode_of,
    set_barcode_setting,
)
from barcodes.encoders import Symbology

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece


@pytest.fixture
def editor():
    doc = Doc()
    return SimpleNamespace(
        doc=doc,
        history_manager=doc.history_manager,
        context=SimpleNamespace(machine=None),
    )


def test_add_barcode_creates_asset_and_workpiece(editor):
    wp = add_barcode(editor, (40.0, 30.0))
    asset = barcode_of(wp)
    assert isinstance(asset, BarcodeAsset)
    assert asset.content == DEFAULT_CONTENT
    assert editor.doc.get_asset_by_uid(asset.uid) is asset
    assert wp in editor.doc.all_workpieces
    x, y = wp.pos
    w, h = wp.size
    assert (x + w / 2, y + h / 2) == pytest.approx((40.0, 30.0))


def test_add_barcode_is_one_undo_step(editor):
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    editor.history_manager.undo()
    assert editor.doc.all_workpieces == []
    assert editor.doc.get_asset_by_uid(asset.uid) is None


def test_setting_resizes_the_workpiece_and_undoes(editor):
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    width = wp.size[0]
    set_barcode_setting(editor, asset, "module_mm", asset.module_mm * 2)
    assert wp.size[0] == pytest.approx(width * 2)
    editor.history_manager.undo()
    assert wp.size[0] == pytest.approx(width)


def test_switching_symbology(editor):
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    set_barcode_setting(editor, asset, "content", "590123412345")
    set_barcode_setting(editor, asset, "symbology", Symbology.EAN13)
    assert asset.error is None
    width, height = wp.size
    assert width == pytest.approx(95 * asset.module_mm)
    assert height == pytest.approx(asset.bar_height_mm)


def test_unchanged_setting_adds_no_undo_step(editor):
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    set_barcode_setting(editor, asset, "content", asset.content)
    editor.history_manager.undo()
    assert editor.doc.all_workpieces == []


def test_barcode_of_other_items():
    assert barcode_of(WorkPiece("plain")) is None
    assert barcode_of(None) is None
