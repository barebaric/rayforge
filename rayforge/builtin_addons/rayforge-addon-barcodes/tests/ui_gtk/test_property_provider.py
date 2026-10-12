# flake8: noqa: E402
import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from types import SimpleNamespace

import pytest
from barcodes.commands import add_barcode, barcode_of
from barcodes.encoders import Symbology
from barcodes.property_provider import SYMBOLOGIES, BarcodePropertyProvider

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


@pytest.fixture
def provider():
    provider = BarcodePropertyProvider()
    provider.create_widgets()
    return provider


@pytest.mark.ui
def test_handles_only_a_single_barcode(editor, provider):
    wp = add_barcode(editor, (0.0, 0.0))
    other = add_barcode(editor, (30.0, 0.0))
    assert provider.can_handle([wp])
    assert not provider.can_handle([wp, other])
    assert not provider.can_handle([WorkPiece("plain")])


@pytest.mark.ui
def test_shows_the_settings(editor, provider):
    wp = add_barcode(editor, (0.0, 0.0))
    provider.update_widgets(editor, [wp])
    assert provider.content_row.get_text() == "Rayforge"
    assert provider.symbology_row.get_selected() == 0
    assert provider.error_row.get_visible()
    assert not provider.bar_height_row.get_visible()
    assert not provider.quiet_zone_row.get_sensitive()


@pytest.mark.ui
def test_edits_go_through_undo(editor, provider):
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    provider.update_widgets(editor, [wp])
    provider.content_row.set_text("590123412345")
    provider.content_row.emit("apply")
    provider.symbology_row.set_selected(SYMBOLOGIES.index(Symbology.EAN13))
    assert asset.symbology is Symbology.EAN13
    assert asset.content == "590123412345"
    assert provider.bar_height_row.get_visible()
    assert not provider.error_row.get_visible()
    editor.history_manager.undo()
    assert asset.symbology is Symbology.QR


@pytest.mark.ui
def test_invalid_content_is_flagged(editor, provider):
    wp = add_barcode(editor, (0.0, 0.0))
    asset = barcode_of(wp)
    assert asset is not None
    provider.update_widgets(editor, [wp])
    provider.symbology_row.set_selected(SYMBOLOGIES.index(Symbology.EAN13))
    assert asset.error is not None
    assert provider.content_row.has_css_class("error")
