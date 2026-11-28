from typing import cast

import pytest
from barcodes.asset import BarcodeAsset
from raygeo.barcode import QrErrorCorrection, Symbology
from raygeo.template import CounterSettings

from rayforge.core.asset import IAsset
from rayforge.core.doc import Doc
from rayforge.core.geometry_provider import IGeometryProvider
from rayforge.core.workpiece import WorkPiece
from rayforge.image.structures import FillStyle


def _size(asset: BarcodeAsset, text_values=None):
    geometry, _fills = asset.get_geometry(text_values=text_values)
    x0, y0, x1, y1 = geometry.rect()
    return x1 - x0, y1 - y0


def test_is_an_asset_and_a_geometry_provider():
    asset = BarcodeAsset(content="x")
    assert isinstance(asset, IAsset)
    assert isinstance(asset, IGeometryProvider)
    assert asset.provider_type_name == "barcode"
    assert asset.renderer is not None


def test_round_trip():
    asset = BarcodeAsset(
        name="Label",
        content="SN-{counter:04d}",
        symbology=Symbology.CODE128,
        error_correction=QrErrorCorrection.H,
        module_mm=0.4,
        bar_height_mm=9.0,
        quiet_zone=12,
        show_text=True,
        invert=True,
    )
    restored = BarcodeAsset.from_dict(asset.to_dict())
    assert restored.to_dict() == asset.to_dict()
    assert restored.uid == asset.uid


def test_qr_geometry_and_fill():
    asset = BarcodeAsset(content="hello", module_mm=0.5)
    geometry, fills = asset.get_geometry()
    assert not geometry.is_empty()
    assert len(fills) == 1
    assert fills[0].style is FillStyle.SOLID
    width, height = _size(asset)
    assert width == pytest.approx(height)
    assert width == pytest.approx(21 * 0.5)


def test_templates_resolve_with_text_values():
    asset = BarcodeAsset(content="{counter}", symbology=Symbology.CODE128)
    narrow, _ = _size(asset, {"counter": 99})
    wide, _ = _size(asset, {"counter": 100})
    assert asset.variables_used() == {"counter"}
    assert wide > narrow


def test_resolved_text_cache_is_stable_for_volatile_content():
    asset = BarcodeAsset(content="{uuid8()}", symbology=Symbology.CODE128)
    cache: dict = {}
    first, _ = asset.get_geometry(resolved_text_cache=cache)
    second, _ = asset.get_geometry(resolved_text_cache=cache)
    assert cache[0]
    assert first.to_polygons(0.01) == second.to_polygons(0.01)


def test_invalid_content_gives_a_placeholder_and_an_error():
    asset = BarcodeAsset(content="12AB", symbology=Symbology.EAN13)
    geometry, fills = asset.get_geometry()
    assert not geometry.is_empty()
    assert fills == []
    assert asset.error is not None
    asset.content = "590123412345"
    asset.get_geometry()
    assert asset.error is None


def test_default_quiet_zone_is_the_symbology_minimum():
    assert BarcodeAsset(symbology=Symbology.QR).effective_quiet_zone == 4
    assert BarcodeAsset(symbology=Symbology.EAN13).effective_quiet_zone == 9
    assert BarcodeAsset(quiet_zone=2).effective_quiet_zone == 2


def test_workpiece_from_barcode():
    doc = Doc()
    asset = BarcodeAsset(content="hello", module_mm=0.5)
    doc.add_asset(asset)
    wp = WorkPiece.from_geometry_provider(asset)
    doc.active_setup.add_child(wp)
    assert wp.size == pytest.approx((10.5, 10.5))
    assert cast(BarcodeAsset, wp.get_geometry_provider()) is asset
    assert wp.boundaries is not None


def test_thumbnail():
    assert BarcodeAsset(content="x").get_thumbnail(64)


def test_document_round_trip():
    doc = Doc()
    asset = BarcodeAsset(content="hello", symbology=Symbology.QR)
    doc.add_asset(asset)
    wp = WorkPiece.from_geometry_provider(asset)
    doc.active_setup.add_child(wp)
    restored = Doc.from_dict(doc.to_dict())
    (restored_wp,) = restored.all_workpieces
    provider = restored_wp.get_geometry_provider()
    assert isinstance(provider, BarcodeAsset)
    assert provider.content == "hello"
    assert restored_wp.boundaries is not None


def test_counter_offsets_and_save_load():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=99)
    asset = BarcodeAsset(content="{counter}", symbology=Symbology.CODE128)
    doc.add_asset(asset)
    first = WorkPiece.from_geometry_provider(asset)
    doc.active_setup.add_child(first)
    second = WorkPiece.from_geometry_provider(asset)
    doc.active_setup.add_child(second)

    assert first.variable_offset == 0
    assert second.variable_offset == 1
    assert second.natural_width_mm > first.natural_width_mm

    restored = Doc.from_dict(doc.to_dict())
    restored_wps = restored.all_workpieces
    assert sorted(wp.variable_offset for wp in restored_wps) == [0, 1]
    assert restored.all_workpieces[0].boundaries is not None


def test_asset_name_change_sends_update():
    asset = BarcodeAsset()
    changes = []

    def on_updated(sender):
        changes.append(sender)

    asset.updated.connect(on_updated)
    asset.name = "Renamed"
    assert changes == [asset]
