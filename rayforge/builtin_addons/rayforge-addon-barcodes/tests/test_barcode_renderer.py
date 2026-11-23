from barcodes.asset import BarcodeAsset
from barcodes.renderer import BARCODE_RENDERER
from raygeo.barcode import Symbology


def test_renderer_draws_filled_geometry():
    asset = BarcodeAsset(content="Rayforge")
    geometry, fills = asset.get_geometry()
    image = BARCODE_RENDERER.render_base_image(
        b"", 64, 64, boundaries=geometry, fills=fills
    )
    assert image is not None
    assert image.width == 64
    assert image.height == 64


def test_renderer_draws_placeholder_stroke():
    asset = BarcodeAsset(content="12AB", symbology=Symbology.EAN13)
    geometry, fills = asset.get_geometry()
    assert fills == []
    image = BARCODE_RENDERER.render_base_image(
        b"", 64, 64, boundaries=geometry, fills=fills
    )
    assert image is not None
