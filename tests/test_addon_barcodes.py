import pytest

from rayforge.core.asset_registry import asset_type_registry
from rayforge.image import renderer_registry


@pytest.mark.asyncio
async def test_barcodes_addon_loads(context_initializer):
    assert "barcodes" in context_initializer.addon_mgr.loaded_addons
    asset_class = asset_type_registry.get("barcode")
    assert asset_class is not None
    assert asset_class.__name__ == "BarcodeAsset"
    assert asset_class.asset_type_name == "barcode"
    renderer = renderer_registry.get_by_name("BarcodeRenderer")
    assert renderer is not None
    assert type(renderer).__name__ == "BarcodeRenderer"
