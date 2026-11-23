"""Pytest configuration for the barcodes builtin addon tests."""

import pytest
from barcodes.asset import BarcodeAsset
from barcodes.renderer import BARCODE_RENDERER

from rayforge.core.asset_registry import asset_type_registry
from rayforge.image import renderer_registry

ADDON_NAME = "barcodes"


@pytest.fixture(scope="session", autouse=True)
def register_barcode_components():
    """Registers the barcode asset and renderer for all tests."""
    asset_type_registry.register(
        BarcodeAsset, BarcodeAsset.asset_type_name, ADDON_NAME
    )
    renderer_registry.register(BARCODE_RENDERER, ADDON_NAME)
    yield
    asset_type_registry.unregister_all_from_addon(ADDON_NAME)
    renderer_registry.unregister_all_from_addon(ADDON_NAME)
