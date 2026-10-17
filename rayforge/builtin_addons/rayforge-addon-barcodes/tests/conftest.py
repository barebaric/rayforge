"""
Pytest configuration for the barcodes builtin addon tests.
"""

import pytest
from barcodes.asset import BarcodeAsset

from rayforge.core.asset_registry import asset_type_registry

ADDON_NAME = "barcodes"


@pytest.fixture(scope="session", autouse=True)
def register_barcode_asset():
    """Registers the barcode asset type for all tests."""
    asset_type_registry.register(
        BarcodeAsset, BarcodeAsset.asset_type_name, ADDON_NAME
    )
    yield
    asset_type_registry.unregister_all_from_addon(ADDON_NAME)
