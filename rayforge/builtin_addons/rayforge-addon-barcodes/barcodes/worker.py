"""
Backend entry point for the barcodes addon.

Registers the barcode asset type, which worker processes need to
rebuild barcode workpieces.
"""

from rayforge.core.hooks import hookimpl

from .asset import BarcodeAsset

ADDON_NAME = "barcodes"


@hookimpl
def register_asset_types(asset_type_registry):
    """Register the barcode asset type."""
    asset_type_registry.register(
        BarcodeAsset, BarcodeAsset.asset_type_name, ADDON_NAME
    )
