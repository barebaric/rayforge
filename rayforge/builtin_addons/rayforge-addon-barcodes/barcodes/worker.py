"""
Backend entry point for the barcodes addon.

Registers the barcode asset type and its renderer in every process that
loads documents or renders workpieces.
"""

from rayforge.core.hooks import hookimpl

from .asset import BarcodeAsset
from .renderer import BARCODE_RENDERER

ADDON_NAME = "barcodes"


@hookimpl
def register_asset_types(asset_type_registry):
    """Register the barcode asset type."""
    asset_type_registry.register(
        BarcodeAsset, BarcodeAsset.asset_type_name, ADDON_NAME
    )


@hookimpl
def register_renderers(renderer_registry):
    """Register the barcode renderer."""
    renderer_registry.register(BARCODE_RENDERER, ADDON_NAME)
