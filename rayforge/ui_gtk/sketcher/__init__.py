from ...core.sketcher import Sketch
from ..doceditor.asset_row_factory import asset_row_widget_registry
from ..doceditor.property_providers import property_provider_registry
from .asset_row_widget import SketchAssetRowWidget
from .property_provider import SketchPropertyProvider
from .sketchelement import SketchElement


def register():
    """Register sketch module components with the application.

    This function is called during application initialization to
    register sketch-specific components with their respective registries.
    """
    property_provider_registry.register(SketchPropertyProvider)
    asset_row_widget_registry.register(Sketch, SketchAssetRowWidget)


# Auto-register when module is imported
register()

__all__ = ["SketchAssetRowWidget", "SketchElement", "SketchPropertyProvider"]
