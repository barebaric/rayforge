import logging
from typing import Dict, Optional, Type, TYPE_CHECKING

from gi.repository import Gtk

from ...core.asset import IAsset
from .asset_row_widget import BaseAssetRowWidget, StockAssetRowWidget

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class AssetRowWidgetRegistry:
    """
    Registry for asset row widget classes.

    Allows registration of widget classes for specific asset types.
    Each widget class is a subclass of BaseAssetRowWidget.
    """

    def __init__(self):
        self._widgets: Dict[str, Type[BaseAssetRowWidget]] = {}

    def register(
        self, asset_cls: Type[IAsset], widget_cls: Type[BaseAssetRowWidget]
    ):
        """
        Register a widget class for an asset type.

        Args:
            asset_cls: The asset class (e.g., Sketch, StockAsset)
            widget_cls: The widget class to create for this asset type
        """
        type_name = asset_cls.asset_type_name
        self._widgets[type_name] = widget_cls
        logger.debug(
            f"Registered asset row widget for '{type_name}': "
            f"{widget_cls.__name__}"
        )

    def unregister(self, asset_cls: Type[IAsset]) -> bool:
        """
        Unregister a widget class for an asset type.

        Returns True if the widget was found and removed.
        """
        type_name = asset_cls.asset_type_name
        if type_name in self._widgets:
            del self._widgets[type_name]
            return True
        return False

    def create(
        self, asset: IAsset, doc: "Doc", editor: "DocEditor"
    ) -> Optional[BaseAssetRowWidget]:
        """
        Create a widget instance for the given asset.

        Args:
            asset: The asset instance
            doc: The document
            editor: The document editor

        Returns:
            A widget instance, or None if no widget is registered
        """
        widget_cls = self._widgets.get(asset.asset_type_name)
        if widget_cls:
            return widget_cls(doc, asset, editor)
        return None


asset_row_widget_registry = AssetRowWidgetRegistry()


def register_builtin_widgets():
    """Register built-in asset row widgets.

    Note: SketchAssetRowWidget is registered separately by the
    sketcher module to keep sketch-related code together.
    """
    from ...core.stock_asset import StockAsset

    asset_row_widget_registry.register(StockAsset, StockAssetRowWidget)


def create_asset_row_widget(
    asset: IAsset, editor: "DocEditor"
) -> Optional[Gtk.Widget]:
    """
    Factory function to create the appropriate GTK widget for an asset row.
    """
    widget = asset_row_widget_registry.create(asset, editor.doc, editor)
    if widget is None:
        logger.warning(
            f"No asset row widget registered for type: {asset.asset_type_name}"
        )
    return widget
