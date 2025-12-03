import logging
from typing import Optional, TYPE_CHECKING, cast
from gi.repository import Gtk
from ...core.asset import IAsset
from ...core.stock_asset import StockAsset
from ...core.sketcher.sketch import Sketch
from .asset_row_widget import StockAssetRowWidget, SketchAssetRowWidget

if TYPE_CHECKING:
    from ..editor import DocEditor

logger = logging.getLogger(__name__)


def create_asset_row_widget(
    asset: IAsset, editor: "DocEditor"
) -> Optional[Gtk.Widget]:
    """
    Factory function to create the appropriate GTK widget for an asset row.
    """
    doc = editor.doc

    if asset.asset_type_name == "stock":
        # Cast the asset to the specific type for the constructor
        stock_item = cast(StockAsset, asset)
        return StockAssetRowWidget(doc, stock_item, editor)

    elif asset.asset_type_name == "sketch":
        # Cast the asset to the specific type for the constructor
        sketch = cast(Sketch, asset)
        return SketchAssetRowWidget(doc, sketch, editor)

    else:
        logger.warning(
            f"No asset row widget registered for type: {asset.asset_type_name}"
        )
        return None
