from typing import Dict, Optional, Set, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .asset import IAsset


class AssetTypeRegistry:
    """
    Registry for IAsset classes.

    Allows explicit registration of asset types for lookup by type name.
    This enables dynamic asset deserialization and supports addon-provided
    asset types.
    """

    def __init__(self):
        self._types: Dict[str, Type["IAsset"]] = {}
        self._addon_items: Dict[str, Set[str]] = {}
        self._builtins_registered: bool = False

    def _register_builtins(self) -> None:
        """
        Register built-in asset types.

        Called automatically on first access to ensure core assets
        are available even without full context initialization.
        """
        if self._builtins_registered:
            return

        from .sketcher.sketch import Sketch
        from .source_asset import SourceAsset
        from .stock_asset import StockAsset

        self._types["stock"] = StockAsset
        self._types["source"] = SourceAsset
        self._types["sketch"] = Sketch
        self._builtins_registered = True

    def register(
        self,
        asset_class: Type["IAsset"],
        type_name: str,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register an asset class.

        Args:
            asset_class: The IAsset subclass to register.
            type_name: The type name for serialization (e.g., "sketch").
            addon_name: Optional name of the addon registering this asset.
                        Used for cleanup when addon is unloaded.
        """
        self._types[type_name] = asset_class
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(type_name)

    def unregister(self, type_name: str) -> bool:
        """
        Unregister an asset class by type name.

        Args:
            type_name: The type name of the asset to unregister.

        Returns:
            True if the asset was unregistered, False if not found.
        """
        if type_name in self._types:
            del self._types[type_name]
            for addon_name, items in self._addon_items.items():
                items.discard(type_name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all assets registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of assets unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for type_name in items:
            if type_name in self._types:
                del self._types[type_name]
                count += 1
        return count

    def get(self, type_name: str) -> Optional[Type["IAsset"]]:
        """
        Look up an asset class by type name.

        Args:
            type_name: The type name of the asset.

        Returns:
            The asset class, or None if not found.
        """
        self._register_builtins()
        return self._types.get(type_name)

    def all_types(self) -> Dict[str, Type["IAsset"]]:
        """
        Return a copy of all registered asset types.

        Returns:
            Dictionary mapping type names to asset classes.
        """
        self._register_builtins()
        return self._types.copy()


asset_type_registry = AssetTypeRegistry()
