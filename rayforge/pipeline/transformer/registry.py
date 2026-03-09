from typing import Dict, Optional, Set, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import OpsTransformer


class TransformerRegistry:
    """
    Registry for OpsTransformer classes.

    Allows explicit registration of transformer types for lookup by name.
    This replaces the introspection-based approach with a cleaner
    explicit registration pattern.
    """

    def __init__(self):
        self._transformers: Dict[str, Type["OpsTransformer"]] = {}
        self._addon_items: Dict[str, Set[str]] = {}

    def register(
        self,
        transformer_class: Type["OpsTransformer"],
        name: Optional[str] = None,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register a transformer class.

        Args:
            transformer_class: The OpsTransformer subclass to register.
            name: Optional name to use as the registry key.
                  If not provided, the class name is used.
            addon_name: Optional name of the addon registering this
                        transformer. Used for cleanup when addon is
                        unloaded.
        """
        key = name if name else transformer_class.__name__
        self._transformers[key] = transformer_class
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(key)

    def unregister(self, name: str) -> bool:
        """
        Unregister a transformer class by name.

        Args:
            name: The name of the transformer to unregister.

        Returns:
            True if the transformer was unregistered, False if not found.
        """
        if name in self._transformers:
            del self._transformers[name]
            for addon_name, items in self._addon_items.items():
                items.discard(name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all transformers registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of transformers unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._transformers:
                del self._transformers[name]
                count += 1
        return count

    def get(self, name: str) -> Optional[Type["OpsTransformer"]]:
        """
        Look up a transformer class by name.

        Args:
            name: The class name of the transformer.

        Returns:
            The transformer class, or None if not found.
        """
        from rayforge.worker_init import ensure_addons_loaded

        ensure_addons_loaded()

        return self._transformers.get(name)

    def all_transformers(self) -> Dict[str, Type["OpsTransformer"]]:
        """
        Return a copy of all registered transformers.

        Returns:
            Dictionary mapping transformer names to classes.
        """
        return self._transformers.copy()


transformer_registry = TransformerRegistry()
