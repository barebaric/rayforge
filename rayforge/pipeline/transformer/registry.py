from typing import TYPE_CHECKING, Optional

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
        self._transformers: dict[str, type["OpsTransformer"]] = {}
        self._addon_items: dict[str, set[str]] = {}

    def register(
        self,
        transformer_class: type["OpsTransformer"],
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

    def get(self, name: str) -> Optional[type["OpsTransformer"]]:
        """
        Look up a transformer class by name.

        Args:
            name: The class name of the transformer.

        Returns:
            The transformer class, or None if not found.
        """
        return self._transformers.get(name)


transformer_registry = TransformerRegistry()
