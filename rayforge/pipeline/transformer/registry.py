from typing import TYPE_CHECKING

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
        self._transformers: dict[str, type[OpsTransformer]] = {}
        self._addon_items: dict[str, set[str]] = {}

    def register(
        self,
        transformer_class: type["OpsTransformer"],
        name: str | None = None,
        addon_name: str | None = None,
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

    def get(self, name: str) -> type["OpsTransformer"] | None:
        """
        Look up a transformer class by name.

        Args:
            name: The class name of the transformer.

        Returns:
            The transformer class, or None if not found.
        """
        return self._transformers.get(name)

    def progress_label(self, spec_name: str) -> str | None:
        """
        Look up the UI label for a raygeo transformer spec ``name()``.

        The label comes from the registered transformer whose
        ``SPEC_NAME`` matches, via its :attr:`label` property.

        Args:
            spec_name: A raygeo transformer spec ``name()`` (e.g.
                ``"overscan"``).

        Returns:
            The UI label, or None when no registered transformer
            declares that spec name.
        """
        for transformer_class in self._transformers.values():
            if transformer_class.SPEC_NAME == spec_name:
                return transformer_class().label
        return None


transformer_registry = TransformerRegistry()
