from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .capability import StepCapability


class StepCapabilityRegistry:
    """
    Registry for step capability instances.

    Capabilities are registered by their ``name`` (e.g. ``"CUT"``) so
    recipes can resolve a serialized capability name back to an
    instance at runtime. Addons register their capabilities from the
    ``register_step_capabilities`` hook.
    """

    def __init__(self):
        self._capabilities: dict[str, StepCapability] = {}
        self._addon_items: dict[str, set[str]] = {}

    def register(
        self, capability: "StepCapability", addon_name: str | None = None
    ) -> None:
        """
        Register a capability instance.

        Args:
            capability: The capability instance to register. Its
                ``name`` is used as the registry key.
            addon_name: Optional name of the addon registering this
                capability. Used for cleanup when the addon is unloaded.
        """
        name = capability.name
        self._capabilities[name] = capability
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(name)

    def unregister(self, name: str) -> bool:
        """
        Unregister a capability by name.

        Args:
            name: The capability name to unregister.

        Returns:
            True if the capability was unregistered, False if not found.
        """
        if name in self._capabilities:
            del self._capabilities[name]
            for items in self._addon_items.values():
                items.discard(name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all capabilities registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of capabilities unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._capabilities:
                del self._capabilities[name]
                count += 1
        return count

    def get(self, name: str) -> Optional["StepCapability"]:
        """
        Look up a capability by name.

        Args:
            name: The serialized capability name.

        Returns:
            The capability instance, or None if not registered.
        """
        return self._capabilities.get(name)

    def all_capabilities(self) -> list["StepCapability"]:
        """
        Return all registered capabilities in registration order.

        Returns:
            A list of all capability instances.
        """
        return list(self._capabilities.values())


step_capability_registry = StepCapabilityRegistry()
