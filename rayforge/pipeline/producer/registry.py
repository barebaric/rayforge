from typing import Dict, Optional, Set, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import OpsProducer


class ProducerRegistry:
    """
    Registry for OpsProducer classes.

    Allows explicit registration of producer types for lookup by name.
    This replaces the introspection-based approach with a cleaner
    explicit registration pattern.
    """

    def __init__(self):
        self._producers: Dict[str, Type["OpsProducer"]] = {}
        self._addon_items: Dict[str, Set[str]] = {}

    def register(
        self,
        producer_class: Type["OpsProducer"],
        name: Optional[str] = None,
        addon_name: Optional[str] = None,
    ) -> None:
        """
        Register a producer class.

        Args:
            producer_class: The OpsProducer subclass to register.
            name: Optional name to use as the registry key.
                  If not provided, the class name is used.
            addon_name: Optional name of the addon registering this producer.
                        Used for cleanup when addon is unloaded.
        """
        key = name if name else producer_class.__name__
        self._producers[key] = producer_class
        if addon_name:
            if addon_name not in self._addon_items:
                self._addon_items[addon_name] = set()
            self._addon_items[addon_name].add(key)

    def unregister(self, name: str) -> bool:
        """
        Unregister a producer class by name.

        Args:
            name: The name of the producer to unregister.

        Returns:
            True if the producer was unregistered, False if not found.
        """
        if name in self._producers:
            del self._producers[name]
            for addon_name, items in self._addon_items.items():
                items.discard(name)
            return True
        return False

    def unregister_all_from_addon(self, addon_name: str) -> int:
        """
        Unregister all producers registered by a specific addon.

        Args:
            addon_name: The name of the addon.

        Returns:
            The number of producers unregistered.
        """
        if addon_name not in self._addon_items:
            return 0
        items = self._addon_items.pop(addon_name)
        count = 0
        for name in items:
            if name in self._producers:
                del self._producers[name]
                count += 1
        return count

    def get(self, name: str) -> Optional[Type["OpsProducer"]]:
        """
        Look up a producer class by name.

        Args:
            name: The class name of the producer.

        Returns:
            The producer class, or None if not found.
        """
        # Ensure addons are loaded in worker processes so the registry
        # is populated before we try to look up producer classes.
        from rayforge.worker_init import ensure_addons_loaded

        ensure_addons_loaded()

        return self._producers.get(name)

    def all_producers(self) -> Dict[str, Type["OpsProducer"]]:
        """
        Return a copy of all registered producers.

        Returns:
            Dictionary mapping producer names to classes.
        """
        return self._producers.copy()


producer_registry = ProducerRegistry()
