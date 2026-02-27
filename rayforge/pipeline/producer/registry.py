from typing import Dict, Optional, Type, TYPE_CHECKING

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

    def register(
        self,
        producer_class: Type["OpsProducer"],
        name: Optional[str] = None,
    ) -> None:
        """
        Register a producer class.

        Args:
            producer_class: The OpsProducer subclass to register.
            name: Optional name to use as the registry key.
                  If not provided, the class name is used.
        """
        key = name if name else producer_class.__name__
        self._producers[key] = producer_class

    def get(self, name: str) -> Optional[Type["OpsProducer"]]:
        """
        Look up a producer class by name.

        Args:
            name: The class name of the producer.

        Returns:
            The producer class, or None if not found.
        """
        return self._producers.get(name)

    def all_producers(self) -> Dict[str, Type["OpsProducer"]]:
        """
        Return a copy of all registered producers.

        Returns:
            Dictionary mapping producer names to classes.
        """
        return self._producers.copy()


producer_registry = ProducerRegistry()
