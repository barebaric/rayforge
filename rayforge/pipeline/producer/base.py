from abc import ABC, abstractmethod
from ...core.ops import Ops


class OpsProducer(ABC):
    """
    Given a Cairo surface, an OpsProducer outputs an Ops object.
    Examples may include:

    - Tracing a bitmap to produce a path (Ops object).
    - Reading vector data from an image to turn it into Ops.
    """

    @abstractmethod
    def run(
        self, laser, surface, pixels_per_mm, y_offset_mm: float = 0.0
    ) -> Ops:
        pass

    def can_scale(self) -> bool:
        """
        Returns True if the produced Ops object is scalable. This allows
        the consumer to cache the Ops object more often, as it does not
        need to be re-made just because the input image was resized.
        """
        return True

    def to_dict(self) -> dict:
        """
        Serializes the producer configuration to a dictionary.

        This dictionary can be used with `OpsProducer.from_dict` to
        recreate the producer instance.
        """
        return {
            "type": self.__class__.__name__,
            "params": {},  # All current producers are stateless
        }

    @staticmethod
    def from_dict(data: dict):
        """
        Deserializes a producer from a dictionary.

        This is a factory method that looks up the producer class by its
        name from the central registry and instantiates it.
        """
        # Local import to avoid a circular dependency at module-load time.
        # The producer_by_name map is built in the package's __init__.py,
        # which imports this module.
        from . import producer_by_name

        producer_type = data.get("type")
        if not producer_type:
            raise ValueError("Input dictionary must contain a 'type' key.")

        ProducerClass = producer_by_name.get(producer_type)

        if not ProducerClass:
            raise ValueError(f"Unknown producer type: '{producer_type}'")

        # Instantiate the class with parameters from the dictionary.
        # This allows for future producers to have configurable state.
        params = data.get("params", {})
        return ProducerClass(**params)
