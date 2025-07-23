from abc import ABC, abstractmethod
from ..models.ops import Ops


class OpsProducer(ABC):
    """
    Given a Cairo surface, an OpsProducer outputs an Ops object.
    Examples may include:

    - Tracing a bitmap to produce a path (Ops object).
    - Reading vector data from an image to turn it into Ops.
    """
    @abstractmethod
    def run(self, laser, surface, pixels_per_mm) -> Ops:
        pass

    def can_scale(self) -> bool:
        """
        Returns True if the produced Ops object is scalable. This allows
        the consumer to cache the Ops object more often, as it does not
        need to be re-made just because the input image was resized.
        """
        return True
