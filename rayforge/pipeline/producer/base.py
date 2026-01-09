from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Dict, Any
from enum import Enum, auto
from ..artifact import WorkPieceArtifact


if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ...shared.tasker.proxy import BaseExecutionContext


class CutSide(Enum):
    """Defines which side of a path the laser cut should be on."""

    CENTERLINE = auto()
    """The center of the laser beam follows the path directly."""
    INSIDE = auto()
    """The final cut will be inside the original path."""
    OUTSIDE = auto()
    """The final cut will be outside the original path."""


class OpsProducer(ABC):
    """
    Given a Cairo surface, an OpsProducer outputs an Ops object.
    Examples may include:

    - Tracing a bitmap to produce a path (Ops object).
    - Reading vector data from an image to turn it into Ops.
    """

    @abstractmethod
    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        proxy: Optional["BaseExecutionContext"] = None,
    ) -> WorkPieceArtifact:
        pass

    def is_vector_producer(self) -> bool:
        """
        Specifies the generation strategy for the producer.

        - True: Use the vector/full-render path. The producer can handle
          vector inputs directly, or it traces a single, fully-rendered
          raster image.
        - False: Use the chunked raster path. The producer requires the
          input to be rendered and fed to it in horizontal strips.

        This controls the *process* of generation, while the artifact's
        `is_scalable` flag controls the caching behavior of the *product*.
        """
        return True

    @property
    def requires_full_render(self) -> bool:
        """
        Returns True if a producer requires the entire workpiece to be
        rendered into a single surface, even though its output is scalable.
        This is essential for algorithms that need a global view of the image,
        like hulling, and forces the pipeline to provide a raster input.
        """
        return False

    @property
    def supports_kerf(self) -> bool:
        """
        Returns True if this producer's logic can apply kerf compensation.
        This is typically True for vector-based producers.
        """
        return False

    @property
    def supports_cut_speed(self) -> bool:
        """
        Returns True if this producer's logic supports a fixed cut speed.
        Producers that have variable speed logic may not.
        """
        return True

    @property
    def supports_power(self) -> bool:
        """
        Returns True if this producer's logic supports a fixed power setting.
        Producers that have variable power logic may not.
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
            "params": {},  # Default for stateless producers
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OpsProducer":
        """
        Deserializes a producer from a dictionary.

        This is a factory method that looks up the producer class by its
        name from the central registry and dispatches to the class's own
        `from_dict` method.
        """
        from . import producer_by_name

        producer_type = data.get("type")
        if not producer_type:
            raise ValueError("Input dictionary must contain a 'type' key.")

        ProducerClass = producer_by_name.get(producer_type)
        if not ProducerClass:
            raise ValueError(f"Unknown producer type: '{producer_type}'")

        # Dispatch to the specific class's from_dict method
        return ProducerClass.from_dict(data)
