from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Tuple, Dict, Any
from enum import Enum, auto
from dataclasses import dataclass, asdict
from ...core.ops import Ops

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece


class CoordinateSystem(Enum):
    """Defines the coordinate space in which Ops were generated."""

    PIXEL_SPACE = auto()
    NATIVE_VECTOR_SPACE = auto()
    MILLIMETER_SPACE = auto()


@dataclass
class PipelineArtifact:
    """A self-describing container for the output of an OpsProducer."""

    ops: Ops
    is_scalable: bool
    source_coordinate_system: CoordinateSystem
    source_dimensions: Optional[Tuple[float, float]] = None
    generation_size: Optional[Tuple[float, float]] = None

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the artifact to a dictionary for inter-process transfer.
        """
        data = asdict(self)
        data["ops"] = self.ops.to_dict()
        data["source_coordinate_system"] = self.source_coordinate_system.name
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineArtifact":
        """Deserializes a dictionary back into a PipelineArtifact instance."""
        return cls(
            ops=Ops.from_dict(data["ops"]),
            is_scalable=data["is_scalable"],
            source_coordinate_system=CoordinateSystem[
                data["source_coordinate_system"]
            ],
            source_dimensions=tuple(data["source_dimensions"])
            if data["source_dimensions"]
            else None,
            generation_size=tuple(data["generation_size"])
            if data["generation_size"]
            else None,
        )


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
        laser,
        surface,
        pixels_per_mm,
        *,
        workpiece: "Optional[WorkPiece]" = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        pass

    def can_scale(self) -> bool:
        """
        Returns True if the produced Ops object is scalable. This allows
        the consumer to cache the Ops object more often, as it does not
        need to be re-made just because the input image was resized.
        """
        return True

    @property
    def requires_full_render(self) -> bool:
        """
        Returns True if a producer requires the entire workpiece to be
        rendered into a single surface, even if its output is scalable.
        This is essential for algorithms that need a global view of the image,
        like hulling, and forces the pipeline to provide a raster input.
        """
        return False

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
