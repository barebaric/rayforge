from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
from ...core.ops import Ops
from ..producer.base import CoordinateSystem


class Artifact(ABC):
    """
    An abstract base class for the self-describing output of an OpsProducer.
    This replaces the previous dataclass-based approach to allow for more
    flexible methods and inheritance.
    """

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        source_dimensions: Optional[Tuple[float, float]] = None,
        generation_size: Optional[Tuple[float, float]] = None,
    ):
        self.ops = ops
        self.is_scalable = is_scalable
        self.source_coordinate_system = source_coordinate_system
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size
        # The 'type' property will be defined by subclasses for serialization
        self.type: str = "base"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the common artifact properties to a dictionary.
        Subclasses should extend this.
        """
        return {
            "type": self.type,
            "ops": self.ops.to_dict(),
            "is_scalable": self.is_scalable,
            "source_coordinate_system": self.source_coordinate_system.name,
            "source_dimensions": self.source_dimensions,
            "generation_size": self.generation_size,
        }

    @classmethod
    @abstractmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """
        Deserializes a dictionary into an Artifact instance.
        Must be implemented by subclasses.
        """
        raise NotImplementedError
