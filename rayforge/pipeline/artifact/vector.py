from __future__ import annotations
from typing import Dict, Any
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .base import Artifact


class VectorArtifact(Artifact):
    """
    Represents a resolution-independent vector artifact.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = "vector"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VectorArtifact":
        """Deserializes a dictionary back into a VectorArtifact instance."""
        return cls(
            ops=Ops.from_dict(data["ops"]),
            is_scalable=data["is_scalable"],
            source_coordinate_system=CoordinateSystem[
                data["source_coordinate_system"]
            ],
            source_dimensions=tuple(data["source_dimensions"])
            if data.get("source_dimensions")
            else None,
            generation_size=tuple(data["generation_size"])
            if data.get("generation_size")
            else None,
        )
