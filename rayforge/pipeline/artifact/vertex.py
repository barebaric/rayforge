from __future__ import annotations
from typing import Dict, Any
import numpy as np
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .base import Artifact


class VertexArtifact(Artifact):
    """
    Represents a vertex-based artifact with pre-computed GPU-friendly vertex
    arrays.
    """

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        powered_vertices: np.ndarray,
        powered_colors: np.ndarray,
        travel_vertices: np.ndarray,
        zero_power_vertices: np.ndarray,
        source_dimensions: tuple[float, float] | None = None,
        generation_size: tuple[float, float] | None = None,
    ):
        super().__init__(
            ops=ops,
            is_scalable=is_scalable,
            source_coordinate_system=source_coordinate_system,
            source_dimensions=source_dimensions,
            generation_size=generation_size,
        )
        self.type = "vertex"
        self.powered_vertices = powered_vertices
        self.powered_colors = powered_colors
        self.travel_vertices = travel_vertices
        self.zero_power_vertices = zero_power_vertices

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the VertexArtifact to a dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "powered_vertices": self.powered_vertices.tolist(),
                "powered_colors": self.powered_colors.tolist(),
                "travel_vertices": self.travel_vertices.tolist(),
                "zero_power_vertices": self.zero_power_vertices.tolist(),
            }
        )
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexArtifact":
        """Deserializes a dictionary back into a VertexArtifact instance."""
        return cls(
            ops=Ops.from_dict(data["ops"]),
            is_scalable=data["is_scalable"],
            source_coordinate_system=CoordinateSystem[
                data["source_coordinate_system"]
            ],
            powered_vertices=np.array(
                data["powered_vertices"], dtype=np.float32
            ).reshape(-1, 3),
            powered_colors=np.array(
                data["powered_colors"], dtype=np.float32
            ).reshape(-1, 4),
            travel_vertices=np.array(
                data["travel_vertices"], dtype=np.float32
            ).reshape(-1, 3),
            zero_power_vertices=np.array(
                data["zero_power_vertices"], dtype=np.float32
            ).reshape(-1, 3),
            source_dimensions=tuple(data["source_dimensions"])
            if data.get("source_dimensions")
            else None,
            generation_size=tuple(data["generation_size"])
            if data.get("generation_size")
            else None,
        )
