from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any
from ...core.ops import Ops
from ..coord import CoordinateSystem


class Artifact:
    """
    A self-describing output of an OpsProducer.
    This class uses composition to hold different types of data (vector,
    vertex, raster) instead of a complex inheritance hierarchy.
    """

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        source_dimensions: Optional[Tuple[float, float]] = None,
        generation_size: Optional[Tuple[float, float]] = None,
        vertex_data: Optional[Dict[str, np.ndarray]] = None,
        raster_data: Optional[Dict[str, Any]] = None,
    ):
        self.ops = ops
        self.is_scalable = is_scalable
        self.source_coordinate_system = source_coordinate_system
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size
        self.vertex_data = vertex_data
        self.raster_data = raster_data

    @property
    def artifact_type(self) -> str:
        """Determines the artifact type based on its data components."""
        if self.raster_data:
            return "hybrid_raster"
        if self.vertex_data:
            return "vertex"
        return "vector"

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the artifact properties to a dictionary."""
        data = {
            "artifact_type": self.artifact_type,
            "ops": self.ops.to_dict(),
            "is_scalable": self.is_scalable,
            "source_coordinate_system": self.source_coordinate_system.name,
            "source_dimensions": self.source_dimensions,
            "generation_size": self.generation_size,
        }
        if self.vertex_data:
            data["vertex_data"] = {
                key: arr.tolist() for key, arr in self.vertex_data.items()
            }
        if self.raster_data:
            raster_serializable = self.raster_data.copy()
            if "power_texture_data" in raster_serializable:
                raster_serializable["power_texture_data"] = (
                    raster_serializable["power_texture_data"].tolist()
                )
            data["raster_data"] = raster_serializable
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Deserializes a dictionary into an Artifact instance."""
        vertex_data = None
        if "vertex_data" in data and data["vertex_data"]:
            vertex_data = {
                "powered_vertices": np.array(
                    data["vertex_data"].get("powered_vertices", []),
                    dtype=np.float32,
                ).reshape(-1, 3),
                "powered_colors": np.array(
                    data["vertex_data"].get("powered_colors", []),
                    dtype=np.float32,
                ).reshape(-1, 4),
                "travel_vertices": np.array(
                    data["vertex_data"].get("travel_vertices", []),
                    dtype=np.float32,
                ).reshape(-1, 3),
                "zero_power_vertices": np.array(
                    data["vertex_data"].get("zero_power_vertices", []),
                    dtype=np.float32,
                ).reshape(-1, 3),
            }

        raster_data = None
        if "raster_data" in data and data["raster_data"]:
            raster_data = data["raster_data"].copy()
            if "power_texture_data" in raster_data:
                raster_data["power_texture_data"] = np.array(
                    raster_data["power_texture_data"], dtype=np.uint8
                )

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
            vertex_data=vertex_data,
            raster_data=raster_data,
        )
