from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass, field
from ...core.ops import Ops
from ..coord import CoordinateSystem


@dataclass
class VertexData:
    """A container for GPU-friendly vertex arrays."""

    powered_vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    powered_colors: np.ndarray = field(
        default_factory=lambda: np.empty((0, 4), dtype=np.float32)
    )
    travel_vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )
    zero_power_vertices: np.ndarray = field(
        default_factory=lambda: np.empty((0, 3), dtype=np.float32)
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "powered_vertices": self.powered_vertices.tolist(),
            "powered_colors": self.powered_colors.tolist(),
            "travel_vertices": self.travel_vertices.tolist(),
            "zero_power_vertices": self.zero_power_vertices.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VertexData":
        return cls(
            powered_vertices=np.array(
                data.get("powered_vertices", []), dtype=np.float32
            ).reshape(-1, 3),
            powered_colors=np.array(
                data.get("powered_colors", []), dtype=np.float32
            ).reshape(-1, 4),
            travel_vertices=np.array(
                data.get("travel_vertices", []), dtype=np.float32
            ).reshape(-1, 3),
            zero_power_vertices=np.array(
                data.get("zero_power_vertices", []), dtype=np.float32
            ).reshape(-1, 3),
        )


@dataclass
class TextureData:
    """A container for texture-based raster data."""

    power_texture_data: np.ndarray
    dimensions_mm: Tuple[float, float]
    position_mm: Tuple[float, float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "power_texture_data": self.power_texture_data.tolist(),
            "dimensions_mm": self.dimensions_mm,
            "position_mm": self.position_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextureData":
        return cls(
            power_texture_data=np.array(
                data["power_texture_data"], dtype=np.uint8
            ),
            dimensions_mm=tuple(data["dimensions_mm"]),
            position_mm=tuple(data["position_mm"]),
        )


class Artifact:
    """
    A self-describing output of an OpsProducer.
    This class uses composition to hold different types of data (vector,
    vertex, texture) instead of a complex inheritance hierarchy.
    """

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        source_dimensions: Optional[Tuple[float, float]] = None,
        generation_size: Optional[Tuple[float, float]] = None,
        vertex_data: Optional[VertexData] = None,
        texture_data: Optional[TextureData] = None,
        gcode_bytes: Optional[np.ndarray] = None,
        op_map_bytes: Optional[np.ndarray] = None,
    ):
        self.ops = ops
        self.is_scalable = is_scalable
        self.source_coordinate_system = source_coordinate_system
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size
        self.vertex_data = vertex_data
        self.texture_data = texture_data
        self.gcode_bytes = gcode_bytes
        self.op_map_bytes = op_map_bytes

    @property
    def artifact_type(self) -> str:
        """Determines the artifact type based on its data components."""
        if self.gcode_bytes is not None or self.op_map_bytes is not None:
            return "final_job"
        if self.texture_data:
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
            data["vertex_data"] = self.vertex_data.to_dict()
        if self.texture_data:
            data["texture_data"] = self.texture_data.to_dict()
        if self.gcode_bytes is not None:
            data["gcode_bytes"] = self.gcode_bytes.tolist()
        if self.op_map_bytes is not None:
            data["op_map_bytes"] = self.op_map_bytes.tolist()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Artifact":
        """Deserializes a dictionary into an Artifact instance."""
        vertex_data = (
            VertexData.from_dict(data["vertex_data"])
            if "vertex_data" in data and data["vertex_data"]
            else None
        )
        texture_data = (
            TextureData.from_dict(data["texture_data"])
            if "texture_data" in data and data["texture_data"]
            else None
        )
        gcode_bytes = (
            np.array(data["gcode_bytes"], dtype=np.uint8)
            if "gcode_bytes" in data and data["gcode_bytes"] is not None
            else None
        )
        op_map_bytes = (
            np.array(data["op_map_bytes"], dtype=np.uint8)
            if "op_map_bytes" in data and data["op_map_bytes"] is not None
            else None
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
            texture_data=texture_data,
            gcode_bytes=gcode_bytes,
            op_map_bytes=op_map_bytes,
        )
