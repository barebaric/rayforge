from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import numpy as np
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .vertex import VertexArtifact


class HybridRasterArtifact(VertexArtifact):
    """
    An artifact that combines raster texture data with vector operations.
    It inherits from VertexArtifact to include pre-computed vertex arrays for
    rendering the vector component.
    """

    def __init__(
        self,
        power_texture_data: np.ndarray,
        dimensions_mm: Tuple[float, float],
        position_mm: Tuple[float, float],
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        # Optional vertex data fields inherited from VertexArtifact
        powered_vertices: Optional[np.ndarray] = None,
        powered_colors: Optional[np.ndarray] = None,
        travel_vertices: Optional[np.ndarray] = None,
        zero_power_vertices: Optional[np.ndarray] = None,
        source_dimensions: Optional[Tuple[float, float]] = None,
        generation_size: Optional[Tuple[float, float]] = None,
    ):
        """Initializes the HybridRasterArtifact."""
        # Handle default empty arrays for vertex data if not provided
        if powered_vertices is None:
            powered_vertices = np.empty((0, 3), dtype=np.float32)
        if powered_colors is None:
            powered_colors = np.empty((0, 4), dtype=np.float32)
        if travel_vertices is None:
            travel_vertices = np.empty((0, 3), dtype=np.float32)
        if zero_power_vertices is None:
            zero_power_vertices = np.empty((0, 3), dtype=np.float32)

        super().__init__(
            ops=ops,
            is_scalable=is_scalable,
            source_coordinate_system=source_coordinate_system,
            powered_vertices=powered_vertices,
            powered_colors=powered_colors,
            travel_vertices=travel_vertices,
            zero_power_vertices=zero_power_vertices,
            source_dimensions=source_dimensions,
            generation_size=generation_size,
        )
        self.type = "hybrid_raster"
        self.power_texture_data = power_texture_data
        self.dimensions_mm = dimensions_mm
        self.position_mm = position_mm

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the artifact to a dictionary."""
        base_dict = super().to_dict()
        base_dict.update(
            {
                "power_texture_data": self.power_texture_data.tolist(),
                "dimensions_mm": self.dimensions_mm,
                "position_mm": self.position_mm,
            }
        )
        return base_dict

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridRasterArtifact":
        """Deserializes a dictionary into a HybridRasterArtifact instance."""
        return cls(
            power_texture_data=np.array(
                data["power_texture_data"], dtype=np.uint8
            ),
            dimensions_mm=tuple(data["dimensions_mm"]),
            position_mm=tuple(data["position_mm"]),
            # Fields from VertexArtifact
            ops=Ops.from_dict(data["ops"]),
            is_scalable=data["is_scalable"],
            source_coordinate_system=CoordinateSystem[
                data["source_coordinate_system"]
            ],
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
            source_dimensions=tuple(data["source_dimensions"])
            if data.get("source_dimensions")
            else None,
            generation_size=tuple(data["generation_size"])
            if data.get("generation_size")
            else None,
        )
