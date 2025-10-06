from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .base import Artifact


class HybridRasterArtifact(Artifact):
    """
    A hybrid artifact for high-performance raster rendering. This class
    replaces the original HybridRasterArtifact dataclass.
    """

    def __init__(
        self,
        power_texture_data: np.ndarray,
        dimensions_mm: Tuple[float, float],
        position_mm: Tuple[float, float],
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.power_texture_data = power_texture_data
        self.dimensions_mm = dimensions_mm
        self.position_mm = position_mm
        self.type = "hybrid_raster"

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the artifact to a dictionary for inter-process transfer.
        """
        data = super().to_dict()
        data.update(
            {
                "power_texture_data": self.power_texture_data.tolist(),
                "dimensions_mm": self.dimensions_mm,
                "position_mm": self.position_mm,
            }
        )
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HybridRasterArtifact":
        """Deserializes a dict back into a HybridRasterArtifact instance."""
        return cls(
            ops=Ops.from_dict(data["ops"]),
            is_scalable=data["is_scalable"],
            source_coordinate_system=CoordinateSystem[
                data["source_coordinate_system"]
            ],
            power_texture_data=np.array(
                data["power_texture_data"], dtype=np.uint8
            ),
            dimensions_mm=tuple(data["dimensions_mm"]),
            position_mm=tuple(data["position_mm"]),
            source_dimensions=tuple(data["source_dimensions"])
            if data.get("source_dimensions") is not None
            else None,
            generation_size=tuple(data["generation_size"])
            if data.get("generation_size")
            else None,
        )
