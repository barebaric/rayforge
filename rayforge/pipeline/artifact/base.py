from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np

from .handle import BaseArtifactHandle


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


class BaseArtifact(ABC):
    @property
    def artifact_type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def build_handle(self, key: str) -> BaseArtifactHandle:
        pass
