from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np

from .handle import BaseArtifactHandle


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
