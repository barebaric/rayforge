from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from .handle import BaseArtifactHandle


@dataclass
class TextureData:
    """A container for texture-based raster data."""

    power_texture_data: np.ndarray
    dimensions_mm: tuple[float, float]
    position_mm: tuple[float, float]


class BaseArtifact(ABC):
    @property
    def artifact_type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def build_handle(self, key: str) -> BaseArtifactHandle:
        pass
