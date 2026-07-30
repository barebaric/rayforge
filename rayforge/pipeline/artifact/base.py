from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .handle import BaseArtifactHandle


@dataclass
class TextureData:
    """A container for texture-based raster data."""

    power_texture_data: np.ndarray
    dimensions_mm: Tuple[float, float]
    position_mm: Tuple[float, float]


class BaseArtifact(ABC):
    @property
    def artifact_type(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def build_handle(self, key: str) -> BaseArtifactHandle:
        pass
