from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
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


@dataclass
class TextureInstance:
    """Represents a single texture and its placement in the world."""

    texture_data: TextureData
    world_transform: np.ndarray  # 4x4 matrix

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texture_data": self.texture_data.to_dict(),
            "world_transform": self.world_transform.tolist(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TextureInstance":
        return cls(
            texture_data=TextureData.from_dict(data["texture_data"]),
            world_transform=np.array(
                data["world_transform"], dtype=np.float32
            ),
        )


class BaseArtifact(ABC):
    """
    Abstract base class for all artifact types in the pipeline.
    Contains common fields shared by all artifact types and manages a registry
    of all its subclasses for dynamic instantiation.
    """

    _registry: Dict[str, Type[BaseArtifact]] = {}

    def __init_subclass__(cls, **kwargs):
        """
        This special method is called whenever a class inherits from
        BaseArtifact. It automatically registers the new artifact type.
        """
        super().__init_subclass__(**kwargs)
        cls._registry[cls.__name__] = cls

    @classmethod
    def get_registered_class(cls, name: str) -> Type[BaseArtifact]:
        """Looks up an artifact class in the registry by its name."""
        try:
            return cls._registry[name]
        except KeyError:
            raise TypeError(
                f"Unknown artifact type '{name}'. Was its module imported?"
            )

    @property
    def artifact_type(self) -> str:
        """Returns the type of the artifact."""
        return self.__class__.__name__

    @abstractmethod
    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> BaseArtifactHandle:
        """
        Creates the appropriate, typed handle for this artifact.
        Each subclass must implement this to return its specific handle type.
        """
        raise NotImplementedError

    @abstractmethod
    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        """
        Gets a dictionary of all NumPy arrays that need to be stored in
        shared memory for this artifact.
        """
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def from_storage(
        cls: Type[BaseArtifact],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> BaseArtifact:
        """
        Reconstructs an artifact instance from its handle and a dictionary of
        NumPy array views from shared memory.
        """
        raise NotImplementedError
