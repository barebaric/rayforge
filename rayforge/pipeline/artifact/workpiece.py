from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, Type, TYPE_CHECKING
from ..coord import CoordinateSystem
from .base import BaseArtifact, VertexData, TextureData
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ...core.ops import Ops


class WorkPieceArtifactHandle(BaseArtifactHandle):
    """A handle for a WorkPieceArtifact, with specific metadata."""

    def __init__(
        self,
        # Required arguments
        is_scalable: bool,
        source_coordinate_system_name: str,
        generation_size: Tuple[float, float],
        shm_name: str,
        handle_class_name: str,
        artifact_type_name: str,
        # Optional arguments
        source_dimensions: Optional[Tuple[float, float]] = None,
        array_metadata: Optional[Dict[str, Any]] = None,
        dimensions_mm: Optional[Tuple[float, float]] = None,
        position_mm: Optional[Tuple[float, float]] = None,
        **_kwargs,
    ):
        super().__init__(
            shm_name=shm_name,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            array_metadata=array_metadata,
        )
        self.is_scalable = is_scalable
        self.source_coordinate_system_name = source_coordinate_system_name
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size
        self.dimensions_mm = dimensions_mm
        self.position_mm = position_mm


class WorkPieceArtifact(BaseArtifact):
    """
    Represents an intermediate artifact produced during the pipeline,
    containing vertex and texture data for visualization.
    """

    def __init__(
        self,
        ops: "Ops",
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        generation_size: Tuple[float, float],
        source_dimensions: Optional[Tuple[float, float]] = None,
        vertex_data: Optional[VertexData] = None,
        texture_data: Optional[TextureData] = None,
    ):
        super().__init__()
        self.ops = ops
        self.is_scalable = is_scalable
        self.source_coordinate_system = source_coordinate_system
        self.source_dimensions = source_dimensions
        self.vertex_data: Optional[VertexData] = vertex_data
        self.texture_data: Optional[TextureData] = texture_data
        self.generation_size = generation_size

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = {
            "ops": self.ops.to_dict(),
            "is_scalable": self.is_scalable,
            "source_coordinate_system": self.source_coordinate_system.name,
            "source_dimensions": self.source_dimensions,
            "generation_size": self.generation_size,
        }
        if self.vertex_data:
            result["vertex_data"] = self.vertex_data.to_dict()
        if self.texture_data:
            result["texture_data"] = self.texture_data.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPieceArtifact":
        """Creates an artifact from a dictionary."""
        from ...core.ops import Ops

        ops = Ops.from_dict(data["ops"])
        common_args = {
            "ops": ops,
            "is_scalable": data["is_scalable"],
            "source_coordinate_system": CoordinateSystem[
                data["source_coordinate_system"]
            ],
            "source_dimensions": data.get("source_dimensions"),
            "generation_size": tuple(data["generation_size"]),
        }
        if "vertex_data" in data:
            common_args["vertex_data"] = VertexData.from_dict(
                data["vertex_data"]
            )
        if "texture_data" in data:
            common_args["texture_data"] = TextureData.from_dict(
                data["texture_data"]
            )
        return cls(**common_args)

    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> WorkPieceArtifactHandle:
        texture_dims = None
        texture_pos = None
        if self.texture_data:
            texture_dims = self.texture_data.dimensions_mm
            texture_pos = self.texture_data.position_mm

        return WorkPieceArtifactHandle(
            shm_name=shm_name,
            handle_class_name=WorkPieceArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            is_scalable=self.is_scalable,
            source_coordinate_system_name=self.source_coordinate_system.name,
            source_dimensions=self.source_dimensions,
            array_metadata=array_metadata,
            generation_size=self.generation_size,
            dimensions_mm=texture_dims,
            position_mm=texture_pos,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        arrays = self.ops.to_numpy_arrays()
        if self.texture_data is not None:
            arrays["power_texture_data"] = self.texture_data.power_texture_data
        if self.vertex_data is not None:
            arrays["powered_vertices"] = self.vertex_data.powered_vertices
            arrays["powered_colors"] = self.vertex_data.powered_colors
            arrays["travel_vertices"] = self.vertex_data.travel_vertices
            arrays["zero_power_vertices"] = (
                self.vertex_data.zero_power_vertices
            )
        return arrays

    @classmethod
    def from_storage(
        cls: Type[WorkPieceArtifact],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> WorkPieceArtifact:
        if not isinstance(handle, WorkPieceArtifactHandle):
            raise TypeError(
                "WorkPieceArtifact requires a WorkPieceArtifactHandle"
            )
        from ...core.ops import Ops

        ops = Ops.from_numpy_arrays(arrays)
        vertex_data = None
        if all(
            key in arrays
            for key in [
                "powered_vertices",
                "powered_colors",
                "travel_vertices",
                "zero_power_vertices",
            ]
        ):
            vertex_data = VertexData(
                powered_vertices=arrays["powered_vertices"].copy(),
                powered_colors=arrays["powered_colors"].copy(),
                travel_vertices=arrays["travel_vertices"].copy(),
                zero_power_vertices=arrays["zero_power_vertices"].copy(),
            )
        texture_data = None
        if "power_texture_data" in arrays:
            if handle.dimensions_mm is None or handle.position_mm is None:
                raise ValueError(
                    "Handle for texture artifact is missing required "
                    "dimensions_mm or position_mm metadata."
                )
            texture_data = TextureData(
                power_texture_data=arrays["power_texture_data"].copy(),
                dimensions_mm=handle.dimensions_mm,
                position_mm=handle.position_mm,
            )
        return cls(
            ops=ops,
            is_scalable=handle.is_scalable,
            source_coordinate_system=CoordinateSystem[
                handle.source_coordinate_system_name
            ],
            source_dimensions=handle.source_dimensions,
            generation_size=handle.generation_size,
            vertex_data=vertex_data,
            texture_data=texture_data,
        )
