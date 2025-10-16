from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Type
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .base import BaseArtifact, VertexData, TextureData
from .handle import BaseArtifactHandle


@dataclass
class StepArtifactHandle(BaseArtifactHandle):
    """A handle for a StepArtifact."""

    # Metadata for reconstructing a potential composite texture
    dimensions_mm: Optional[Tuple[float, float]] = None
    position_mm: Optional[Tuple[float, float]] = None


class StepArtifact(BaseArtifact):
    """
    Represents an intermediate artifact for an entire Step, after all its
    WorkPieceArtifacts have been aggregated and per-step transformers have
    been applied.

    This artifact contains the final, accurate data for 3D visualization.
    """

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        source_dimensions: Optional[Tuple[float, float]] = None,
        time_estimate: Optional[float] = None,
        vertex_data: Optional[VertexData] = None,
        texture_data: Optional[TextureData] = None,
    ):
        super().__init__(
            ops=ops,
            is_scalable=is_scalable,
            source_coordinate_system=source_coordinate_system,
            source_dimensions=source_dimensions,
            time_estimate=time_estimate,
        )
        self.vertex_data: Optional[VertexData] = vertex_data
        self.texture_data: Optional[TextureData] = texture_data

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = super().to_dict()
        if self.vertex_data:
            result["vertex_data"] = self.vertex_data.to_dict()
        if self.texture_data:
            result["texture_data"] = self.texture_data.to_dict()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepArtifact":
        """Creates an artifact from a dictionary."""
        ops = Ops.from_dict(data["ops"])
        common_args = {
            "ops": ops,
            "is_scalable": data["is_scalable"],
            "source_coordinate_system": CoordinateSystem[
                data["source_coordinate_system"]
            ],
            "source_dimensions": data.get("source_dimensions"),
            "time_estimate": data.get("time_estimate"),
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
    ) -> StepArtifactHandle:
        """Creates the appropriate, typed handle for this artifact."""
        texture_dims = None
        texture_pos = None
        if self.texture_data:
            texture_dims = self.texture_data.dimensions_mm
            texture_pos = self.texture_data.position_mm

        return StepArtifactHandle(
            shm_name=shm_name,
            handle_class_name=StepArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            is_scalable=self.is_scalable,
            source_coordinate_system_name=self.source_coordinate_system.name,
            source_dimensions=self.source_dimensions,
            time_estimate=self.time_estimate,
            array_metadata=array_metadata,
            dimensions_mm=texture_dims,
            position_mm=texture_pos,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        """
        Gets a dictionary of all NumPy arrays that need to be stored in
        shared memory for this artifact.
        """
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
        cls: Type[StepArtifact],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> StepArtifact:
        """
        Reconstructs an artifact instance from its handle and a dictionary of
        NumPy array views from shared memory.
        """
        if not isinstance(handle, StepArtifactHandle):
            raise TypeError(
                "StepArtifact requires a StepArtifactHandle for reconstruction"
            )

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
            time_estimate=handle.time_estimate,
            vertex_data=vertex_data,
            texture_data=texture_data,
        )
