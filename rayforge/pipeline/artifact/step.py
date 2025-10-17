from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Type, List
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .base import BaseArtifact, VertexData, TextureData
from .handle import BaseArtifactHandle


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


@dataclass
class StepArtifactHandle(BaseArtifactHandle):
    """A handle for a StepArtifact."""

    # We no longer need texture metadata here, as the full instance data
    # will be stored within the artifact's shared memory.
    pass


class StepArtifact(BaseArtifact):
    """
    Represents an intermediate artifact for an entire Step, after all its
    WorkPieceArtifacts have been aggregated. This is a "render bundle"
    consumed by the 3D canvas.
    """

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        source_coordinate_system: CoordinateSystem,
        source_dimensions: Optional[Tuple[float, float]] = None,
        time_estimate: Optional[float] = None,
        vertex_data: Optional[VertexData] = None,
        texture_instances: Optional[List[TextureInstance]] = None,
    ):
        super().__init__(
            ops=ops,
            is_scalable=is_scalable,
            source_coordinate_system=source_coordinate_system,
            source_dimensions=source_dimensions,
            time_estimate=time_estimate,
        )
        self.vertex_data: Optional[VertexData] = vertex_data
        self.texture_instances: List[TextureInstance] = (
            texture_instances if texture_instances is not None else []
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = super().to_dict()
        if self.vertex_data:
            result["vertex_data"] = self.vertex_data.to_dict()
        result["texture_instances"] = [
            ti.to_dict() for ti in self.texture_instances
        ]
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
        common_args["texture_instances"] = [
            TextureInstance.from_dict(ti)
            for ti in data.get("texture_instances", [])
        ]
        return cls(**common_args)

    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> StepArtifactHandle:
        """Creates the appropriate, typed handle for this artifact."""
        return StepArtifactHandle(
            shm_name=shm_name,
            handle_class_name=StepArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            is_scalable=self.is_scalable,
            source_coordinate_system_name=self.source_coordinate_system.name,
            source_dimensions=self.source_dimensions,
            time_estimate=self.time_estimate,
            array_metadata=array_metadata,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        """
        Gets a dictionary of all NumPy arrays that need to be stored in
        shared memory for this artifact.
        """
        arrays = self.ops.to_numpy_arrays()
        if self.vertex_data is not None:
            arrays["powered_vertices"] = self.vertex_data.powered_vertices
            arrays["powered_colors"] = self.vertex_data.powered_colors
            arrays["travel_vertices"] = self.vertex_data.travel_vertices
            arrays["zero_power_vertices"] = (
                self.vertex_data.zero_power_vertices
            )

        # Store each texture's data and transform matrix
        for i, instance in enumerate(self.texture_instances):
            arrays[f"texture_data_{i}"] = (
                instance.texture_data.power_texture_data
            )
            arrays[f"texture_transform_{i}"] = instance.world_transform
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

        texture_instances = []
        i = 0
        while f"texture_data_{i}" in arrays:
            tex_data_arr = arrays[f"texture_data_{i}"]
            transform_arr = arrays[f"texture_transform_{i}"]
            # Reconstruct dimensions from the texture array shape
            h, w = tex_data_arr.shape
            # This assumes 1:1 pixel to mm, which may need refinement,
            # but is a reasonable starting point for reconstruction.
            dims = (float(w), float(h))
            texture_data = TextureData(
                power_texture_data=tex_data_arr.copy(),
                dimensions_mm=dims,
                position_mm=(0, 0),  # Position is now in the transform
            )
            instance = TextureInstance(
                texture_data=texture_data,
                world_transform=transform_arr.copy(),
            )
            texture_instances.append(instance)
            i += 1

        return cls(
            ops=ops,
            is_scalable=handle.is_scalable,
            source_coordinate_system=CoordinateSystem[
                handle.source_coordinate_system_name
            ],
            source_dimensions=handle.source_dimensions,
            time_estimate=handle.time_estimate,
            vertex_data=vertex_data,
            texture_instances=texture_instances,
        )
