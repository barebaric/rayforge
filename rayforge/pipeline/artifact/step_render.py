from __future__ import annotations
import numpy as np
from typing import Optional, List, Dict, Any, Type
from .base import BaseArtifact, VertexData, TextureInstance, TextureData
from .handle import BaseArtifactHandle


class StepRenderArtifactHandle(BaseArtifactHandle):
    """A handle for a StepRenderArtifact."""

    pass


class StepRenderArtifact(BaseArtifact):
    """
    Represents a lightweight artifact for rendering a Step.
    Contains only visual data (vertices, textures) and is designed for
    fast transfer and consumption by the UI.
    """

    def __init__(
        self,
        vertex_data: Optional[VertexData] = None,
        texture_instances: Optional[List[TextureInstance]] = None,
    ):
        self.vertex_data: Optional[VertexData] = vertex_data
        self.texture_instances: List[TextureInstance] = (
            texture_instances if texture_instances is not None else []
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result: Dict[str, Any] = {}
        if self.vertex_data:
            result["vertex_data"] = self.vertex_data.to_dict()
        result["texture_instances"] = [
            ti.to_dict() for ti in self.texture_instances
        ]
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepRenderArtifact":
        """Creates an artifact from a dictionary."""
        common_args = {}
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
    ) -> StepRenderArtifactHandle:
        """Creates the appropriate, typed handle for this artifact."""
        return StepRenderArtifactHandle(
            shm_name=shm_name,
            handle_class_name=StepRenderArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            array_metadata=array_metadata,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        """
        Gets a dictionary of all NumPy arrays that need to be stored in
        shared memory for this artifact.
        """
        arrays: Dict[str, np.ndarray] = {}  # No ops
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
        cls: Type[StepRenderArtifact],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> StepRenderArtifact:
        """
        Reconstructs an artifact instance from its handle and a dictionary of
        NumPy array views from shared memory.
        """
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
        while (
            f"texture_data_{i}" in arrays
            and f"texture_transform_{i}" in arrays
        ):
            tex_data_arr = arrays[f"texture_data_{i}"]
            transform_arr = arrays[f"texture_transform_{i}"]
            h, w = tex_data_arr.shape
            dims = (float(w), float(h))
            texture_data = TextureData(
                power_texture_data=tex_data_arr.copy(),
                dimensions_mm=dims,
                position_mm=(0, 0),
            )
            instance = TextureInstance(
                texture_data=texture_data,
                world_transform=transform_arr.copy(),
            )
            texture_instances.append(instance)
            i += 1
        return cls(
            vertex_data=vertex_data,
            texture_instances=texture_instances,
        )
