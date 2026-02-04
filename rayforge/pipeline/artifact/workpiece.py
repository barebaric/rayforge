from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Dict, Any, Type, TYPE_CHECKING
from ..coord import CoordinateSystem
from .base import BaseArtifact
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
    ):
        super().__init__()
        self.ops = ops
        self.is_scalable = is_scalable
        self.source_coordinate_system = source_coordinate_system
        self.source_dimensions = source_dimensions
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
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPieceArtifact":
        """Creates an artifact from a dictionary."""
        from ...core.ops import Ops

        ops = Ops.from_dict(data["ops"])
        return cls(
            ops=ops,
            is_scalable=data["is_scalable"],
            source_coordinate_system=CoordinateSystem[
                data["source_coordinate_system"]
            ],
            source_dimensions=data.get("source_dimensions"),
            generation_size=tuple(data["generation_size"]),
        )

    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> WorkPieceArtifactHandle:
        return WorkPieceArtifactHandle(
            shm_name=shm_name,
            handle_class_name=WorkPieceArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            is_scalable=self.is_scalable,
            source_coordinate_system_name=self.source_coordinate_system.name,
            source_dimensions=self.source_dimensions,
            array_metadata=array_metadata,
            generation_size=self.generation_size,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        arrays = self.ops.to_numpy_arrays()
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
        return cls(
            ops=ops,
            is_scalable=handle.is_scalable,
            source_coordinate_system=CoordinateSystem[
                handle.source_coordinate_system_name
            ],
            source_dimensions=handle.source_dimensions,
            generation_size=handle.generation_size,
        )
