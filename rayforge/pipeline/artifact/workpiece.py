from __future__ import annotations
import logging
import numpy as np
from typing import Optional, Tuple, Dict, Any, Type, TYPE_CHECKING, cast
from ..coord import CoordinateSystem
from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ...core.ops import Ops


class WorkPieceArtifactHandle(BaseArtifactHandle):
    """A handle for a WorkPieceArtifact, with specific metadata."""

    logger = logging.getLogger(__name__)

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
        self.logger.debug(
            f"WorkPieceArtifactHandle.__init__: "
            f"source_dimensions={self.source_dimensions}, "
            f"generation_size={self.generation_size}"
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPieceArtifactHandle":
        cls.logger.debug(
            f"WorkPieceArtifactHandle.from_dict: data.source_dimensions="
            f"{data.get('source_dimensions')}, "
            f"data.generation_size={data.get('generation_size')}"
        )
        handle = cast("WorkPieceArtifactHandle", super().from_dict(data))
        cls.logger.debug(
            f"WorkPieceArtifactHandle.from_dict: handle.source_dimensions="
            f"{handle.source_dimensions}, "
            f"handle.generation_size={handle.generation_size}"
        )
        return handle


class WorkPieceArtifact(BaseArtifact):
    """
    Represents an intermediate artifact produced during the pipeline,
    containing vertex and texture data for visualization.
    """

    logger = logging.getLogger(__name__)

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

        cls.logger.debug(
            f"WorkPieceArtifact.from_dict: data.source_dimensions="
            f"{data.get('source_dimensions')}, "
            f"data.generation_size={tuple(data['generation_size'])}"
        )
        ops = Ops.from_dict(data["ops"])
        artifact = cls(
            ops=ops,
            is_scalable=data["is_scalable"],
            source_coordinate_system=CoordinateSystem[
                data["source_coordinate_system"]
            ],
            source_dimensions=data.get("source_dimensions"),
            generation_size=tuple(data["generation_size"]),
        )
        cls.logger.debug(
            f"WorkPieceArtifact.from_dict: artifact.source_dimensions="
            f"{artifact.source_dimensions}, "
            f"artifact.generation_size={artifact.generation_size}"
        )
        return artifact

    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> WorkPieceArtifactHandle:
        self.logger.debug(
            f"WorkPieceArtifact.create_handle: "
            f"source_dimensions={self.source_dimensions}, "
            f"generation_size={self.generation_size}"
        )
        handle = WorkPieceArtifactHandle(
            shm_name=shm_name,
            handle_class_name=WorkPieceArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            is_scalable=self.is_scalable,
            source_coordinate_system_name=self.source_coordinate_system.name,
            source_dimensions=self.source_dimensions,
            array_metadata=array_metadata,
            generation_size=self.generation_size,
        )
        self.logger.debug(
            f"WorkPieceArtifact.create_handle: "
            f"handle.source_dimensions={handle.source_dimensions}, "
            f"handle.generation_size={handle.generation_size}"
        )
        return handle

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

        # Create a deep copy of the arrays to break the link to shared memory
        copied_arrays = {k: v.copy() for k, v in arrays.items()}
        ops = Ops.from_numpy_arrays(copied_arrays)

        return cls(
            ops=ops,
            is_scalable=handle.is_scalable,
            source_coordinate_system=CoordinateSystem[
                handle.source_coordinate_system_name
            ],
            source_dimensions=handle.source_dimensions,
            generation_size=handle.generation_size,
        )
