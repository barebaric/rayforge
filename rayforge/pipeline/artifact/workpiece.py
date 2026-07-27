from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, cast

from ..coord import CoordinateSystem
from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from raygeo.ops import Ops


class WorkPieceArtifactHandle(BaseArtifactHandle):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        is_scalable: bool,
        source_coordinate_system_name: str,
        generation_size: Tuple[float, float],
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        source_dimensions: Optional[Tuple[float, float]] = None,
        array_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__(
            key=key,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            generation_id=generation_id,
            array_metadata=array_metadata,
        )
        self.is_scalable = is_scalable
        self.source_coordinate_system_name = source_coordinate_system_name
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size

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
        generation_id: int,
        source_dimensions: Optional[Tuple[float, float]] = None,
    ):
        super().__init__()
        self.ops = ops
        self.is_scalable = is_scalable
        self.source_coordinate_system = source_coordinate_system
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size
        self.generation_id = generation_id

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        result = {
            "ops": self.ops.to_dict(),
            "is_scalable": self.is_scalable,
            "source_coordinate_system": self.source_coordinate_system.name,
            "source_dimensions": self.source_dimensions,
            "generation_size": self.generation_size,
            "generation_id": self.generation_id,
        }
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkPieceArtifact":
        """Creates an artifact from a dictionary."""
        from raygeo.ops import Ops

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
            generation_id=data["generation_id"],
        )
        cls.logger.debug(
            f"WorkPieceArtifact.from_dict: artifact.source_dimensions="
            f"{artifact.source_dimensions}, "
            f"artifact.generation_size={artifact.generation_size}"
        )
        return artifact

    def build_handle(self, key: str) -> WorkPieceArtifactHandle:
        return WorkPieceArtifactHandle(
            key=key,
            handle_class_name=WorkPieceArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
            is_scalable=self.is_scalable,
            source_coordinate_system_name=self.source_coordinate_system.name,
            source_dimensions=self.source_dimensions,
            generation_size=self.generation_size,
        )
