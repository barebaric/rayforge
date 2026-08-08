from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from raygeo.ops import Ops


class WorkPieceArtifactHandle(BaseArtifactHandle):
    logger = logging.getLogger(__name__)

    def __init__(
        self,
        is_scalable: bool,
        generation_size: tuple[float, float],
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        source_dimensions: tuple[float, float] | None = None,
        array_metadata: dict[str, Any] | None = None,
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
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size


class WorkPieceArtifact(BaseArtifact):
    """
    Represents an intermediate artifact produced during the pipeline,
    containing vertex and texture data for visualization.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        ops: Ops,
        is_scalable: bool,
        generation_size: tuple[float, float],
        generation_id: int,
        source_dimensions: tuple[float, float] | None = None,
    ):
        super().__init__()
        self.ops = ops
        self.is_scalable = is_scalable
        self.source_dimensions = source_dimensions
        self.generation_size = generation_size
        self.generation_id = generation_id

    def build_handle(self, key: str) -> WorkPieceArtifactHandle:
        return WorkPieceArtifactHandle(
            key=key,
            handle_class_name=WorkPieceArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
            is_scalable=self.is_scalable,
            source_dimensions=self.source_dimensions,
            generation_size=self.generation_size,
        )
