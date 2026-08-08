from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from raygeo.ops import Ops


class StepOpsArtifactHandle(BaseArtifactHandle):
    def __init__(
        self,
        key: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
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


class StepOpsArtifact(BaseArtifact):
    """
    Represents an artifact containing only the final, transformed operations
    for a Step. This is consumed by the JobPipelineStage.
    """

    def __init__(
        self,
        ops: Ops,
        generation_id: int,
    ):
        super().__init__()
        self.ops = ops
        self.generation_id = generation_id

    def build_handle(self, key: str) -> StepOpsArtifactHandle:
        return StepOpsArtifactHandle(
            key=key,
            handle_class_name=StepOpsArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
        )
