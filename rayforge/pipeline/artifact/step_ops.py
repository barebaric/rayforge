from __future__ import annotations
from typing import Optional, Dict, Any, Type, TYPE_CHECKING
import numpy as np
from .base import BaseArtifact
from .handle import BaseArtifactHandle

if TYPE_CHECKING:
    from ...core.ops import Ops


class StepOpsArtifactHandle(BaseArtifactHandle):
    """A handle for a StepOpsArtifact."""

    def __init__(
        self,
        time_estimate: Optional[float],
        shm_name: str,
        handle_class_name: str,
        artifact_type_name: str,
        generation_id: int,
        array_metadata: Optional[Dict[str, Any]] = None,
        **_kwargs,
    ):
        super().__init__(
            shm_name=shm_name,
            handle_class_name=handle_class_name,
            artifact_type_name=artifact_type_name,
            generation_id=generation_id,
            array_metadata=array_metadata,
        )
        self.time_estimate = time_estimate


class StepOpsArtifact(BaseArtifact):
    """
    Represents an artifact containing only the final, transformed operations
    for a Step. This is consumed by the JobPipelineStage.
    """

    def __init__(
        self,
        ops: "Ops",
        generation_id: int,
        time_estimate: Optional[float] = None,
    ):
        super().__init__()
        self.ops = ops
        self.generation_id = generation_id
        self.time_estimate = time_estimate

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        return {
            "ops": self.ops.to_dict(),
            "generation_id": self.generation_id,
            "time_estimate": self.time_estimate,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepOpsArtifact":
        """Creates an artifact from a dictionary."""
        from ...core.ops import Ops

        ops = Ops.from_dict(data["ops"])
        return cls(
            ops=ops,
            generation_id=data["generation_id"],
            time_estimate=data.get("time_estimate"),
        )

    def create_handle(
        self,
        shm_name: str,
        array_metadata: Dict[str, Dict[str, Any]],
    ) -> StepOpsArtifactHandle:
        """Creates the appropriate, typed handle for this artifact."""
        return StepOpsArtifactHandle(
            shm_name=shm_name,
            handle_class_name=StepOpsArtifactHandle.__name__,
            artifact_type_name=self.__class__.__name__,
            generation_id=self.generation_id,
            time_estimate=self.time_estimate,
            array_metadata=array_metadata,
        )

    def get_arrays_for_storage(self) -> Dict[str, np.ndarray]:
        """
        Gets a dictionary of all NumPy arrays that need to be stored in
        shared memory for this artifact.
        """
        return self.ops.to_numpy_arrays()

    @classmethod
    def from_storage(
        cls: Type[StepOpsArtifact],
        handle: BaseArtifactHandle,
        arrays: Dict[str, np.ndarray],
    ) -> StepOpsArtifact:
        """
        Reconstructs an artifact instance from its handle and a dictionary of
        NumPy array views from shared memory.
        """
        if not isinstance(handle, StepOpsArtifactHandle):
            raise TypeError("StepOpsArtifact requires a StepOpsArtifactHandle")
        from ...core.ops import Ops

        # Create a deep copy of the arrays to break the link to shared memory
        copied_arrays = {k: v.copy() for k, v in arrays.items()}
        ops = Ops.from_numpy_arrays(copied_arrays)
        return cls(
            ops=ops,
            generation_id=handle.generation_id,
            time_estimate=handle.time_estimate,
        )
