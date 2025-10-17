from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Type
import numpy as np
from ...core.ops import Ops
from ..coord import CoordinateSystem
from .base import BaseArtifact
from .handle import BaseArtifactHandle


@dataclass
class StepOpsArtifactHandle(BaseArtifactHandle):
    """A handle for a StepOpsArtifact."""

    pass


class StepOpsArtifact(BaseArtifact):
    """
    Represents an artifact containing only the final, transformed operations
    for a Step. This is consumed by the JobGeneratorStage.
    """

    def __init__(
        self,
        ops: Ops,
        time_estimate: Optional[float] = None,
    ):
        # is_scalable is always false for a step artifact because workpiece
        # scaling has already been applied.
        super().__init__(
            ops=ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            time_estimate=time_estimate,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the artifact to a dictionary for serialization."""
        return super().to_dict()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StepOpsArtifact":
        """Creates an artifact from a dictionary."""
        ops = Ops.from_dict(data["ops"])
        return cls(
            ops=ops,
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
        ops = Ops.from_numpy_arrays(arrays)
        return cls(
            ops=ops,
            time_estimate=handle.time_estimate,
        )
