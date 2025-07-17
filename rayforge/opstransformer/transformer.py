from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from ..models.ops import Ops
from ..task import ExecutionContext


class OpsTransformer(ABC):
    """
    Transforms an Ops object in-place.
    Examples may include:

    - Applying travel path optimizations
    - Applying arc welding
    """
    @abstractmethod
    def run(
        self,
        ops: Ops,
        context: Optional[ExecutionContext] = None
    ) -> None:
        """
        Runs the transformation.

        Args:
            ops: The Ops object to transform in-place.
            context: Used for progress and cancellation hooks.
        """
        pass
