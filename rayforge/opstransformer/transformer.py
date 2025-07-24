from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional
from ..models.ops import Ops
from ..tasker import BaseExecutionContext


class OpsTransformer(ABC):
    """
    Transforms an Ops object in-place.
    Examples may include:

    - Applying travel path optimizations
    - Applying arc welding
    """
    def __init__(self, enabled: bool = True, **kwargs):
        self._enabled = enabled

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, enabled: bool) -> None:
        self._enabled = enabled

    @property
    @abstractmethod
    def label(self) -> str:
        """A short label for the transformation, used in UI."""
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        """A brief one-line description of the transformation."""
        pass

    @abstractmethod
    def run(
        self,
        ops: Ops,
        context: Optional[BaseExecutionContext] = None
    ) -> None:
        """
        Runs the transformation.

        Args:
            ops: The Ops object to transform in-place.
            context: Used for progress and cancellation hooks.
        """
        pass
