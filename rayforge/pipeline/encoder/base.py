from abc import ABC, abstractmethod
from typing import Any
from ...core.ops import Ops


class OpsEncoder(ABC):
    """
    Transforms an Ops object into something else.
    Examples:

    - Ops to image (a cairo surface)
    - Ops to a G-code string
    """
    @abstractmethod
    def encode(self, ops: Ops, *args, **kwargs) -> Any:
        pass
