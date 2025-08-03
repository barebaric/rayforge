from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING
from ..models.ops import Ops
if TYPE_CHECKING:
    from ..models.machine import Machine


class OpsEncoder(ABC):
    """
    Transforms an Ops object into something else.
    Examples:

    - Ops to image (a cairo surface)
    - Ops to a G-code string
    """
    @abstractmethod
    def encode(self, pos: Ops, machine: "Machine", *args, **kwargs) -> Any:
        pass
