from abc import ABC, abstractmethod
from typing import Any, Dict, List
from dataclasses import dataclass, field
from ...core.ops import Ops


@dataclass
class MachineCodeOpMap:
    """
    A container for a bidirectional mapping between Ops command indices and
    Machine language (e.g. G-code) line numbers.

    Attributes:
        op_to_machine_code: Maps an Ops command index to a list of G-code line
                     numbers it generated. An empty list means the command
                     produced no G-code.
        machine_code_to_op: Maps a G-code line number back to the Ops command
                     index that generated it.
    """

    op_to_machine_code: Dict[int, List[int]] = field(default_factory=dict)
    machine_code_to_op: Dict[int, int] = field(default_factory=dict)


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
