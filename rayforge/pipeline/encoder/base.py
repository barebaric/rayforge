from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from raygeo.ops import Ops


@dataclass
class MachineCodeOpMap:
    """
    A container for a bidirectional mapping between Ops command indices and
    Machine language (e.g. G-code) line numbers.

    Both fields are flat lists rather than dicts: op indices and line
    numbers are dense integer ranges, so indexing replaces hashing.

    Attributes:
        op_to_machine_code: One ``(start_line, line_count)`` span per Ops
                     command index. An empty span means the command
                     produced no G-code.
        machine_code_to_op: Maps a G-code line number back to the Ops
                     command index that generated it; ``-1`` means the
                     line has no owning op.
    """

    op_to_machine_code: list[tuple[int, int]] = field(default_factory=list)
    machine_code_to_op: list[int] = field(default_factory=list)

    def op_for_line(self, line_idx: int) -> int | None:
        """Op index for a machine-code line, or ``None`` if the line is
        out of range or has no owning op."""
        if 0 <= line_idx < len(self.machine_code_to_op):
            mapped = self.machine_code_to_op[line_idx]
            if mapped != -1:
                return mapped
        return None


@dataclass
class EncodedOutput:
    """
    Base class for encoder output.

    Attributes:
        text: Human-readable machine code representation for UI display.
        op_map: Bidirectional mapping between ops indices and line numbers.
        driver_data: Optional driver-specific data (e.g., binary for Ruida).
    """

    text: str
    op_map: MachineCodeOpMap
    driver_data: dict[str, Any] = field(default_factory=dict)


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
