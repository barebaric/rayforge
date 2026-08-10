from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from raygeo.ops import Ops


class MachineCodeOpMap:
    """
    A bidirectional mapping between Ops command indices and Machine
    language (e.g. G-code) line numbers.

    Data is stored as compact numpy ``int32`` arrays (4 bytes/element)
    instead of Python lists of ints/tuples (~28-56 bytes/element).  When
    constructed from raygeo's ``bytearray`` getters, the arrays are
    zero-copy ``np.frombuffer`` views over the bytearray, so no
    per-element Python objects are ever materialized
    (1.6 M ops: 196 MB list-of-tuples → 13 MB).

    Access is through the :meth:`span_for_op` and :meth:`op_for_line`
    methods (plus :attr:`op_count` / :attr:`line_count`) rather than
    raw list indexing.
    """

    def __init__(
        self,
        op_to_machine_code: bytearray | None = None,
        machine_code_to_op: bytearray | None = None,
    ) -> None:
        self._spans: np.ndarray = np.empty((0, 2), dtype=np.int32)
        self._line_to_op: np.ndarray = np.empty(0, dtype=np.int32)
        self._spans_bytes: bytearray | None = None
        self._line_to_op_bytes: bytearray | None = None
        if op_to_machine_code is not None:
            self._set_spans(op_to_machine_code)
        if machine_code_to_op is not None:
            self._set_line_to_op(machine_code_to_op)

    @classmethod
    def from_lists(
        cls,
        op_to_machine_code: list[tuple[int, int]],
        machine_code_to_op: list[int],
    ) -> "MachineCodeOpMap":
        """Build a map from the legacy list-of-tuples / list-of-ints
        representation (used by encoders that construct the spans
        incrementally, e.g. the Ruida encoder)."""
        spans_bytes = bytearray()
        for start, count in op_to_machine_code:
            spans_bytes.extend(start.to_bytes(4, "little", signed=True))
            spans_bytes.extend(count.to_bytes(4, "little", signed=True))
        line_to_op_bytes = bytearray()
        for v in machine_code_to_op:
            line_to_op_bytes.extend(v.to_bytes(4, "little", signed=True))
        return cls(
            op_to_machine_code=spans_bytes,
            machine_code_to_op=line_to_op_bytes,
        )

    def _set_spans(self, value: bytearray) -> None:
        if len(value) % 8 != 0:
            raise ValueError(
                "op_to_machine_code bytearray length must be a multiple of 8"
            )
        self._spans = np.frombuffer(value, dtype=np.int32).reshape(-1, 2)
        self._spans_bytes = value

    def _set_line_to_op(self, value: bytearray) -> None:
        if len(value) % 4 != 0:
            raise ValueError(
                "machine_code_to_op bytearray length must be a multiple of 4"
            )
        self._line_to_op = np.frombuffer(value, dtype=np.int32)
        self._line_to_op_bytes = value

    @property
    def op_count(self) -> int:
        """Number of Ops commands in the map."""
        return self._spans.shape[0]

    @property
    def line_count(self) -> int:
        """Number of G-code lines in the map."""
        return len(self._line_to_op)

    def span_for_op(self, op_index: int) -> tuple[int, int]:
        """Return ``(start_line, line_count)`` for *op_index*.

        A ``line_count`` of zero means the op produced no G-code.
        Raises :class:`IndexError` when *op_index* is out of range.
        """
        if not 0 <= op_index < self._spans.shape[0]:
            raise IndexError(f"op index out of range: {op_index}")
        start, count = self._spans[op_index]
        return (int(start), int(count))

    def op_for_line(self, line_idx: int) -> int | None:
        """Op index for a machine-code line, or ``None`` if the line is
        out of range or has no owning op."""
        if 0 <= line_idx < len(self._line_to_op):
            mapped = self._line_to_op[line_idx]
            if mapped != -1:
                return int(mapped)
        return None

    @property
    def op_to_machine_code_bytes(self) -> bytearray:
        """The interleaved ``(start, count)`` i32 payload as a
        bytearray, for passing directly to raygeo's
        ``EncodeOutput.MachineCode`` constructor."""
        if self._spans_bytes is None:
            self._spans_bytes = bytearray(self._spans.tobytes())
        return self._spans_bytes

    @property
    def machine_code_to_op_bytes(self) -> bytearray:
        """The line→op i32 payload as a bytearray, for passing directly
        to raygeo's ``EncodeOutput.MachineCode`` constructor."""
        if self._line_to_op_bytes is None:
            self._line_to_op_bytes = bytearray(self._line_to_op.tobytes())
        return self._line_to_op_bytes


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
