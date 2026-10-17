"""
Document-level variables for text templates.

A document carries a :class:`VariableText` instance whose values are
injected into the ``{...}`` templates of every geometry provider (for
example the text boxes of a sketch). Each workpiece adds its own
*variable offset*, so copies of the same design can show consecutive
serial numbers.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import Any

from blinker import Signal

SERIAL_NAME = "serial"

_PLACEHOLDER_RE = re.compile(r"\{([^{}]+)\}")
_IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class SerialSettings:
    """
    An immutable serial counter.

    Attributes:
        current: The value the next job starts with.
        start: The value :meth:`reset` returns to, and where the
            counter wraps to after ``end``.
        step: How much the value grows per copy and per job.
        end: Optional last value. Values past it wrap to ``start``.
        auto_advance: Whether a finished job moves ``current`` on.
        advance_on_export: Whether exporting G-code moves ``current``
            on as well, for jobs that run from a memory card.
    """

    current: int = 1
    start: int = 1
    step: int = 1
    end: int | None = None
    auto_advance: bool = True
    advance_on_export: bool = False

    def _range_length(self) -> int | None:
        """Number of values in ``start..end``, or None if unbounded."""
        if self.end is None or self.step <= 0 or self.end < self.start:
            return None
        return (self.end - self.start) // self.step + 1

    def value_at(self, offset: int) -> int:
        """The value shown by a copy that is *offset* copies further."""
        value = self.current + offset * self.step
        length = self._range_length()
        if length is None:
            return value
        index = (value - self.start) // self.step
        return self.start + (index % length) * self.step

    def values_for(self, copies: int) -> list[int]:
        """The values of *copies* consecutive copies."""
        return [self.value_at(i) for i in range(max(copies, 0))]

    def advanced(self, copies: int = 1) -> SerialSettings:
        """A counter moved on (or back, if negative) by *copies*."""
        return replace(self, current=self.value_at(copies))

    def reset(self) -> SerialSettings:
        """A counter whose current value is back at ``start``."""
        return replace(self, current=self.start)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current,
            "start": self.start,
            "step": self.step,
            "end": self.end,
            "auto_advance": self.auto_advance,
            "advance_on_export": self.advance_on_export,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SerialSettings:
        default = cls()
        start = int(data.get("start", default.start))
        end = data.get("end")
        if end is not None:
            end = int(end)
            if end < start:
                end = None
        return cls(
            current=int(data.get("current", default.current)),
            start=start,
            step=int(data.get("step", default.step)),
            end=end,
            auto_advance=bool(data.get("auto_advance", default.auto_advance)),
            advance_on_export=bool(
                data.get("advance_on_export", default.advance_on_export)
            ),
        )


class VariableText:
    """
    The variables a document offers to text templates.

    Emits :attr:`changed` whenever a value changes, so workpieces whose
    templates reference a variable can regenerate.
    """

    def __init__(self):
        self.changed = Signal()
        self._serial = SerialSettings()

    @property
    def serial(self) -> SerialSettings:
        return self._serial

    @serial.setter
    def serial(self, value: SerialSettings) -> None:
        if value == self._serial:
            return
        self._serial = value
        self.changed.send(self)

    @property
    def names(self) -> set[str]:
        """The template names this object provides."""
        return {SERIAL_NAME}

    def template_values(self, offset: int = 0) -> dict[str, Any]:
        """The template values for a copy with the given offset."""
        return {SERIAL_NAME: self._serial.value_at(offset)}

    def to_dict(self) -> dict[str, Any]:
        return {"serial": self._serial.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> VariableText:
        instance = cls()
        if data:
            serial = data.get("serial")
            if isinstance(serial, dict):
                instance._serial = SerialSettings.from_dict(serial)
        return instance


def _names_in_expression(expression: str) -> set[str]:
    """Names an expression reads, ignoring string literals."""
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError:
        return set(_IDENTIFIER_RE.findall(expression))
    return {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}


def referenced_template_names(templates: Iterable[str]) -> set[str]:
    """
    Collects the names used inside the ``{...}`` placeholders of
    *templates*. A format spec after the first colon is ignored.
    """
    names: set[str] = set()
    for template in templates:
        for match in _PLACEHOLDER_RE.finditer(template or ""):
            expression, _sep, _spec = match.group(1).partition(":")
            names |= _names_in_expression(expression)
    return names
