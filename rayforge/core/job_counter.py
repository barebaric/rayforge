"""
The document's job counter.

A document carries one :class:`JobCounter`. Its value is passed as
``counter`` to the ``{...}`` templates of every geometry provider (for
example the text boxes of a sketch). Each workpiece adds its own
*variable offset*, so copies of the same design can show consecutive
counter values.

The counter arithmetic and serialization live in raygeo; this wrapper
adds the document-facing signal and the template values.
"""

from __future__ import annotations

from typing import Any

from blinker import Signal
from raygeo.template import CounterSettings

COUNTER_NAME = "counter"


class JobCounter:
    """
    The document's job counter.

    Emits :attr:`changed` whenever a value changes; the document then
    regenerates the geometry that shows counter values.
    """

    def __init__(self):
        self.changed = Signal()
        self._counter = CounterSettings()

    @property
    def counter(self) -> CounterSettings:
        return self._counter

    @counter.setter
    def counter(self, value: CounterSettings) -> None:
        if value == self._counter:
            return
        self._counter = value
        self.changed.send(self)

    @property
    def names(self) -> set[str]:
        """The template names this object provides."""
        return {COUNTER_NAME}

    def template_values(self, offset: int = 0) -> dict[str, Any]:
        """The template values for a copy with the given offset."""
        return {COUNTER_NAME: self._counter.value_at(offset)}

    def to_dict(self) -> dict[str, Any]:
        return {"counter": self._counter.to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> JobCounter:
        instance = cls()
        if data:
            counter = data.get("counter")
            if isinstance(counter, dict):
                instance._counter = CounterSettings.from_dict(counter)
        return instance
