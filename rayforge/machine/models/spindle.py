from gettext import gettext as _
from typing import Any, Dict, Iterable, Tuple

from raygeo.ops.state import CoolantMode

from ...core.capability import MachineCapability
from .head import _HEAD_SERIALIZED_KEYS, Head

_COOLANT_MODE_BY_NAME = {
    mode.name: mode for mode in (CoolantMode.FLOOD, CoolantMode.MIST)
}


def _normalize_cooling_methods(
    methods: Iterable[CoolantMode],
) -> Tuple[CoolantMode, ...]:
    """Keep only real coolant methods, in a stable order.

    ``CoolantMode.OFF`` is always available and therefore not a
    supported method; it is stripped along with any non-``CoolantMode``
    values.
    """
    return tuple(
        m
        for m in methods
        if isinstance(m, CoolantMode) and m is not CoolantMode.OFF
    )


class SpindleHead(Head):
    """A motorized spindle head, implying the MILL machine capability."""

    HEAD_TYPE: str = "SpindleHead"

    def __init__(self):
        super().__init__()
        self.name: str = _("Spindle Head")
        self.max_rpm: int = 20000
        self.min_rpm: int = 1000
        self.cooling_methods: Tuple[CoolantMode, ...] = ()

    @property
    def machine_capability(self) -> MachineCapability:
        return MachineCapability.MILL

    def set_max_rpm(self, rpm: int):
        if self.max_rpm == rpm:
            return
        self.max_rpm = int(rpm)
        if self.min_rpm > self.max_rpm:
            self.min_rpm = self.max_rpm
        self.changed.send(self)

    def set_min_rpm(self, rpm: int):
        if self.min_rpm == rpm:
            return
        self.min_rpm = int(rpm)
        if self.max_rpm < self.min_rpm:
            self.max_rpm = self.min_rpm
        self.changed.send(self)

    def set_cooling_methods(self, methods: Iterable[CoolantMode]):
        """Sets the coolant methods this head supports."""
        normalized = _normalize_cooling_methods(methods)
        if self.cooling_methods == normalized:
            return
        self.cooling_methods = normalized
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "max_rpm": self.max_rpm,
                "min_rpm": self.min_rpm,
                "cooling_methods": [m.name for m in self.cooling_methods],
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpindleHead":
        known_keys = _HEAD_SERIALIZED_KEYS | {
            "max_rpm",
            "min_rpm",
            "cooling_methods",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        sh = super().from_dict(data)
        sh.max_rpm = data.get("max_rpm", sh.max_rpm)
        sh.min_rpm = data.get("min_rpm", sh.min_rpm)
        raw_methods = data.get("cooling_methods", ())
        methods: Tuple[CoolantMode, ...] = ()
        if isinstance(raw_methods, (list, tuple)):
            parsed = [
                _COOLANT_MODE_BY_NAME[name]
                for name in raw_methods
                if name in _COOLANT_MODE_BY_NAME
            ]
            methods = _normalize_cooling_methods(parsed)
        sh.cooling_methods = methods
        sh.extra = extra
        return sh
