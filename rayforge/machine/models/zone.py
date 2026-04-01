import uuid
from enum import Enum
from gettext import gettext as _
from typing import Any, Dict

from blinker import Signal


class ZoneShape(Enum):
    RECT = "rect"
    BOX = "box"
    CYLINDER = "cylinder"


class Zone:
    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name: str = _("No-Go Zone")
        self.shape: ZoneShape = ZoneShape.RECT
        self.params: Dict[str, float] = {
            "x": 0.0,
            "y": 0.0,
            "w": 10.0,
            "h": 10.0,
        }
        self.enabled: bool = True
        self.changed = Signal()
        self.extra: Dict[str, Any] = {}

    def set_name(self, name: str):
        if self.name == name:
            return
        self.name = name
        self.changed.send(self)

    def set_shape(self, shape: ZoneShape):
        if self.shape == shape:
            return
        self.shape = shape
        if shape == ZoneShape.BOX:
            self.params.setdefault("z", 0.0)
            self.params.setdefault("d", 10.0)
            self.params.setdefault("w", self.params.get("w", 10.0))
            self.params.setdefault("h", self.params.get("h", 10.0))
        elif shape == ZoneShape.CYLINDER:
            self.params.setdefault("z", 0.0)
            self.params.setdefault("radius", 5.0)
            self.params.setdefault("height", 10.0)
        self.changed.send(self)

    def set_param(self, key: str, value: float):
        if self.params.get(key) == value:
            return
        self.params[key] = value
        self.changed.send(self)

    def set_enabled(self, enabled: bool):
        if self.enabled == enabled:
            return
        self.enabled = enabled
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uid": self.uid,
            "name": self.name,
            "shape": self.shape.value,
            "params": dict(self.params),
            "enabled": self.enabled,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Zone":
        known_keys = {
            "uid",
            "name",
            "shape",
            "params",
            "enabled",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        zone = cls()
        zone.uid = data.get("uid", str(uuid.uuid4()))
        zone.name = data.get("name", _("No-Go Zone"))
        zone.shape = ZoneShape(data.get("shape", "rect"))
        zone.params = dict(data.get("params", zone.params))
        zone.enabled = data.get("enabled", True)
        zone.extra = extra
        return zone

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("changed", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.changed = Signal()
