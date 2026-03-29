from typing import Dict, Any
from blinker import Signal
import uuid
from gettext import gettext as _

from ..driver.driver import Axis


class RotaryModule:
    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name: str = _("Rotary Module")
        self.axis: Axis = Axis.A
        self.x: float = 0.0
        self.y: float = 0.0
        self.z: float = 0.0
        self.length: float = 200.0
        self.chuck_diameter: float = 40.0
        self.tailstock_diameter: float = 20.0
        self.max_height: float = 60.0
        self.default_diameter: float = 25.0
        self.changed = Signal()
        self.extra: Dict[str, Any] = {}

    def set_name(self, name: str):
        self.name = name
        self.changed.send(self)

    def set_axis(self, axis: Axis):
        axis.assert_single_axis()
        if self.axis == axis:
            return
        self.axis = axis
        self.changed.send(self)

    def set_position(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z
        self.changed.send(self)

    def set_length(self, length: float):
        self.length = length
        self.changed.send(self)

    def set_chuck_diameter(self, diameter: float):
        self.chuck_diameter = diameter
        self.changed.send(self)

    def set_tailstock_diameter(self, diameter: float):
        self.tailstock_diameter = diameter
        self.changed.send(self)

    def set_max_height(self, height: float):
        self.max_height = height
        self.changed.send(self)

    def set_default_diameter(self, diameter: float):
        self.default_diameter = diameter
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uid": self.uid,
            "name": self.name,
            "axis": self.axis.name,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "length": self.length,
            "chuck_diameter": self.chuck_diameter,
            "tailstock_diameter": self.tailstock_diameter,
            "max_height": self.max_height,
            "default_diameter": self.default_diameter,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RotaryModule":
        known_keys = {
            "uid",
            "name",
            "axis",
            "x",
            "y",
            "z",
            "length",
            "chuck_diameter",
            "tailstock_diameter",
            "max_height",
            "default_diameter",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        rm = cls()
        rm.uid = data.get("uid", str(uuid.uuid4()))
        rm.name = data.get("name", _("Rotary Module"))
        rm.axis = Axis[data.get("axis", "A")]
        rm.x = data.get("x", 0.0)
        rm.y = data.get("y", 0.0)
        rm.z = data.get("z", 0.0)
        rm.length = data.get("length", 200.0)
        rm.chuck_diameter = data.get("chuck_diameter", 40.0)
        rm.tailstock_diameter = data.get("tailstock_diameter", 20.0)
        rm.max_height = data.get("max_height", 60.0)
        rm.default_diameter = data.get("default_diameter", 25.0)
        rm.extra = extra
        return rm

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("changed", None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.changed = Signal()
