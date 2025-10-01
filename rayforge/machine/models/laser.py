from typing import Dict, Any, Tuple
from blinker import Signal
import uuid


class Laser:
    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name: str = _("Laser Head")
        self.tool_number: int = 0
        self.max_power: int = 1000  # Max power (0-1000 for GRBL)
        self.frame_power: int = 0  # 0 = framing not supported
        self.spot_size_mm: Tuple[float, float] = 0.1, 0.1  # millimeters
        self.changed = Signal()

    def set_name(self, name: str):
        self.name = name
        self.changed.send(self)

    def set_tool_number(self, tool_number: int):
        self.tool_number = tool_number
        self.changed.send(self)

    def set_max_power(self, power):
        self.max_power = power
        self.changed.send(self)

    def set_frame_power(self, power):
        self.frame_power = power
        self.changed.send(self)

    def set_spot_size(self, spot_size_x_mm, spot_size_y_mm):
        self.spot_size_mm = spot_size_x_mm, spot_size_y_mm
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "uid": self.uid,
            "name": self.name,
            "tool_number": self.tool_number,
            "max_power": self.max_power,
            "frame_power": self.frame_power,
            "spot_size_mm": self.spot_size_mm,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Laser":
        lh = cls()
        lh.uid = data.get("uid", str(uuid.uuid4()))
        lh.name = data.get("name", _("Laser Head"))
        lh.tool_number = data.get("tool_number", lh.tool_number)
        lh.max_power = data.get("max_power", lh.max_power)
        lh.frame_power = data.get("frame_power", lh.frame_power)
        lh.spot_size_mm = data.get("spot_size_mm", lh.spot_size_mm)
        return lh

    def __getstate__(self):
        """Prepare the object for pickling. Removes unpickleable Signal."""
        state = self.__dict__.copy()
        # The 'changed' signal is not pickleable, so we remove it.
        state.pop("changed", None)
        return state

    def __setstate__(self, state):
        """Restore the object after unpickling. Recreates the Signal."""
        self.__dict__.update(state)
        # Re-create the 'changed' signal that was removed during pickling.
        self.changed = Signal()
