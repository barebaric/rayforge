from typing import Dict, Any, Tuple
from blinker import Signal
import uuid
from gettext import gettext as _


class Laser:
    def __init__(self):
        self.uid: str = str(uuid.uuid4())
        self.name: str = _("Laser Head")
        self.tool_number: int = 0
        self.max_power: int = 1000  # Max power (0-1000 for GRBL)
        self.frame_power_percent: float = 0  # in percent (0-1.0)
        self.focus_power_percent: float = 0  # in percent (0-1.0)
        self.frame_speed: int = 0  # mm/min, 0 = use machine max travel speed
        self.frame_repeat_count: int = 20
        self.frame_corner_pause: float = 0  # seconds
        self.spot_size_mm: Tuple[float, float] = 0.1, 0.1  # millimeters
        self.cut_color: str = "#ff00ff"  # Magenta for cut
        self.raster_color: str = "#000000"  # Black for raster
        self.changed = Signal()
        self.extra: Dict[str, Any] = {}

    def set_name(self, name: str):
        self.name = name
        self.changed.send(self)

    def set_tool_number(self, tool_number: int):
        self.tool_number = tool_number
        self.changed.send(self)

    def set_max_power(self, power):
        self.max_power = power
        self.changed.send(self)

    def set_frame_power(self, power: float):
        """Set frame power in percent (0.0 - 1.0)."""
        self.frame_power_percent = power
        self.changed.send(self)

    def set_focus_power(self, power):
        """Set focus power in percent (0.0 - 1.0)."""
        self.focus_power_percent = power
        self.changed.send(self)

    def set_frame_speed(self, speed: int):
        self.frame_speed = speed
        self.changed.send(self)

    def set_frame_repeat_count(self, count: int):
        self.frame_repeat_count = count
        self.changed.send(self)

    def set_frame_corner_pause(self, duration: float):
        self.frame_corner_pause = duration
        self.changed.send(self)

    def _gcode_to_percent(self, gcode_value) -> float:
        """Convert gcode power value (0-max_power) to percentage (0-100)."""
        if self.max_power <= 0:
            return 0
        return gcode_value / self.max_power

    def _percent_to_gcode(self, percent):
        """Convert percentage (0-100) to gcode power value (0-max_power)."""
        return int(round((percent / 100) * self.max_power))

    def set_spot_size(self, spot_size_x_mm, spot_size_y_mm):
        self.spot_size_mm = spot_size_x_mm, spot_size_y_mm
        self.changed.send(self)

    def set_cut_color(self, color: str):
        self.cut_color = color
        self.changed.send(self)

    def set_raster_color(self, color: str):
        self.raster_color = color
        self.changed.send(self)

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "uid": self.uid,
            "name": self.name,
            "tool_number": self.tool_number,
            "max_power": self.max_power,
            "frame_power_percent": self.frame_power_percent * 100,
            "focus_power_percent": self.focus_power_percent * 100,
            "frame_speed": self.frame_speed,
            "frame_repeat_count": self.frame_repeat_count,
            "frame_corner_pause": self.frame_corner_pause,
            "spot_size_mm": self.spot_size_mm,
            "cut_color": self.cut_color,
            "raster_color": self.raster_color,
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Laser":
        known_keys = {
            "uid",
            "name",
            "tool_number",
            "max_power",
            "frame_power_percent",
            "focus_power_percent",
            "frame_power",
            "focus_power",
            "frame_speed",
            "frame_repeat_count",
            "frame_corner_pause",
            "spot_size_mm",
            "cut_color",
            "raster_color",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        lh = cls()
        lh.uid = data.get("uid", str(uuid.uuid4()))
        lh.name = data.get("name", _("Laser Head"))
        lh.tool_number = data.get("tool_number", lh.tool_number)
        lh.max_power = data.get("max_power", lh.max_power)

        # Handle backward compatibility for frame power
        if "frame_power_percent" in data:
            lh.frame_power_percent = data["frame_power_percent"] / 100.0
        else:
            # Old format: convert from gcode units to percentage
            frame_power = data.get("frame_power", 0)
            lh.frame_power_percent = frame_power / lh.max_power

        # Handle backward compatibility for focus power
        if "focus_power_percent" in data:
            lh.focus_power_percent = data["focus_power_percent"] / 100.0
        else:
            # Old format: convert from gcode units to percentage
            focus_power = data.get("focus_power", 0)
            lh.focus_power_percent = focus_power / lh.max_power

        lh.spot_size_mm = data.get("spot_size_mm", lh.spot_size_mm)
        lh.cut_color = data.get("cut_color", lh.cut_color)
        lh.raster_color = data.get("raster_color", lh.raster_color)
        lh.frame_speed = data.get("frame_speed", lh.frame_speed)
        lh.frame_repeat_count = data.get(
            "frame_repeat_count", lh.frame_repeat_count
        )
        lh.frame_corner_pause = data.get(
            "frame_corner_pause", lh.frame_corner_pause
        )
        lh.extra = extra
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
