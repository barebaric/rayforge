from enum import Enum
from gettext import gettext as _
from typing import Any, Optional

from ...core.capability import MachineCapability
from .head import _HEAD_SERIALIZED_KEYS, Head, head_setting


class LaserType(Enum):
    DIODE = "diode"
    CO2 = "co2"
    FIBER = "fiber"

    @property
    def supports_pwm(self) -> bool:
        return self in (LaserType.CO2, LaserType.FIBER)

    @property
    def default_wavelength_nm(self) -> float:
        """Typical emission wavelength for this laser type, in nm.

        Diodes are ~455 nm (blue, the common desktop diode band; the
        445 nm value sometimes quoted is the lower edge of the range),
        CO₂ is 10600 nm, fiber is 1064 nm.
        """
        if self is LaserType.CO2:
            return 10600.0
        if self is LaserType.FIBER:
            return 1064.0
        return 455.0


# Minimum sane laser spot size in mm. Guards against unconfigured
# (zero) spot data reaching the raster resolution code, which divides
# by the spot size.
MIN_SPOT_SIZE_MM = 0.1

# Default focal distance in mm used for 3D visuals when a laser head
# has no positive focal distance configured.  The beam renderer and the
# head-model placement must agree on this value so the beam spans
# exactly from the workpiece to the head's nozzle.
DEFAULT_FOCAL_DISTANCE_MM = 50.0

# Nominal optical output power in watts used when a laser head has no
# physical wattage configured. Chosen as a mid-range desktop value
# (common diode/CO₂ machines span 3–80 W); the physical model falls back
# to this so unconfigured heads still produce a plausible burn.
DEFAULT_MAX_POWER_WATTS = 40.0

# Fallback operation colour when a laser's cut/raster colour is unset.
DEFAULT_LASER_COLOR = "#ff00ff"


def effective_focal_distance(head: Head | None) -> float:
    """The focal distance used for 3D visuals for a laser head.

    Returns the head's configured focal distance when positive, the
    shared default otherwise.  Non-laser heads (e.g. spindles) have no
    focal distance and yield ``0.0``.
    """
    if not isinstance(head, LaserHead):
        return 0.0
    if head.focal_distance and head.focal_distance > 0:
        return head.focal_distance
    return DEFAULT_FOCAL_DISTANCE_MM


class LaserHead(Head):
    """A laser head, implying the LASER machine capability."""

    HEAD_TYPE: str = "LaserHead"

    # Reviewable settings. The ``head_setting`` markers carry the
    # review label and the conversion to the serialized (profile YAML)
    # value domain; the instance values live in the 0-1 percent domain.
    max_power = head_setting(_("Max Power"))
    frame_power_percent = head_setting(
        _("Frame Power"),
        to_yaml=lambda v: v * 100,
        from_yaml=lambda v: v / 100.0,
    )
    focus_power_percent = head_setting(
        _("Focus Power"),
        to_yaml=lambda v: v * 100,
        from_yaml=lambda v: v / 100.0,
    )
    frame_speed = head_setting(_("Frame Speed"))
    frame_repeat_count = head_setting(_("Frame Repeat Count"))
    frame_corner_pause = head_setting(_("Frame Corner Pause"))
    spot_size_mm = head_setting(_("Spot Size (mm)"))
    focal_distance = head_setting(_("Focal Distance"))
    laser_type = head_setting(
        _("Laser Type"),
        to_yaml=lambda v: v.value,
        from_yaml=lambda v: LaserType(v),
    )
    wavelength_nm = head_setting(_("Wavelength (nm)"))
    max_power_watts = head_setting(_("Max Optical Power (W)"))
    pwm_frequency = head_setting(_("PWM Frequency"))
    max_pwm_frequency = head_setting(_("Max PWM Frequency"))
    pulse_width = head_setting(_("Pulse Width"))
    min_pulse_width = head_setting(_("Min Pulse Width"))
    max_pulse_width = head_setting(_("Max Pulse Width"))

    def __init__(self):
        super().__init__()
        self.name = _("Laser Head")
        self.max_power = 1000  # Max power (0-1000 for GRBL)
        self.frame_power_percent = 0  # in percent (0-1.0)
        self.focus_power_percent = 0  # in percent (0-1.0)
        self.frame_speed = 0  # mm/min, 0 = use machine max travel speed
        self.frame_repeat_count = 20
        self.frame_corner_pause = 0  # seconds
        self.spot_size_mm = 0.1, 0.1  # millimeters
        self.cut_color: str = DEFAULT_LASER_COLOR
        self.focal_distance = 0.0
        self.laser_type = LaserType.DIODE
        self.wavelength_nm = 0.0  # 0 ⇒ LaserType.default_wavelength_nm
        self.max_power_watts = 0.0  # 0 ⇒ DEFAULT_MAX_POWER_WATTS
        self.pwm_frequency = 500
        self.max_pwm_frequency = 5000
        self.pulse_width = 50
        self.min_pulse_width = 5
        self.max_pulse_width = 500

    @property
    def machine_capability(self) -> MachineCapability:
        return MachineCapability.LASER

    @property
    def kerf_mm(self) -> float:
        """The kerf-compensation displacement for this head, in mm.

        Derived from the beam spot size: the path is shifted by half
        the spot width so the cut part comes out dimensionally
        accurate. Read-only; change ``spot_size_mm`` to alter it.
        """
        spot_x = self.spot_size_mm[0]
        return spot_x / 2.0

    @staticmethod
    def get_spot_size(head: Optional["LaserHead"]) -> tuple[float, float]:
        """The effective spot size ``(x, y)`` in mm for a laser head.

        Falls back to a sane minimum when no laser head is available.
        A missing or zero spot size (an unconfigured head) is clamped so
        raster resolution code never divides by zero.
        """
        if head is None:
            return MIN_SPOT_SIZE_MM, MIN_SPOT_SIZE_MM
        spot_x, spot_y = head.spot_size_mm
        if not spot_x or spot_x <= 0:
            spot_x = MIN_SPOT_SIZE_MM
        if not spot_y or spot_y <= 0:
            spot_y = MIN_SPOT_SIZE_MM
        return spot_x, spot_y

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
        return round((percent / 100) * self.max_power)

    def set_spot_size(self, spot_size_x_mm, spot_size_y_mm):
        self.spot_size_mm = spot_size_x_mm, spot_size_y_mm
        self.changed.send(self)

    def set_cut_color(self, color: str | None):
        self.cut_color = color or DEFAULT_LASER_COLOR
        self.changed.send(self)

    def set_focal_distance(self, distance: float):
        if self.focal_distance == distance:
            return
        self.focal_distance = distance
        self.changed.send(self)

    def set_laser_type(self, laser_type: LaserType):
        if self.laser_type == laser_type:
            return
        self.laser_type = laser_type
        self.changed.send(self)

    def effective_wavelength_nm(self) -> float:
        """The emission wavelength in nm, falling back to the laser
        type's default when no explicit value is configured."""
        if self.wavelength_nm and self.wavelength_nm > 0:
            return self.wavelength_nm
        return self.laser_type.default_wavelength_nm

    def effective_max_power_watts(self) -> float:
        """The optical output power in watts at full power, falling
        back to the nominal default when no physical wattage is set."""
        if self.max_power_watts and self.max_power_watts > 0:
            return self.max_power_watts
        return DEFAULT_MAX_POWER_WATTS

    def watts_at(self, power_fraction: float) -> float:
        """Optical power in watts for a given 0–1 power fraction.

        ``power_fraction`` is the duty cycle (S-value / max_power) the
        controller commands; the returned value is the time-averaged
        optical power the material sees, which for CW lasers with PWM
        power control equals ``max_power_watts * power_fraction``.
        """
        return self.effective_max_power_watts() * power_fraction

    def set_pwm_frequency(self, frequency: int):
        frequency = max(1, min(frequency, self.max_pwm_frequency))
        if self.pwm_frequency == frequency:
            return
        self.pwm_frequency = frequency
        self.changed.send(self)

    def set_max_pwm_frequency(self, max_frequency: int):
        max_frequency = max(1, max_frequency)
        if self.max_pwm_frequency == max_frequency:
            return
        self.max_pwm_frequency = max_frequency
        self.pwm_frequency = min(self.pwm_frequency, max_frequency)
        self.changed.send(self)

    def set_pulse_width(self, width: int):
        width = max(self.min_pulse_width, min(width, self.max_pulse_width))
        if self.pulse_width == width:
            return
        self.pulse_width = width
        self.changed.send(self)

    def set_min_pulse_width(self, min_width: int):
        min_width = max(1, min_width)
        if self.min_pulse_width == min_width:
            return
        self.min_pulse_width = min_width
        self.max_pulse_width = max(self.max_pulse_width, min_width)
        self.pulse_width = max(self.pulse_width, min_width)
        self.changed.send(self)

    def set_max_pulse_width(self, max_width: int):
        max_width = max(1, max_width)
        if self.max_pulse_width == max_width:
            return
        self.max_pulse_width = max_width
        self.min_pulse_width = min(self.min_pulse_width, max_width)
        self.pulse_width = min(self.pulse_width, max_width)
        self.changed.send(self)

    def to_dict(self) -> dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "max_power": self.max_power,
                "frame_power_percent": self.frame_power_percent * 100,
                "focus_power_percent": self.focus_power_percent * 100,
                "frame_speed": self.frame_speed,
                "frame_repeat_count": self.frame_repeat_count,
                "frame_corner_pause": self.frame_corner_pause,
                "spot_size_mm": self.spot_size_mm,
                "cut_color": self.cut_color,
                "focal_distance": self.focal_distance,
                "laser_type": self.laser_type.value,
                "wavelength_nm": self.wavelength_nm,
                "max_power_watts": self.max_power_watts,
                "pwm_frequency": self.pwm_frequency,
                "max_pwm_frequency": self.max_pwm_frequency,
                "pulse_width": self.pulse_width,
                "min_pulse_width": self.min_pulse_width,
                "max_pulse_width": self.max_pulse_width,
            }
        )
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LaserHead":
        known_keys = _HEAD_SERIALIZED_KEYS | {
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
            "focal_distance",
            "laser_type",
            "wavelength_nm",
            "max_power_watts",
            "pwm_frequency",
            "max_pwm_frequency",
            "pulse_width",
            "min_pulse_width",
            "max_pulse_width",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        lh = super().from_dict(data)
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
        lh.cut_color = data.get("cut_color") or DEFAULT_LASER_COLOR
        lh.frame_speed = data.get("frame_speed", lh.frame_speed)
        lh.frame_repeat_count = data.get(
            "frame_repeat_count", lh.frame_repeat_count
        )
        lh.frame_corner_pause = data.get(
            "frame_corner_pause", lh.frame_corner_pause
        )
        lh.focal_distance = data.get("focal_distance", 0.0)
        lh.laser_type = LaserType(
            data.get("laser_type", LaserType.DIODE.value)
        )
        lh.wavelength_nm = data.get("wavelength_nm", 0.0)
        lh.max_power_watts = data.get("max_power_watts", 0.0)
        lh.pwm_frequency = data.get("pwm_frequency", lh.pwm_frequency)
        lh.max_pwm_frequency = data.get(
            "max_pwm_frequency", lh.max_pwm_frequency
        )
        lh.pulse_width = data.get("pulse_width", lh.pulse_width)
        lh.min_pulse_width = data.get("min_pulse_width", lh.min_pulse_width)
        lh.max_pulse_width = data.get("max_pulse_width", lh.max_pulse_width)
        lh.extra = extra
        return lh


# Backward-compatible alias for code that still imports `Laser`.
Laser = LaserHead
