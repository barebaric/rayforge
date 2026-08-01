"""Laser-domain step base class.

Intermediate base for all laser steps. Declares the laser process
attributes and the laser-specific behaviour (initial ops, summary,
settlers, serialization of the laser keys).
"""

from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, Optional, cast

from raygeo.ops import Ops
from raygeo.ops.state import AirAssistMode

from rayforge.core.step import Step
from rayforge.machine.models.laser import LaserHead
from rayforge.shared.units.formatter import format_value

if TYPE_CHECKING:
    from rayforge.machine.models.machine import Machine


class LaserStep(Step):
    """Base for all laser-domain steps. Owns laser attributes."""

    def __init__(self, typelabel, name=None):
        self.power: float = 1.0
        self.max_power: int = 1000
        self.air_assist: bool = False
        self.kerf_mm: float = 0.0
        self.tab_power: float = 0.0
        self.frequency: int = 0
        self.pulse_width: int = 0
        super().__init__(typelabel, name=name)

    def create_initial_ops(self) -> Ops:
        """Build the initial Ops object with step-wide machine settings."""
        ops = Ops()
        ops.set_power(self.power)
        ops.set_feed_rate(self.cut_speed)
        ops.set_rapid_rate(self.travel_speed)
        ops.set_air_assist(
            AirAssistMode.ON if self.air_assist else AirAssistMode.OFF
        )
        if self.frequency:
            ops.set_frequency(self.frequency)
        if self.pulse_width:
            ops.set_pulse_width(self.pulse_width)
        return ops

    def get_selected_laser(self, machine: "Machine") -> Optional[LaserHead]:
        """Typed convenience — returns the selected LaserHead or None."""
        head = self.get_selected_head(machine)
        if isinstance(head, LaserHead):
            return head
        return None

    def set_power(self, power: float):
        if not (0.0 <= power <= 1.0):
            raise ValueError("Power must be between 0.0 and 1.0")
        if self.power != power:
            self.power = power
            self.updated.send(self)

    def set_air_assist(self, enabled: bool):
        if self.air_assist != enabled:
            self.air_assist = bool(enabled)
            self.updated.send(self)

    def set_kerf_mm(self, kerf: float):
        """Sets the kerf (beam width) in millimeters for this process."""
        if self.kerf_mm != kerf:
            self.kerf_mm = float(kerf)
            self.updated.send(self)

    def set_tab_power(self, power: float):
        if not (0.0 <= power <= 1.0):
            raise ValueError("Tab power must be between 0.0 and 1.0")
        if self.tab_power != power:
            self.tab_power = power
            self.updated.send(self)

    def set_frequency(self, frequency: int):
        if self.frequency != frequency:
            self.frequency = int(frequency)
            self.updated.send(self)

    def set_pulse_width(self, width: int):
        if self.pulse_width != width:
            self.pulse_width = int(width)
            self.updated.send(self)

    def get_summary(self) -> str:
        power_percent = round(self.power * 100)
        speed_str = format_value(self.cut_speed, "speed")
        return _("{power_percent}% power, {speed_str}").format(
            power_percent=power_percent, speed_str=speed_str
        )

    def to_dict(self) -> Dict[str, Any]:
        result = super().to_dict()
        result.update(
            {
                "power": self.power,
                "max_power": self.max_power,
                "air_assist": self.air_assist,
                "kerf_mm": self.kerf_mm,
                "tab_power": self.tab_power,
                "frequency": self.frequency,
                "pulse_width": self.pulse_width,
            }
        )
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LaserStep":
        step = cast("LaserStep", super().from_dict(data))
        step.power = data.get("power", step.power)
        step.max_power = data.get("max_power", step.max_power)
        step.air_assist = data.get("air_assist", step.air_assist)
        step.kerf_mm = data.get("kerf_mm", step.kerf_mm)
        step.tab_power = data.get("tab_power", step.tab_power)
        step.frequency = data.get("frequency", step.frequency)
        step.pulse_width = data.get("pulse_width", step.pulse_width)
        return step

    @classmethod
    def _serialized_keys(cls) -> frozenset[str]:
        return super()._serialized_keys() | frozenset(
            {
                "power",
                "max_power",
                "air_assist",
                "kerf_mm",
                "tab_power",
                "frequency",
                "pulse_width",
            }
        )
