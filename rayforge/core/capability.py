from __future__ import annotations

import enum
from gettext import gettext as _


class MachineCapability(enum.Enum):
    """
    Hardware capabilities of a machine (e.g., LASER, MILL).

    These describe what the machine's hardware can do and are used to
    filter which steps are offered to the user.
    """

    LASER = "LASER"
    MILL = "MILL"
    PWM = "PWM"
    ROTARY = "ROTARY"
    # Future: PROBE, DWELL, ...

    @property
    def label(self) -> str:
        """User-facing label for this capability."""
        return _MACHINE_CAPABILITY_LABELS[self]

    @property
    def description(self) -> str:
        """User-facing description for this capability."""
        return _MACHINE_CAPABILITY_DESCRIPTIONS[self]


_MACHINE_CAPABILITY_LABELS = {
    MachineCapability.LASER: _("Laser"),
    MachineCapability.MILL: _("Mill"),
    MachineCapability.PWM: _("PWM"),
    MachineCapability.ROTARY: _("Rotary"),
}

_MACHINE_CAPABILITY_DESCRIPTIONS = {
    MachineCapability.LASER: _("Cutting and engraving with a laser"),
    MachineCapability.MILL: _("Milling and routing with a spindle"),
    MachineCapability.PWM: _("Pulse-width-modulated laser power control"),
    MachineCapability.ROTARY: _(
        "Rotary axis attachment for cylindrical objects"
    ),
}
