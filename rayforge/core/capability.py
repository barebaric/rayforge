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
    DRAG_KNIFE = "DRAG_KNIFE"
    TANGENTIAL_KNIFE = "TANGENTIAL_KNIFE"
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
    MachineCapability.DRAG_KNIFE: _("Drag Knife"),
    MachineCapability.TANGENTIAL_KNIFE: _("Tangential Knife"),
}

_MACHINE_CAPABILITY_DESCRIPTIONS = {
    MachineCapability.LASER: _("Cutting and engraving with a laser"),
    MachineCapability.MILL: _("Milling and routing with a spindle"),
    MachineCapability.PWM: _("Pulse-width-modulated laser power control"),
    MachineCapability.ROTARY: _(
        "Rotary axis attachment for cylindrical objects"
    ),
    MachineCapability.DRAG_KNIFE: _(
        "Cutting with a trailing-blade drag knife"
    ),
    MachineCapability.TANGENTIAL_KNIFE: _(
        "Cutting with a rotary tangential knife"
    ),
}
