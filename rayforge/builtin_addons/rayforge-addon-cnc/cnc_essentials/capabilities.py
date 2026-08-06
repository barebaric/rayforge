"""CNC-domain capabilities."""

from gettext import gettext as _

from rayforge.core.capability import MachineCapability, StepCapability
from rayforge.core.varset import (
    IntVar,
    LengthVar,
    SpeedVar,
    VarSet,
)


class MillCapability(StepCapability):
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.MILL})

    @property
    def name(self) -> str:
        return "MILL"

    @property
    def label(self) -> str:
        return _("Mill")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                LengthVar(
                    "tool_diameter",
                    _("Tool Diameter"),
                    default=6.0,
                    min_val=0.1,
                    max_val=50.0,
                ),
                IntVar(
                    "spindle_rpm",
                    _("Spindle RPM"),
                    default=12000,
                    min_val=100,
                    max_val=60000,
                ),
                SpeedVar(
                    "cut_speed",
                    _("Feed Rate"),
                    default=500,
                    min_val=1,
                    role="cut",
                ),
                SpeedVar(
                    "plunge_speed",
                    _("Plunge Rate"),
                    default=200,
                    min_val=1,
                    role="cut",
                ),
                SpeedVar(
                    "travel_speed",
                    _("Travel Speed"),
                    default=5000,
                    min_val=1,
                    role="travel",
                ),
                LengthVar(
                    "target_depth",
                    _("Target Depth"),
                    default=-5.0,
                    min_val=-50.0,
                    max_val=0.0,
                ),
                LengthVar(
                    "depth_per_pass",
                    _("Depth per Pass"),
                    default=1.0,
                    min_val=0.1,
                    max_val=10.0,
                ),
                LengthVar(
                    "safe_z",
                    _("Safe Z Height"),
                    default=2.0,
                    min_val=0.0,
                    max_val=50.0,
                ),
            ]
        )


# Instantiate singleton of the step capability.
MILL = MillCapability()
