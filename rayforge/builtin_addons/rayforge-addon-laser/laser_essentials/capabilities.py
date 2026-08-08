"""Laser-domain step capabilities.

Holds the concrete step capabilities (CUT, ENGRAVE, SCORE,
MATERIAL_TEST) and the ``LaserHeadVar`` settings var.
"""

from gettext import gettext as _
from typing import Optional

from rayforge.context import get_context
from rayforge.core.capability import MachineCapability, StepCapability
from rayforge.core.varset import (
    BoolVar,
    ChoiceVar,
    SliderFloatVar,
    SpeedVar,
    VarSet,
)


class LaserHeadVar(ChoiceVar):
    """
    A special ChoiceVar that dynamically populates its choices with the
    names of the laser heads from the currently active machine.

    It also handles the mapping between human-readable names (for the UI)
    and the UIDs (for data storage).
    """

    def __init__(
        self,
        key: str = "selected_head_uid",
        label: str = _("Laser Head"),
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
    ):
        """
        Initializes a new LaserHeadVar instance.

        Args:
            key: The unique machine-readable identifier.
            label: The human-readable name for the UI.
            description: A longer, human-readable description.
            default: The default value (a laser head UID).
            value: The initial value. If provided, it overrides the default.
        """
        self.name_to_uid_map: dict[str, str] = {}
        self.uid_to_name_map: dict[str, str] = {}
        head_names: list[str] = []

        active_machine = get_context().machine
        if active_machine and active_machine.heads:
            laser_heads = [
                h
                for h in active_machine.heads
                if h.machine_capability is MachineCapability.LASER
            ]
            self.name_to_uid_map = {h.name: h.uid for h in laser_heads}
            self.uid_to_name_map = {h.uid: h.name for h in laser_heads}
            head_names = sorted(list(self.name_to_uid_map.keys()))

        # The value stored in the Var itself is the UID.
        # We need to translate the initial name-based value to a UID.
        initial_value_uid = value
        if value and value in self.name_to_uid_map:
            initial_value_uid = self.name_to_uid_map[value]

        super().__init__(
            key=key,
            label=label,
            choices=head_names,
            description=description,
            default=default,
            value=initial_value_uid,
        )

    def get_display_for_value(self, value: Optional[str]) -> Optional[str]:
        """Given a UID (value), return the display name."""
        if value is None:
            return None
        return self.uid_to_name_map.get(value, value)

    def get_value_for_display(self, display: Optional[str]) -> Optional[str]:
        """Given a display name, return the UID (value)."""
        if display is None:
            return None
        return self.name_to_uid_map.get(display, display)


class CutCapability(StepCapability):
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})

    @property
    def name(self) -> str:
        return "CUT"

    @property
    def label(self) -> str:
        return _("Cut")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                LaserHeadVar(
                    description=_("Optionally force a specific laser head")
                ),
                SliderFloatVar(
                    key="power",
                    label=_("Power"),
                    default=0.8,
                    min_val=0.0,
                    max_val=1.0,
                    show_value=True,
                    format_suffix="%",
                ),
                SliderFloatVar(
                    key="tab_power",
                    label=_("Tab Power"),
                    description=_(
                        "Laser power at tab positions (% of cut power)"
                    ),
                    default=0.0,
                    min_val=0.0,
                    max_val=1.0,
                    show_value=True,
                    format_suffix="%",
                ),
                SpeedVar(
                    key="cut_speed",
                    label=_("Cut Speed"),
                    default=500,
                    min_val=1,
                    role="cut",
                ),
                SpeedVar(
                    key="travel_speed",
                    label=_("Travel Speed"),
                    default=5000,
                    min_val=1,
                    role="travel",
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=False,
                ),
            ]
        )


class EngraveCapability(StepCapability):
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})

    @property
    def name(self) -> str:
        return "ENGRAVE"

    @property
    def label(self) -> str:
        return _("Engrave")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                LaserHeadVar(
                    description=_("Optionally force a specific laser head")
                ),
                SliderFloatVar(
                    key="power",
                    label=_("Power"),
                    default=0.2,
                    min_val=0.0,
                    max_val=1.0,
                    show_value=True,
                    format_suffix="%",
                ),
                SpeedVar(
                    key="cut_speed",
                    label=_("Engrave Speed"),
                    default=4000,
                    min_val=1,
                    role="cut",
                ),
                SpeedVar(
                    key="travel_speed",
                    label=_("Travel Speed"),
                    default=5000,
                    min_val=1,
                    role="travel",
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=False,
                ),
            ]
        )


class ScoreCapability(StepCapability):
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})

    @property
    def name(self) -> str:
        return "SCORE"

    @property
    def label(self) -> str:
        return _("Score")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                LaserHeadVar(
                    description=_("Optionally force a specific laser head")
                ),
                SliderFloatVar(
                    key="power",
                    label=_("Power"),
                    default=0.1,
                    min_val=0.0,
                    max_val=1.0,
                    show_value=True,
                    format_suffix="%",
                ),
                SpeedVar(
                    key="cut_speed",
                    label=_("Score Speed"),
                    default=5000,
                    min_val=1,
                    role="cut",
                ),
                SpeedVar(
                    key="travel_speed",
                    label=_("Travel Speed"),
                    default=5000,
                    min_val=1,
                    role="travel",
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=False,
                ),
            ]
        )


class MaterialTestCapability(StepCapability):
    REQUIRED_MACHINE_CAPS = frozenset({MachineCapability.LASER})

    @property
    def name(self) -> str:
        return "MATERIAL_TEST"

    @property
    def label(self) -> str:
        return _("Material Test")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=False,
                ),
            ]
        )


# Instantiate singletons of each step capability.
CUT = CutCapability()
ENGRAVE = EngraveCapability()
SCORE = ScoreCapability()
MATERIAL_TEST = MaterialTestCapability()
