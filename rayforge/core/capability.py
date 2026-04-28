from __future__ import annotations
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from gettext import gettext as _
from ..context import get_context
from .varset import (
    ChoiceVar,
    IntVar,
    VarSet,
    SpeedVar,
    SliderFloatVar,
    BoolVar,
    FloatVar,
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
        key: str = "selected_laser_uid",
        label: str = "Laser Head",
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
        self.name_to_uid_map: Dict[str, str] = {}
        self.uid_to_name_map: Dict[str, str] = {}
        head_names: list[str] = []

        active_machine = get_context().machine
        if active_machine and active_machine.heads:
            self.name_to_uid_map = {
                h.name: h.uid for h in active_machine.heads
            }
            self.uid_to_name_map = {
                h.uid: h.name for h in active_machine.heads
            }
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


class Capability(ABC):
    """
    Abstract base class for a Step capability (e.g., Cut, Engrave).

    Each subclass represents a single high-level task and encapsulates:
    - A unique name for serialization.
    - A user-facing label.
    - A VarSet that serves as the template for its settings.

    Capabilities can be combined with the | operator to produce a
    merged VarSet containing all settings from both operands.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """A unique, machine-readable name for serialization (e.g., 'CUT')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def label(self) -> str:
        """A translatable, user-facing label (e.g., 'Cut')."""
        raise NotImplementedError

    @property
    @abstractmethod
    def varset(self) -> VarSet:
        """
        The VarSet that defines the settings template for this capability.
        """
        raise NotImplementedError

    def get_setting_keys(self) -> List[str]:
        """
        Returns a list of keys for the settings defined by this capability.
        """
        return [var.key for var in self.varset.vars]

    def __str__(self) -> str:
        return self.label

    def __or__(self, other: "Capability") -> "Capability":
        if not isinstance(other, Capability):
            return NotImplemented
        return _CombinedCapability(self, other)


class CutCapability(Capability):
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
                ),
                SpeedVar(
                    key="cut_speed",
                    label=_("Cut Speed"),
                    default=500,
                    min_val=1,
                ),
                SpeedVar(
                    key="travel_speed",
                    label=_("Travel Speed"),
                    default=5000,
                    min_val=1,
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=False,
                ),
            ]
        )


class EngraveCapability(Capability):
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
                ),
                SpeedVar(
                    key="cut_speed",
                    label=_("Engrave Speed"),
                    default=4000,
                    min_val=1,
                ),
                SpeedVar(
                    key="travel_speed",
                    label=_("Travel Speed"),
                    default=5000,
                    min_val=1,
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=False,
                ),
            ]
        )


class ScoreCapability(Capability):
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
                ),
                SpeedVar(
                    key="cut_speed",
                    label=_("Score Speed"),
                    default=5000,
                    min_val=1,
                ),
                SpeedVar(
                    key="travel_speed",
                    label=_("Travel Speed"),
                    default=5000,
                    min_val=1,
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=False,
                ),
            ]
        )


class KerfCapability(Capability):
    @property
    def name(self) -> str:
        return "WITH_KERF"

    @property
    def label(self) -> str:
        return _("Kerf")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                FloatVar(
                    key="kerf_mm",
                    label=_("Kerf"),
                    description=_("The effective width of the laser beam"),
                    default=0.1,
                    min_val=0.0,
                    max_val=2.0,
                ),
            ]
        )


class MaterialTestCapability(Capability):
    @property
    def name(self) -> str:
        return "MATERIAL_TEST"

    @property
    def label(self) -> str:
        return _("Material Test")

    @property
    def varset(self) -> VarSet:
        return VarSet(vars=[])


class PWMCapability(Capability):
    def __init__(
        self,
        frequency: int,
        max_frequency: int,
        pulse_width: int,
        min_pulse_width: int,
        max_pulse_width: int,
    ):
        self._frequency = frequency
        self._max_frequency = max_frequency
        self._pulse_width = pulse_width
        self._min_pulse_width = min_pulse_width
        self._max_pulse_width = max_pulse_width

    @property
    def name(self) -> str:
        return "PWM"

    @property
    def label(self) -> str:
        return _("PWM")

    @property
    def varset(self) -> VarSet:
        return VarSet(
            vars=[
                IntVar(
                    key="frequency",
                    label=_("Frequency"),
                    description=_("PWM frequency in Hz"),
                    default=self._frequency,
                    min_val=1,
                    max_val=self._max_frequency,
                ),
                IntVar(
                    key="pulse_width",
                    label=_("Pulse Width"),
                    description=_("Pulse width in microseconds"),
                    default=self._pulse_width,
                    min_val=self._min_pulse_width,
                    max_val=self._max_pulse_width,
                ),
            ]
        )


class _CombinedCapability(Capability):
    """
    A capability produced by combining two others with the | operator.
    Merges VarSets, with the right operand overriding shared keys.
    """

    def __init__(self, left: Capability, right: Capability):
        self._left = left
        self._right = right
        self._merged_varset: Optional[VarSet] = None

    @property
    def name(self) -> str:
        return f"{self._left.name}|{self._right.name}"

    @property
    def label(self) -> str:
        return f"{self._left.label} + {self._right.label}"

    @property
    def varset(self) -> VarSet:
        if self._merged_varset is None:
            from .varset import Var as VarType

            merged: Dict[str, VarType] = {}
            for cap in self._flatten():
                for var in cap.varset:
                    merged[var.key] = var
            self._merged_varset = VarSet(vars=list(merged.values()))
        return self._merged_varset

    def _flatten(self) -> List[Capability]:
        caps: List[Capability] = []
        for part in (self._left, self._right):
            if isinstance(part, _CombinedCapability):
                caps.extend(part._flatten())
            else:
                caps.append(part)
        return caps


# Instantiate singletons of each capability
CUT = CutCapability()
ENGRAVE = EngraveCapability()
SCORE = ScoreCapability()
WITH_KERF = KerfCapability()
MATERIAL_TEST = MaterialTestCapability()

# A list of all available capability instances, for populating UI dropdowns
ALL_CAPABILITIES: List[Capability] = [
    CUT,
    ENGRAVE,
    SCORE,
    WITH_KERF,
    MATERIAL_TEST,
]

# A map for deserializing from a name string back to a capability instance
CAPABILITIES_BY_NAME: Dict[str, Capability] = {
    cap.name: cap for cap in ALL_CAPABILITIES
}
