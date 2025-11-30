from __future__ import annotations
from typing import Dict, List, Optional
from abc import ABC, abstractmethod
from ..context import get_context
from .varset import (
    ChoiceVar,
    VarSet,
    IntVar,
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
                ),
                IntVar(
                    key="cut_speed",
                    label=_("Cut Speed"),
                    default=500,
                    min_val=1,
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=True,
                ),
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
                ),
                IntVar(
                    key="cut_speed",
                    label=_("Engrave Speed"),
                    default=4000,
                    min_val=1,
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=True,
                ),
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
                ),
                IntVar(
                    key="cut_speed",
                    label=_("Score Speed"),
                    default=5000,
                    min_val=1,
                ),
                BoolVar(
                    key="air_assist",
                    label=_("Air Assist"),
                    default=True,
                ),
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


# Instantiate singletons of each capability
CUT = CutCapability()
ENGRAVE = EngraveCapability()
SCORE = ScoreCapability()

# A list of all available capability instances, for populating UI dropdowns
ALL_CAPABILITIES: List[Capability] = [
    CUT,
    ENGRAVE,
    SCORE,
]

# A map for deserializing from a name string back to a capability instance
CAPABILITIES_BY_NAME: Dict[str, Capability] = {
    cap.name: cap for cap in ALL_CAPABILITIES
}
