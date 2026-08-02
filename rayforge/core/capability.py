from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from gettext import gettext as _
from typing import ClassVar, FrozenSet, List, Optional

from .varset import VarSet, merge_varsets


class MachineCapability(enum.Enum):
    """
    Hardware capabilities of a machine (e.g., LASER, MILL).

    These describe what the machine's hardware can do and are used to
    filter which steps are offered to the user. They are distinct from
    the step `StepCapability` class (CUT, ENGRAVE, ...), which describes
    operation categories for recipe matching.
    """

    LASER = "LASER"
    MILL = "MILL"
    PWM = "PWM"
    # Future: PROBE, DWELL, ROTARY, ...

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
}

_MACHINE_CAPABILITY_DESCRIPTIONS = {
    MachineCapability.LASER: _("Cutting and engraving with a laser"),
    MachineCapability.MILL: _("Milling and routing with a spindle"),
    MachineCapability.PWM: _("Pulse-width-modulated laser power control"),
}


class StepCapability(ABC):
    """
    Abstract base class for a Step capability (e.g., Cut, Engrave).

    Each subclass represents a single high-level task and encapsulates:
    - A unique name for serialization.
    - A user-facing label.
    - A VarSet that serves as the template for its settings.

    Capabilities can be combined with the | operator to produce a
    merged VarSet containing all settings from both operands.

    Concrete capabilities are registered by their domain addons via the
    ``register_step_capabilities`` hook and resolved by name through the
    global :data:`~.capability_registry.step_capability_registry`.
    """

    #: Machine capabilities a machine must have for this capability to
    #: be usable (e.g. a laser capability requires LASER).
    REQUIRED_MACHINE_CAPS: ClassVar[FrozenSet[MachineCapability]] = frozenset()

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

    @property
    def icon_name(self) -> str:
        """The name of the icon that represents this capability."""
        return f"{self.name.lower()}-symbolic"

    def get_setting_keys(self) -> List[str]:
        """
        Returns a list of keys for the settings defined by this capability.
        """
        return [var.key for var in self.varset.vars]

    def __str__(self) -> str:
        return self.label

    def __or__(self, other: "StepCapability") -> "StepCapability":
        if not isinstance(other, StepCapability):
            return NotImplemented
        return _CombinedCapability(self, other)


class _CombinedCapability(StepCapability):
    """
    A capability produced by combining two others with the | operator.
    Merges VarSets, with the right operand overriding shared keys.
    """

    def __init__(self, left: StepCapability, right: StepCapability):
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
            self._merged_varset = merge_varsets(
                *(cap.varset for cap in self._flatten())
            )
        return self._merged_varset

    def _flatten(self) -> List[StepCapability]:
        caps: List[StepCapability] = []
        for part in (self._left, self._right):
            if isinstance(part, _CombinedCapability):
                caps.extend(part._flatten())
            else:
                caps.append(part)
        return caps
