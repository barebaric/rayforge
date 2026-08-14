"""The spindle-head selection VarSet variable."""

from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from rayforge.context import get_context
from rayforge.core.capability import MachineCapability
from rayforge.core.varset import ChoiceVar


class SpindleHeadVar(ChoiceVar):
    """
    A special ChoiceVar that dynamically populates its choices with the
    names of the spindle heads from the currently active machine.

    It also handles the mapping between human-readable names (for the UI)
    and the UIDs (for data storage).
    """

    def __init__(
        self,
        key: str = "selected_head_uid",
        label: str = _("Spindle"),
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        sensitive_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        """
        Initialize a new SpindleHeadVar instance.

        Args:
            key: The unique machine-readable identifier.
            label: The human-readable name for the UI.
            description: A longer, human-readable description.
            default: The default value (a spindle head UID).
            value: The initial value. If provided, it overrides the
                default.
            visible_when: Optional callable that receives a dict of all
                          current var values in the widget and returns
                          True when this var's row should be visible.
            sensitive_when: Optional callable that receives a dict of
                            all current var values in the widget and
                            returns True when this var's row should be
                            interactive.
        """
        self.name_to_uid_map: dict[str, str] = {}
        self.uid_to_name_map: dict[str, str] = {}
        head_names: list[str] = []

        active_machine = get_context().machine
        if active_machine and active_machine.heads:
            spindle_heads = [
                h
                for h in active_machine.heads
                if h.machine_capability is MachineCapability.MILL
            ]
            self.name_to_uid_map = {h.name: h.uid for h in spindle_heads}
            self.uid_to_name_map = {h.uid: h.name for h in spindle_heads}
            head_names = sorted(self.name_to_uid_map.keys())

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
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )

    def get_display_for_value(self, value: str | None) -> str | None:
        """Given a UID (value), return the display name."""
        if value is None:
            return None
        return self.uid_to_name_map.get(value, value)

    def get_value_for_display(self, display: str | None) -> str | None:
        """Given a display name, return the UID (value)."""
        if display is None:
            return None
        return self.name_to_uid_map.get(display, display)
