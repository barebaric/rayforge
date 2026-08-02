"""Travel speed row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from .spin_row import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class TravelSpeedRow(SpinRow):
    """A spin row bound to the base ``Step.travel_speed`` attribute."""

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
        title: str = _("Travel Speed"),
    ):
        super().__init__(
            editor,
            step,
            "travel_speed",
            title,
            _("Speed of rapid positioning moves"),
            1.0,
            float(getattr(step, "max_travel_speed", 10000.0)),
            10.0,
            0,
            is_int=True,
        )

    def _sync_dependencies(self):
        max_speed = getattr(self.step, "max_travel_speed", None)
        if max_speed:
            self.set_range(1.0, float(max_speed))
