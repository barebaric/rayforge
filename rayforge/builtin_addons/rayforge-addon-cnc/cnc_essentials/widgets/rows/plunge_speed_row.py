"""CNC plunge speed row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class PlungeSpeedRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.plunge_speed``."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "plunge_speed",
            _("Plunge Rate"),
            _("Vertical feed rate in mm/min"),
            1.0,
            float(getattr(step, "max_cut_speed", 10000.0)),
            10.0,
            0,
            is_int=True,
        )

    def _sync_dependencies(self):
        max_speed = getattr(self.step, "max_cut_speed", None)
        if max_speed:
            self.set_range(1.0, float(max_speed))
