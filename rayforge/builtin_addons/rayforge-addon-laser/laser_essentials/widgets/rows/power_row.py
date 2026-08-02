"""Laser power row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SliderRow


class PowerRow(SliderRow):
    """A slider row bound to the ``LaserStep.power`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "power",
            _("Power"),
            _("Laser power as a percentage"),
            0.0,
            1.0,
            0.01,
            1,
        )

    def _format(self, value: float) -> str:
        return f"{value * 100:.0f}%"
