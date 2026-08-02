"""Laser kerf row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class KerfRow(SpinRow):
    """A spin row bound to the ``LaserStep.kerf_mm`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "kerf_mm",
            _("Kerf"),
            _("Beam width of the cut in mm"),
            0.0,
            10.0,
            0.01,
            3,
        )
