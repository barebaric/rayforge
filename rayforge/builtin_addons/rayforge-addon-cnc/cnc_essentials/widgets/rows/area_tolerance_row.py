"""CNC area-tolerance row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class AreaToleranceRow(SpinRow):
    """A spin row bound to the step's ``area_tolerance`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "area_tolerance",
            _("Area Tolerance"),
            _("Stopping tolerance in mm²"),
            0.01,
            5.0,
            0.01,
            2,
        )
