"""CNC max-deflection row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow
from rayforge.ui_gtk.shared.unit_spin_row import AngleSpinRow


class MaxDeflectionRow(SpinRow):
    """A spin row bound to the step's ``max_deflection_deg``."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "max_deflection_deg",
            _("Max Deflection"),
            _("Max steering deflection per step (degrees)"),
            1.0,
            60.0,
            1.0,
            0,
            is_int=True,
        )

    def build_widget(self):
        return AngleSpinRow(
            self._title,
            self._subtitle,
            lower=self._lower,
            upper=self._upper,
            digits=self._digits,
        )
