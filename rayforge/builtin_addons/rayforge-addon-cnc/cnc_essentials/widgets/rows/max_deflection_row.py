"""CNC max-deflection row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow
from rayforge.ui_gtk.shared.pref_rows import AngleSpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class MaxDeflectionRow(SpinRow):
    """A spin row bound to the step's ``max_deflection_deg``."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
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
