"""CNC area-tolerance row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class AreaToleranceRow(SpinRow):
    """A spin row bound to the step's ``area_tolerance`` attribute."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
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
