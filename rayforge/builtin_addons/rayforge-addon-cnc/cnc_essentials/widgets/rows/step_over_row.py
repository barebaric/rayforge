"""CNC step-over row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class StepOverRow(SpinRow):
    """A spin row bound to the step's ``step_over`` attribute."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "step_over",
            _("Step Over"),
            _("Lateral step-over between passes"),
            0.1,
            25.0,
            0.1,
            1,
            quantity="length",
        )
