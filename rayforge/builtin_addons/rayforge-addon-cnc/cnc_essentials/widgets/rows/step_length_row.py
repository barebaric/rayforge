"""CNC step-length row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class StepLengthRow(SpinRow):
    """A spin row bound to the step's ``step_length`` attribute."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "step_length",
            _("Step Length"),
            _("Forward step length"),
            0.1,
            5.0,
            0.1,
            1,
            quantity="length",
        )
