"""CNC spindle RPM row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class SpindleRpmRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.spindle_rpm``."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "spindle_rpm",
            _("Spindle RPM"),
            None,
            100,
            60000,
            100,
            0,
            is_int=True,
        )
