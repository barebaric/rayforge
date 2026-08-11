"""CNC target depth row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class TargetDepthRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.target_depth``."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "target_depth",
            _("Target Depth"),
            _("Final depth of the cut (negative is downward)"),
            -50.0,
            0.0,
            0.1,
            2,
            quantity="length",
        )
