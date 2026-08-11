"""CNC depth-per-pass row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class DepthPerPassRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.depth_per_pass``."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "depth_per_pass",
            _("Depth per Pass"),
            _("Depth removed by each pass"),
            0.1,
            10.0,
            0.1,
            2,
            quantity="length",
        )
