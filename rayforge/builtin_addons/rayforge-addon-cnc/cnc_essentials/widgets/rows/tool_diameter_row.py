"""CNC tool diameter row widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class ToolDiameterRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.tool_diameter``."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(
            editor,
            step,
            "tool_diameter",
            _("Tool Diameter"),
            _("Diameter of the cutting tool"),
            0.1,
            50.0,
            0.1,
            2,
            quantity="length",
        )
