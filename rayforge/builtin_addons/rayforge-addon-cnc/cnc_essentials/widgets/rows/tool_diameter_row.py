"""CNC tool diameter row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class ToolDiameterRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.tool_diameter``."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "tool_diameter",
            _("Tool Diameter"),
            _("Diameter of the cutting tool in mm"),
            0.1,
            50.0,
            0.1,
            2,
        )
