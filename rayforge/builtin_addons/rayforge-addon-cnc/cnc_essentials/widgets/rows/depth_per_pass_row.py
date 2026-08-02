"""CNC depth-per-pass row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class DepthPerPassRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.depth_per_pass``."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "depth_per_pass",
            _("Depth per Pass"),
            _("Depth removed by each pass in mm"),
            0.1,
            10.0,
            0.1,
            2,
        )
