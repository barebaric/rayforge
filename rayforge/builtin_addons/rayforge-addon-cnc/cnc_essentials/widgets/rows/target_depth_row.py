"""CNC target depth row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class TargetDepthRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.target_depth``."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "target_depth",
            _("Target Depth"),
            _("Final depth of the cut in mm (negative is downward)"),
            -50.0,
            0.0,
            0.1,
            2,
        )
