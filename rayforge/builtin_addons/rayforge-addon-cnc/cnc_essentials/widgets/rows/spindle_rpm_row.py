"""CNC spindle RPM row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class SpindleRpmRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.spindle_rpm``."""

    def __init__(self, editor: Any, step: Any):
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
