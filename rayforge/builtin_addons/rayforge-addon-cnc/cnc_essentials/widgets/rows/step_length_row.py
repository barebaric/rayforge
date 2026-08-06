"""CNC step-length row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class StepLengthRow(SpinRow):
    """A spin row bound to the step's ``step_length`` attribute."""

    def __init__(self, editor: Any, step: Any):
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
