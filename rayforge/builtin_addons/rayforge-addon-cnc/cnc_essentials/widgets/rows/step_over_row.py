"""CNC step-over row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class StepOverRow(SpinRow):
    """A spin row bound to the step's ``step_over`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "step_over",
            _("Step Over"),
            _("Lateral step-over between passes in mm"),
            0.1,
            25.0,
            0.1,
            1,
        )
