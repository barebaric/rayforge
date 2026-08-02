"""CNC safe-Z row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class SafeZRow(SpinRow):
    """A spin row bound to ``CncAssemblerStep.safe_z``."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "safe_z",
            _("Safe Z Height"),
            _("Height to retract between moves in mm"),
            0.0,
            50.0,
            0.1,
            2,
        )
