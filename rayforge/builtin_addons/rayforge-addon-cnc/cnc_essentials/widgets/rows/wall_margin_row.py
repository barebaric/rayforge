"""CNC wall-margin row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class WallMarginRow(SpinRow):
    """A spin row bound to the step's ``wall_margin`` attribute."""

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "wall_margin",
            _("Wall Margin"),
            _("Extra clearance from the pocket wall in mm"),
            0.0,
            10.0,
            0.1,
            1,
        )
