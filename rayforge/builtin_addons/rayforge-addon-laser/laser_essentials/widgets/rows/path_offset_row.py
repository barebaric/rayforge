"""Laser path-offset row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.core.cut_side import CutSide
from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class PathOffsetRow(SpinRow):
    """A spin row bound to the step's ``path_offset_mm`` attribute.

    The row is insensitive while the cut side is CENTERLINE, where a
    path offset has no effect.
    """

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "path_offset_mm",
            _("Path Offset"),
            _(
                "Absolute distance from the original path; "
                "direction is controlled by Cut Side"
            ),
            0.0,
            100.0,
            0.1,
            2,
        )

    def _sync_dependencies(self):
        self.set_sensitive(
            getattr(self.step, "cut_side", None) != CutSide.CENTERLINE.name
        )
