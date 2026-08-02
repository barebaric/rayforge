"""Laser offset row widget."""

from gettext import gettext as _
from typing import Any

from rayforge.core.cut_side import CutSide
from rayforge.ui_gtk.doceditor.step_settings.rows import SpinRow


class OffsetRow(SpinRow):
    """A spin row bound to the step's ``offset_mm`` attribute.

    The row is insensitive while the cut side is CENTERLINE, where an
    offset has no effect.
    """

    def __init__(self, editor: Any, step: Any):
        super().__init__(
            editor,
            step,
            "offset_mm",
            _("Offset"),
            _(
                "Shifts the path inward/outward per Cut Side (none on "
                "Centerline). Defaults to kerf compensation for the head"
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
