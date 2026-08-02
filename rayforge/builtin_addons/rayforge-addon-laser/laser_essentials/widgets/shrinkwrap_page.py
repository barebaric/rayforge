"""Shrink-wrap step settings widget."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.ui_gtk.doceditor.step_settings.rows import SliderRow

from .rows import CutSideRow, LaserStepSettingsPage, OffsetRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class GravityRow(SliderRow):
    """Slider row bound to the ``gravity`` attribute."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(
            editor,
            step,
            "gravity",
            _("Gravity"),
            _("Pulls the hull inward. 0.0 is a standard convex hull"),
            0.0,
            1.0,
            0.01,
            2,
        )


class ShrinkWrapStepSettingsPage(LaserStepSettingsPage):
    """Settings page for the ShrinkWrapStep."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        self.add_section(
            _("Shrink Wrap"),
            GravityRow,
            CutSideRow,
            OffsetRow,
            description=_("Fit a hull around the content and trace it."),
        )
