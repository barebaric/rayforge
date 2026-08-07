"""Inner profiling step settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from ..rows import StepLengthRow, StepOverRow, WallMarginRow
from .cnc_step_page import CncStepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class ProfileInnerPage(CncStepSettingsPage):
    """Settings page for the inner profiling step."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        self.add_section(
            _("Profiling"),
            StepOverRow,
            StepLengthRow,
            WallMarginRow,
            description=_("Cut the interior profile of the workpiece."),
        )
