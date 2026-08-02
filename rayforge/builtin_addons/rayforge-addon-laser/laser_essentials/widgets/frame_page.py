"""Frame step settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from .rows import CutSideRow, LaserStepSettingsPage, PathOffsetRow

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class FrameStepSettingsPage(LaserStepSettingsPage):
    """Settings page for the FrameStep."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        self.add_section(
            _("Geometry"),
            CutSideRow,
            PathOffsetRow,
            description=_("Cut a frame around the selected content."),
        )
