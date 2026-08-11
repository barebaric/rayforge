"""Outer profiling step settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from ..rows import StepLengthRow, StepOverRow, WallMarginRow
from .cnc_step_page import CncStepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class ProfileOuterPage(CncStepSettingsPage):
    """Settings page for the outer profiling step."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(editor, step)
        self.add_section(
            _("Profiling"),
            StepOverRow,
            StepLengthRow,
            WallMarginRow,
            description=_("Cut the exterior profile of the workpiece."),
        )
