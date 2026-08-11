"""Adaptive clearing step settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from ..rows import (
    AreaToleranceRow,
    MaxDeflectionRow,
    StepLengthRow,
    StepOverRow,
    WallMarginRow,
)
from .cnc_step_page import CncStepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class AdaptiveClearPage(CncStepSettingsPage):
    """Settings page for the adaptive clearing step."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(editor, step)
        self.add_section(
            _("Adaptive Clearing"),
            StepOverRow,
            StepLengthRow,
            MaxDeflectionRow,
            WallMarginRow,
            AreaToleranceRow,
            description=_("Rough out a pocket with adaptive passes."),
        )
