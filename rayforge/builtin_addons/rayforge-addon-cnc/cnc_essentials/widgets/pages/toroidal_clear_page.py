"""Toroidal clearing step settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from ..rows import StepOverRow
from .cnc_step_page import CncStepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class ToroidalClearPage(CncStepSettingsPage):
    """Settings page for the toroidal clearing step."""

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__(editor, step)
        self.add_section(
            _("Clearing"),
            StepOverRow,
            description=_("Clear a pocket with concentric passes."),
        )
