"""Inner profiling step settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING

from .cnc_step_page import CncStepSettingsPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

    from ...steps.cnc_assembler_step import CncAssemblerStep


class ProfileInnerPage(CncStepSettingsPage):
    """Settings page for the inner profiling step."""

    def __init__(self, editor: "DocEditor", step: "CncAssemblerStep"):
        super().__init__(editor, step)
        step_vars = self._step_specific_group()
        if step_vars is None:
            return
        self.add_varset_section(
            _("Profiling"),
            step_vars,
            description=_("Cut the interior profile of the workpiece."),
        )
