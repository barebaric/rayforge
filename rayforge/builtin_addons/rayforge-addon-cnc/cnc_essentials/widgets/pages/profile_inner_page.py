"""Inner profiling step settings page."""

from gettext import gettext as _

from .cnc_step_page import CncStepSettingsPage


class ProfileInnerPage(CncStepSettingsPage):
    """Settings page for the inner profiling step."""

    def _add_step_sections(self):
        step_vars = self._step_specific_group()
        if step_vars is None:
            return
        self.add_varset_section(
            _("Profiling"),
            step_vars,
            description=_("Cut the interior profile of the workpiece."),
        )
