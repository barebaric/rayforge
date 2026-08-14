"""Adaptive clearing step settings page."""

from gettext import gettext as _

from .cnc_step_page import CncStepSettingsPage


class AdaptiveClearPage(CncStepSettingsPage):
    """Settings page for the adaptive clearing step."""

    def _add_step_sections(self):
        step_vars = self._step_specific_group()
        if step_vars is None:
            return
        self.add_varset_section(
            _("Adaptive Clearing"),
            step_vars,
            description=_("Rough out a pocket with adaptive passes."),
        )
