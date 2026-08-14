"""Toroidal clearing step settings page."""

from gettext import gettext as _

from .cnc_step_page import CncStepSettingsPage


class ToroidalClearPage(CncStepSettingsPage):
    """Settings page for the toroidal clearing step."""

    def _add_step_sections(self):
        step_vars = self._step_specific_group()
        if step_vars is None:
            return
        self.add_varset_section(
            _("Clearing"),
            step_vars,
            description=_("Clear a pocket with concentric passes."),
        )
