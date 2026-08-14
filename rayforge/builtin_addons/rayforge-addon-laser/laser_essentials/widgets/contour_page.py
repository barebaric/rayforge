"""Contour step settings widget."""

from gettext import gettext as _

from .laser_step_page import LaserStepSettingsPage


class ContourStepSettingsPage(LaserStepSettingsPage):
    """Settings page for the ContourStep."""

    include_tab_power = True

    def _add_step_sections(self):
        groups = self.step.recipe_varset_groups()
        step_vars = groups[-1][1] if len(groups) > 1 else None
        if step_vars is None:
            return
        self.add_varset_section(
            _("Contour Settings"),
            step_vars,
            description=_("Trace the outline of the selected shapes."),
        )
