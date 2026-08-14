"""Frame step settings page."""

from gettext import gettext as _

from .laser_step_page import LaserStepSettingsPage


class FrameStepSettingsPage(LaserStepSettingsPage):
    """Settings page for the FrameStep."""

    def _add_step_sections(self):
        groups = self.step.recipe_varset_groups()
        step_vars = groups[-1][1] if len(groups) > 1 else None
        if step_vars is None:
            return
        self.add_varset_section(
            _("Geometry"),
            step_vars,
            description=_("Cut a frame around the selected content."),
        )
