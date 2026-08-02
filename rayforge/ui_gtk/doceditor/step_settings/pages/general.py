"""Fallback step settings page."""

from rayforge.ui_gtk.doceditor.step_settings.pages.base import StepSettingsPage


class GeneralStepSettingsPage(StepSettingsPage):
    """Fallback step settings page when no addon provides one."""

    key = ""
    path_prefix = "/step-settings/"
