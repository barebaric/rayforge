"""Helix/ramp/spiral step settings page."""

from .cnc_step_page import CncStepSettingsPage


class HelixPlungePage(CncStepSettingsPage):
    """Settings page for helix/ramp/spiral steps.

    All parameters live on the common CNC sections; no extra rows.
    """
