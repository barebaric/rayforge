"""
Material Test Grid step settings.

The grid itself (presets, grid, labels and ranges) is set in the Test
Settings of the item properties panel, like the other calibration
tests; the step settings keep the laser head and the post-processing.
"""

from .laser_step_page import LaserStepSettingsPage


class MaterialTestGridSettingsPage(LaserStepSettingsPage):
    """Material Test Grid step settings."""

    include_process = False
