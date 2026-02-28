"""
Laser Essentials UI Widgets.

Provides settings widgets for the producers in this addon.
"""

from .contour import ContourProducerSettingsWidget
from .engraver import EngraverSettingsWidget
from .frame import FrameProducerSettingsWidget
from .material_test_grid import MaterialTestGridSettingsWidget
from .shrinkwrap import ShrinkWrapProducerSettingsWidget

__all__ = [
    "ContourProducerSettingsWidget",
    "EngraverSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
]
