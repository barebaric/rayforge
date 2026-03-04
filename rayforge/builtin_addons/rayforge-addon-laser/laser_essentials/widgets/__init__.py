"""
Laser Essentials UI Widgets.

Provides settings widgets for the producers in this addon.
"""

from .contour_widget import ContourProducerSettingsWidget
from .raster_widget import RasterSettingsWidget
from .frame_widget import FrameProducerSettingsWidget
from .material_test_grid_widget import MaterialTestGridSettingsWidget
from .shrinkwrap_widget import ShrinkWrapProducerSettingsWidget

__all__ = [
    "ContourProducerSettingsWidget",
    "RasterSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
]
