"""
Laser Essentials UI Widgets.
"""

from .contour_widget import ContourProducerSettingsWidget
from .frame_widget import FrameProducerSettingsWidget
from .material_test_grid_widget import MaterialTestGridSettingsWidget
from .raster_widget import RasterSettingsWidget
from .shrinkwrap_widget import ShrinkWrapProducerSettingsWidget
from .wavefront_widget import WavefrontSettingsWidget

ASSEMBLER_WIDGETS = {
    "contour": ContourProducerSettingsWidget,
    "frame": FrameProducerSettingsWidget,
    "raster": RasterSettingsWidget,
    "shrinkwrap": ShrinkWrapProducerSettingsWidget,
    "wavefront": WavefrontSettingsWidget,
    "material_test_grid": MaterialTestGridSettingsWidget,
}

__all__ = [
    "WavefrontSettingsWidget",
    "ContourProducerSettingsWidget",
    "RasterSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "ASSEMBLER_WIDGETS",
]
