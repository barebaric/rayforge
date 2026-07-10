"""
Laser Essentials UI Widgets.

Provides settings widgets for the producers in this addon.
"""

from ..producers import (
    ContourProducer,
    FrameProducer,
    MaterialTestGridProducer,
    Rasterizer,
    ShrinkWrapProducer,
    WavefrontProducer,
)
from .contour_widget import ContourProducerSettingsWidget
from .frame_widget import FrameProducerSettingsWidget
from .material_test_grid_widget import MaterialTestGridSettingsWidget
from .raster_widget import RasterSettingsWidget
from .shrinkwrap_widget import ShrinkWrapProducerSettingsWidget
from .wavefront_widget import WavefrontSettingsWidget

PRODUCER_WIDGETS = {
    WavefrontProducer: WavefrontSettingsWidget,
    ContourProducer: ContourProducerSettingsWidget,
    FrameProducer: FrameProducerSettingsWidget,
    MaterialTestGridProducer: MaterialTestGridSettingsWidget,
    Rasterizer: RasterSettingsWidget,
    ShrinkWrapProducer: ShrinkWrapProducerSettingsWidget,
}

__all__ = [
    "WavefrontSettingsWidget",
    "ContourProducerSettingsWidget",
    "RasterSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "PRODUCER_WIDGETS",
]
