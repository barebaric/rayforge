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
)
from .contour_widget import ContourProducerSettingsWidget
from .frame_widget import FrameProducerSettingsWidget
from .material_test_grid_widget import MaterialTestGridSettingsWidget
from .raster_widget import RasterSettingsWidget
from .shrinkwrap_widget import ShrinkWrapProducerSettingsWidget

PRODUCER_WIDGETS = {
    ContourProducer: ContourProducerSettingsWidget,
    FrameProducer: FrameProducerSettingsWidget,
    MaterialTestGridProducer: MaterialTestGridSettingsWidget,
    Rasterizer: RasterSettingsWidget,
    ShrinkWrapProducer: ShrinkWrapProducerSettingsWidget,
}

__all__ = [
    "ContourProducerSettingsWidget",
    "RasterSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "PRODUCER_WIDGETS",
]
