from typing import Dict, Type
from .base import StepComponentSettingsWidget
from .engraver import EngraverSettingsWidget
from .contour import ContourProducerSettingsWidget
from .frame import FrameProducerSettingsWidget
from .material_test_grid import MaterialTestGridSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .overscan import OverscanSettingsWidget
from .shrinkwrap import ShrinkWrapProducerSettingsWidget
from .smooth import SmoothSettingsWidget


WIDGET_REGISTRY: Dict[str, Type[StepComponentSettingsWidget]] = {
    "ContourProducer": ContourProducerSettingsWidget,
    "DepthEngraver": EngraverSettingsWidget,
    "DitherRasterizer": EngraverSettingsWidget,
    "FrameProducer": FrameProducerSettingsWidget,
    "MaterialTestGridProducer": MaterialTestGridSettingsWidget,
    "MultiPassTransformer": MultiPassSettingsWidget,
    "Optimize": OptimizeSettingsWidget,
    "OverscanTransformer": OverscanSettingsWidget,
    "Rasterizer": EngraverSettingsWidget,
    "ShrinkWrapProducer": ShrinkWrapProducerSettingsWidget,
    "Smooth": SmoothSettingsWidget,
}

__all__ = [
    "WIDGET_REGISTRY",
    "ContourProducerSettingsWidget",
    "EngraverSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "MultiPassSettingsWidget",
    "OptimizeSettingsWidget",
    "OverscanSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "SmoothSettingsWidget",
    "StepComponentSettingsWidget",
]
