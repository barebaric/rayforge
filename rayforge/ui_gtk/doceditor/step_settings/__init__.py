from typing import Dict, Type
from .base import StepComponentSettingsWidget
from .depth_engraver import DepthEngraverSettingsWidget
from .contour import ContourProducerSettingsWidget
from .frame import FrameProducerSettingsWidget
from .material_test_grid import MaterialTestGridSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .overscan import OverscanSettingsWidget
from .rasterizer import RasterizerSettingsWidget
from .shrinkwrap import ShrinkWrapProducerSettingsWidget
from .smooth import SmoothSettingsWidget


# This registry maps the class names of pipeline components (str)
# to their corresponding UI widget classes (Type).
WIDGET_REGISTRY: Dict[str, Type[StepComponentSettingsWidget]] = {
    "DepthEngraver": DepthEngraverSettingsWidget,
    "ContourProducer": ContourProducerSettingsWidget,
    "FrameProducer": FrameProducerSettingsWidget,
    "MaterialTestGridProducer": MaterialTestGridSettingsWidget,
    "MultiPassTransformer": MultiPassSettingsWidget,
    "Optimize": OptimizeSettingsWidget,
    "OverscanTransformer": OverscanSettingsWidget,
    "Rasterizer": RasterizerSettingsWidget,
    "ShrinkWrapProducer": ShrinkWrapProducerSettingsWidget,
    "Smooth": SmoothSettingsWidget,
}

__all__ = [
    "WIDGET_REGISTRY",
    "DepthEngraverSettingsWidget",
    "ContourProducerSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "MultiPassSettingsWidget",
    "OptimizeSettingsWidget",
    "OverscanSettingsWidget",
    "RasterizerSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "SmoothSettingsWidget",
    "StepComponentSettingsWidget",
]
