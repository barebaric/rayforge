from typing import Dict, Type
from .base import StepComponentSettingsWidget
from .depth_engraver import DepthEngraverSettingsWidget
from .edge import EdgeTracerSettingsWidget
from .frame import FrameProducerSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .shrinkwrap import ShrinkWrapProducerSettingsWidget
from .smooth import SmoothSettingsWidget
from .rasterizer import RasterizerSettingsWidget


# This registry maps the class names of pipeline components (str)
# to their corresponding UI widget classes (Type).
WIDGET_REGISTRY: Dict[str, Type[StepComponentSettingsWidget]] = {
    "DepthEngraver": DepthEngraverSettingsWidget,
    "EdgeTracer": EdgeTracerSettingsWidget,
    "FrameProducer": FrameProducerSettingsWidget,
    "MultiPassTransformer": MultiPassSettingsWidget,
    "Optimize": OptimizeSettingsWidget,
    "ShrinkWrapProducer": ShrinkWrapProducerSettingsWidget,
    "Smooth": SmoothSettingsWidget,
    "Rasterizer": RasterizerSettingsWidget
}

__all__ = [
    "StepComponentSettingsWidget",
    "WIDGET_REGISTRY",
    "DepthEngraverSettingsWidget",
    "EdgeTracerSettingsWidget",
    "FrameProducerSettingsWidget",
    "MultiPassSettingsWidget",
    "OptimizeSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "SmoothSettingsWidget",
    "RasterizerSettingsWidget"
]
