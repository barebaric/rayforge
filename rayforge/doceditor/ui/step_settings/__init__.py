from typing import Dict, Type
from .base import StepComponentSettingsWidget
from .depth_engraver import DepthEngraverSettingsWidget
from .frame import FrameProducerSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .shrinkwrap import ShrinkWrapProducerSettingsWidget
from .smooth import SmoothSettingsWidget

# This registry maps the class names of pipeline components (str)
# to their corresponding UI widget classes (Type).
WIDGET_REGISTRY: Dict[str, Type[StepComponentSettingsWidget]] = {
    "DepthEngraver": DepthEngraverSettingsWidget,
    "FrameProducer": FrameProducerSettingsWidget,
    "MultiPassTransformer": MultiPassSettingsWidget,
    "Optimize": OptimizeSettingsWidget,
    "ShrinkWrapProducer": ShrinkWrapProducerSettingsWidget,
    "Smooth": SmoothSettingsWidget,
}

__all__ = [
    "StepComponentSettingsWidget",
    "WIDGET_REGISTRY",
    "DepthEngraverSettingsWidget",
    "FrameProducerSettingsWidget",
    "MultiPassSettingsWidget",
    "OptimizeSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "SmoothSettingsWidget",
]
