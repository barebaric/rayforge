from typing import Dict, Type
from .base import StepComponentSettingsWidget
from .multipass import MultiPassSettingsWidget
from .smooth import SmoothSettingsWidget
from .depth_engraver import DepthEngraverSettingsWidget
from .optimize import OptimizeSettingsWidget
from .shrinkwrap import HullProducerSettingsWidget

# This registry maps the class names of pipeline components (str)
# to their corresponding UI widget classes (Type).
WIDGET_REGISTRY: Dict[str, Type[StepComponentSettingsWidget]] = {
    "MultiPassTransformer": MultiPassSettingsWidget,
    "Smooth": SmoothSettingsWidget,
    "DepthEngraver": DepthEngraverSettingsWidget,
    "Optimize": OptimizeSettingsWidget,
    "HullProducer": HullProducerSettingsWidget,
}

__all__ = [
    "StepComponentSettingsWidget",
    "WIDGET_REGISTRY",
    "MultiPassSettingsWidget",
    "SmoothSettingsWidget",
    "DepthEngraverSettingsWidget",
    "OptimizeSettingsWidget",
    "HullProducerSettingsWidget",
]
