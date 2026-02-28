from .registry import step_widget_registry, StepWidgetRegistry
from .base import StepComponentSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .overscan import OverscanSettingsWidget
from .placeholder import PlaceholderSettingsWidget
from .smooth import SmoothSettingsWidget
from rayforge.pipeline.producer.placeholder import PlaceholderProducer
from rayforge.pipeline.transformer.multipass import MultiPassTransformer
from rayforge.pipeline.transformer.optimize import Optimize
from rayforge.pipeline.transformer.overscan import OverscanTransformer
from rayforge.pipeline.transformer.smooth import Smooth

step_widget_registry.register(PlaceholderProducer, PlaceholderSettingsWidget)
step_widget_registry.register(MultiPassTransformer, MultiPassSettingsWidget)
step_widget_registry.register(Optimize, OptimizeSettingsWidget)
step_widget_registry.register(OverscanTransformer, OverscanSettingsWidget)
step_widget_registry.register(Smooth, SmoothSettingsWidget)

__all__ = [
    "step_widget_registry",
    "StepWidgetRegistry",
    "MultiPassSettingsWidget",
    "OptimizeSettingsWidget",
    "OverscanSettingsWidget",
    "PlaceholderSettingsWidget",
    "SmoothSettingsWidget",
    "StepComponentSettingsWidget",
]
