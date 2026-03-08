from .registry import step_widget_registry, StepWidgetRegistry
from .base import StepComponentSettingsWidget
from .crop import CropTransformerSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .overscan import OverscanSettingsWidget
from .placeholder import PlaceholderSettingsWidget
from .smooth import SmoothSettingsWidget
from rayforge.pipeline.producer.placeholder import PlaceholderProducer
from rayforge.pipeline.transformer.crop_transformer import CropTransformer
from rayforge.pipeline.transformer.multipass_transformer import (
    MultiPassTransformer,
)
from rayforge.pipeline.transformer.optimize_transformer import Optimize
from rayforge.pipeline.transformer.overscan_transformer import (
    OverscanTransformer,
)
from rayforge.pipeline.transformer.smooth_transformer import Smooth

step_widget_registry.register(PlaceholderProducer, PlaceholderSettingsWidget)
step_widget_registry.register(CropTransformer, CropTransformerSettingsWidget)
step_widget_registry.register(MultiPassTransformer, MultiPassSettingsWidget)
step_widget_registry.register(Optimize, OptimizeSettingsWidget)
step_widget_registry.register(OverscanTransformer, OverscanSettingsWidget)
step_widget_registry.register(Smooth, SmoothSettingsWidget)

__all__ = [
    "step_widget_registry",
    "StepWidgetRegistry",
    "CropTransformerSettingsWidget",
    "MultiPassSettingsWidget",
    "OptimizeSettingsWidget",
    "OverscanSettingsWidget",
    "PlaceholderSettingsWidget",
    "SmoothSettingsWidget",
    "StepComponentSettingsWidget",
]
