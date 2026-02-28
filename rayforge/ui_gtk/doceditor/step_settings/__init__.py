from .registry import step_widget_registry, StepWidgetRegistry
from .base import StepComponentSettingsWidget
from .contour import ContourProducerSettingsWidget
from .engraver import EngraverSettingsWidget
from .frame import FrameProducerSettingsWidget
from .material_test_grid import MaterialTestGridSettingsWidget
from .multipass import MultiPassSettingsWidget
from .optimize import OptimizeSettingsWidget
from .overscan import OverscanSettingsWidget
from .placeholder import PlaceholderSettingsWidget
from .shrinkwrap import ShrinkWrapProducerSettingsWidget
from .smooth import SmoothSettingsWidget

step_widget_registry.register("ContourProducer", ContourProducerSettingsWidget)
step_widget_registry.register("DepthEngraver", EngraverSettingsWidget)
step_widget_registry.register("DitherRasterizer", EngraverSettingsWidget)
step_widget_registry.register("FrameProducer", FrameProducerSettingsWidget)
step_widget_registry.register(
    "MaterialTestGridProducer", MaterialTestGridSettingsWidget
)
step_widget_registry.register("MultiPassTransformer", MultiPassSettingsWidget)
step_widget_registry.register("Optimize", OptimizeSettingsWidget)
step_widget_registry.register("OverscanTransformer", OverscanSettingsWidget)
step_widget_registry.register("PlaceholderProducer", PlaceholderSettingsWidget)
step_widget_registry.register("Rasterizer", EngraverSettingsWidget)
step_widget_registry.register(
    "ShrinkWrapProducer", ShrinkWrapProducerSettingsWidget
)
step_widget_registry.register("Smooth", SmoothSettingsWidget)

__all__ = [
    "step_widget_registry",
    "StepWidgetRegistry",
    "ContourProducerSettingsWidget",
    "EngraverSettingsWidget",
    "FrameProducerSettingsWidget",
    "MaterialTestGridSettingsWidget",
    "MultiPassSettingsWidget",
    "OptimizeSettingsWidget",
    "OverscanSettingsWidget",
    "PlaceholderSettingsWidget",
    "ShrinkWrapProducerSettingsWidget",
    "SmoothSettingsWidget",
    "StepComponentSettingsWidget",
]
