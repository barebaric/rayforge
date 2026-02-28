from rayforge.ui_gtk.doceditor.step_settings.registry import (
    StepWidgetRegistry,
    step_widget_registry,
)
from rayforge.ui_gtk.doceditor.step_settings import (
    ContourProducerSettingsWidget,
    EngraverSettingsWidget,
    FrameProducerSettingsWidget,
    MaterialTestGridSettingsWidget,
    MultiPassSettingsWidget,
    OptimizeSettingsWidget,
    OverscanSettingsWidget,
    ShrinkWrapProducerSettingsWidget,
    SmoothSettingsWidget,
)


class MockWidget:
    """Mock widget for testing."""

    pass


class TestStepWidgetRegistry:
    def test_register_and_get(self):
        registry = StepWidgetRegistry()
        registry.register("TestProducer", MockWidget)
        assert registry.get("TestProducer") == MockWidget

    def test_get_unknown_returns_none(self):
        registry = StepWidgetRegistry()
        assert registry.get("NonExistentWidget") is None

    def test_all_widgets_returns_copy(self):
        registry = StepWidgetRegistry()
        registry.register("TestProducer", MockWidget)
        all_widgets = registry.all_widgets()
        assert all_widgets == {"TestProducer": MockWidget}
        all_widgets["NewOne"] = MockWidget
        assert "NewOne" not in registry.all_widgets()

    def test_global_registry_has_producer_widgets(self):
        assert (
            step_widget_registry.get("ContourProducer")
            == ContourProducerSettingsWidget
        )
        assert step_widget_registry.get("Rasterizer") == EngraverSettingsWidget
        assert (
            step_widget_registry.get("DepthEngraver") == EngraverSettingsWidget
        )
        assert (
            step_widget_registry.get("DitherRasterizer")
            == EngraverSettingsWidget
        )
        assert (
            step_widget_registry.get("FrameProducer")
            == FrameProducerSettingsWidget
        )
        assert (
            step_widget_registry.get("MaterialTestGridProducer")
            == MaterialTestGridSettingsWidget
        )
        assert (
            step_widget_registry.get("ShrinkWrapProducer")
            == ShrinkWrapProducerSettingsWidget
        )

    def test_global_registry_has_transformer_widgets(self):
        assert (
            step_widget_registry.get("MultiPassTransformer")
            == MultiPassSettingsWidget
        )
        assert step_widget_registry.get("Optimize") == OptimizeSettingsWidget
        assert (
            step_widget_registry.get("OverscanTransformer")
            == OverscanSettingsWidget
        )
        assert step_widget_registry.get("Smooth") == SmoothSettingsWidget
