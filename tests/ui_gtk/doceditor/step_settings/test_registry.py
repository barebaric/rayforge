import pytest

from rayforge.ui_gtk.doceditor.step_settings.registry import (
    StepWidgetRegistry,
    step_widget_registry,
)
from rayforge.ui_gtk.doceditor.step_settings import (
    MultiPassSettingsWidget,
    OptimizeSettingsWidget,
    OverscanSettingsWidget,
    PlaceholderSettingsWidget,
    SmoothSettingsWidget,
)


class MockProducer:
    """Mock producer for testing."""

    pass


class MockWidget:
    """Mock widget for testing."""

    pass


class TestStepWidgetRegistry:
    def test_register_and_get(self):
        registry = StepWidgetRegistry()
        registry.register(MockProducer, MockWidget)
        assert registry.get("MockProducer") == MockWidget

    def test_get_unknown_returns_none(self):
        registry = StepWidgetRegistry()
        assert registry.get("NonExistentWidget") is None

    def test_all_widgets_returns_copy(self):
        registry = StepWidgetRegistry()
        registry.register(MockProducer, MockWidget)
        all_widgets = registry.all_widgets()
        assert all_widgets == {"MockProducer": MockWidget}
        all_widgets["NewOne"] = MockWidget
        assert "NewOne" not in registry.all_widgets()

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
        assert (
            step_widget_registry.get("PlaceholderProducer")
            == PlaceholderSettingsWidget
        )


class TestAddonWidgets:
    """Tests that require the full context (addon loading)."""

    @pytest.mark.usefixtures("context_initializer")
    def test_global_registry_has_producer_widgets(self):
        import importlib

        widgets = importlib.import_module(
            "rayforge_plugins.laser_essentials.laser_essentials.widgets"
        )

        assert (
            step_widget_registry.get("ContourProducer")
            == widgets.ContourProducerSettingsWidget
        )
        assert (
            step_widget_registry.get("Rasterizer")
            == widgets.EngraverSettingsWidget
        )
        assert (
            step_widget_registry.get("DepthEngraver")
            == widgets.EngraverSettingsWidget
        )
        assert (
            step_widget_registry.get("DitherRasterizer")
            == widgets.EngraverSettingsWidget
        )
        assert (
            step_widget_registry.get("FrameProducer")
            == widgets.FrameProducerSettingsWidget
        )
        assert (
            step_widget_registry.get("MaterialTestGridProducer")
            == widgets.MaterialTestGridSettingsWidget
        )
        assert (
            step_widget_registry.get("ShrinkWrapProducer")
            == widgets.ShrinkWrapProducerSettingsWidget
        )
