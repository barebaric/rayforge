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

    @pytest.mark.ui
    @pytest.mark.usefixtures("ui_context_initializer")
    def test_global_registry_has_producer_widgets(self):
        widget = step_widget_registry.get("ContourProducer")
        assert widget is not None
        assert widget.__name__ == "ContourProducerSettingsWidget"

        widget = step_widget_registry.get("Rasterizer")
        assert widget is not None
        assert widget.__name__ == "RasterSettingsWidget"

        widget = step_widget_registry.get("DepthEngraver")
        assert widget is not None
        assert widget.__name__ == "RasterSettingsWidget"

        widget = step_widget_registry.get("DitherRasterizer")
        assert widget is not None
        assert widget.__name__ == "RasterSettingsWidget"

        widget = step_widget_registry.get("FrameProducer")
        assert widget is not None
        assert widget.__name__ == "FrameProducerSettingsWidget"

        widget = step_widget_registry.get("MaterialTestGridProducer")
        assert widget is not None
        assert widget.__name__ == "MaterialTestGridSettingsWidget"

        widget = step_widget_registry.get("ShrinkWrapProducer")
        assert widget is not None
        assert widget.__name__ == "ShrinkWrapProducerSettingsWidget"
