import pytest
from unittest.mock import MagicMock
from rayforge.image.svg.renderer import SvgRenderer
from rayforge.core.vectorization_spec import PassthroughSpec, TraceSpec


@pytest.fixture
def svg_renderer() -> SvgRenderer:
    """Provides an instance of the SvgRenderer for testing."""
    return SvgRenderer()


@pytest.fixture
def mock_render_context() -> MagicMock:
    """Provides a default mock render context."""
    mock_context = MagicMock(name="RenderContext")
    mock_context.data = b"<svg></svg>"
    return mock_context


class TestSvgRenderer:
    def test_compute_spec_standard_render(
        self, svg_renderer, mock_render_context
    ):
        """
        Tests a standard vector render with no special attributes.
        The spec should be a simple passthrough.
        """
        mock_segment = MagicMock()
        mock_segment.layer_id = None
        mock_segment.crop_window_px = None
        mock_segment.vectorization_spec = PassthroughSpec()

        spec = svg_renderer.compute_render_spec(
            mock_segment, (100, 50), mock_render_context
        )

        assert spec.width == 100
        assert spec.height == 50
        assert spec.data == mock_render_context.data
        assert spec.kwargs == {}
        assert spec.apply_mask is False

    def test_compute_spec_with_layer_visibility(
        self, svg_renderer, mock_render_context
    ):
        """
        Tests that when a layer_id is present, it's added to the kwargs
        for the renderer.
        """
        mock_segment = MagicMock()
        mock_segment.layer_id = "layer_123"
        mock_segment.crop_window_px = None
        mock_segment.vectorization_spec = PassthroughSpec()

        spec = svg_renderer.compute_render_spec(
            mock_segment, (200, 150), mock_render_context
        )

        assert spec.kwargs == {"visible_layer_ids": ["layer_123"]}
        assert spec.apply_mask is False

    def test_compute_spec_with_vector_crop_window(
        self, svg_renderer, mock_render_context
    ):
        """
        Tests that a crop_window_px with a PassthroughSpec results in a
        'viewbox' kwarg being added for vector cropping.
        """
        mock_segment = MagicMock()
        mock_segment.layer_id = None
        mock_segment.crop_window_px = (10.0, 20.0, 30.0, 40.0)
        mock_segment.vectorization_spec = PassthroughSpec()

        spec = svg_renderer.compute_render_spec(
            mock_segment, (300, 400), mock_render_context
        )

        assert spec.kwargs == {"viewbox": (10.0, 20.0, 30.0, 40.0)}
        assert spec.apply_mask is False

    def test_compute_spec_with_trace_crop_window(
        self, svg_renderer, mock_render_context
    ):
        """
        Tests that a crop_window_px with a TraceSpec is IGNORED. The renderer
        should not add a 'viewbox' kwarg, because the full SVG needs to be
        rendered for tracing.
        """
        mock_segment = MagicMock()
        mock_segment.layer_id = None
        mock_segment.crop_window_px = (10.0, 20.0, 30.0, 40.0)
        # Key difference: TraceSpec
        mock_segment.vectorization_spec = TraceSpec()

        spec = svg_renderer.compute_render_spec(
            mock_segment, (300, 400), mock_render_context
        )

        # The viewbox should NOT be present.
        assert spec.kwargs == {}
        # Since this is a trace, it is effectively a raster render, so the
        # mask should be applied. However, SvgRenderer always returns False.
        # This is an acceptable inconsistency for now, as the final switchover
        # might use a different renderer for the trace path.
        assert spec.apply_mask is False
