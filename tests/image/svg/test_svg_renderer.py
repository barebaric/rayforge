from unittest.mock import MagicMock

import pytest
from raygeo.svg.color import ColorAttr

from rayforge.core.vectorization_spec import (
    LayerSource,
    PassthroughSpec,
    TraceSpec,
)
from rayforge.image.svg.renderer import SvgRenderer


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

    def test_compute_spec_color_layer_skips_filter(
        self, svg_renderer, mock_render_context
    ):
        """
        Color-layer segments carry a color key (e.g. '#ff0000') as their
        layer id. Instead of filtering by group, the render filters by
        color so the base image shows only that color's content.
        """
        mock_segment = MagicMock()
        mock_segment.layer_id = "#ff0000"
        mock_segment.crop_window_px = None
        mock_segment.vectorization_spec = PassthroughSpec(
            layer_source=LayerSource.COLORS
        )

        spec = svg_renderer.compute_render_spec(
            mock_segment, (200, 150), mock_render_context
        )

        assert spec.kwargs == {
            "color_key": "#ff0000",
            "color_attr": ColorAttr.ANY,
        }
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

    def test_render_base_image_filters_by_color(
        self, svg_renderer, mock_render_context
    ):
        """
        Color-layer renders exclude shapes of other colors from the base
        image.
        """
        svg = b"""
        <svg width="100mm" height="100mm" viewBox="0 0 100 100"
             xmlns="http://www.w3.org/2000/svg">
            <rect x="0" y="0" width="10" height="10" fill="#ff0000"/>
            <rect x="50" y="50" width="10" height="10" fill="#00ff00"/>
        </svg>
        """
        red = svg_renderer.render_base_image(
            svg, 100, 100, color_key="#ff0000", color_attr=ColorAttr.ANY
        )
        assert red is not None
        green = svg_renderer.render_base_image(
            svg, 100, 100, color_key="#00ff00", color_attr=ColorAttr.ANY
        )
        assert green is not None

        # Each render contains only its own color.
        def has_color(image, expected_rgb):
            arr = image.numpy()[..., :3].astype(int)
            r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
            er, eg, eb = expected_rgb
            match = (
                (abs(r - er) < 40) & (abs(g - eg) < 40) & (abs(b - eb) < 40)
            )
            return int(match.sum())

        assert has_color(red, (255, 0, 0)) > 0
        assert has_color(red, (0, 255, 0)) == 0
        assert has_color(green, (0, 255, 0)) > 0
        assert has_color(green, (255, 0, 0)) == 0
