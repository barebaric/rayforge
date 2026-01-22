import pytest
from unittest.mock import MagicMock
from rayforge.image.base_renderer import RasterRenderer, RenderSpecification
from rayforge.core.vectorization_spec import TraceSpec, PassthroughSpec


class ConcreteRasterRenderer(RasterRenderer):
    """A concrete class for testing the abstract RasterRenderer."""

    def render_base_image(self, data, width, height, **kwargs):
        """Minimal implementation to satisfy the ABC."""
        pass


@pytest.fixture
def raster_renderer() -> RasterRenderer:
    """Provides an instance of a concrete RasterRenderer for testing."""
    return ConcreteRasterRenderer()


class TestRasterRenderer:
    def test_compute_spec_for_uncropped_image(self, raster_renderer):
        """
        Tests that when no crop window is specified, the renderer returns a
        simple, direct render specification.
        """
        mock_segment = MagicMock()
        mock_segment.crop_window_px = None
        mock_segment.vectorization_spec = TraceSpec()  # Typical for rasters

        mock_context = MagicMock()
        mock_context.data = b"render_data"
        mock_context.original_data = b"original_data"
        mock_context.source_pixel_dims = (1000, 800)

        target_size = (400, 320)

        spec = raster_renderer.compute_render_spec(
            mock_segment, target_size, mock_context
        )

        assert isinstance(spec, RenderSpecification)
        assert spec.width == 400
        assert spec.height == 320
        assert spec.data == b"render_data"  # Should use the processed data
        assert spec.crop_rect is None
        assert spec.apply_mask is True

    def test_compute_spec_for_cropped_image(self, raster_renderer):
        """
        Tests the core upscale-and-crop logic for a cropped raster image.
        The returned spec should request a larger render of the original data
        and provide a crop rectangle.
        """
        mock_segment = MagicMock()
        # Crop window: (x, y, width, height) in original pixel coords
        mock_segment.crop_window_px = (100.0, 150.0, 200.0, 100.0)
        mock_segment.vectorization_spec = TraceSpec()

        mock_context = MagicMock()
        mock_context.data = b"render_data"
        mock_context.original_data = b"original_data"
        # Full, original image dimensions
        mock_context.source_pixel_dims = (1000, 800)

        # The final desired size of the cropped area
        target_size = (400, 200)

        spec = raster_renderer.compute_render_spec(
            mock_segment, target_size, mock_context
        )

        # Verification
        assert isinstance(spec, RenderSpecification)

        # The render should be of the full original image, but upscaled.
        # Scale factor = target_size / crop_window_size
        # scale_x = 400 / 200 = 2.0
        # scale_y = 200 / 100 = 2.0
        # Expected render size = original_dims * scale_factor
        # expected_width = 1000 * 2.0 = 2000
        # expected_height = 800 * 2.0 = 1600
        assert spec.width == 2000
        assert spec.height == 1600

        # The render data must be the original, uncropped data.
        assert spec.data == b"original_data"

        # The crop rectangle should be scaled by the same factor.
        # expected_crop_x = 100 * 2.0 = 200
        # expected_crop_y = 150 * 2.0 = 300
        # expected_crop_w = 200 * 2.0 = 400
        # expected_crop_h = 100 * 2.0 = 200
        assert spec.crop_rect == (200, 300, 400, 200)
        assert spec.apply_mask is True

    def test_compute_spec_for_vector_passthrough_is_skipped(
        self, raster_renderer
    ):
        """
        Tests that if a vector spec is present, the raster upscale logic is
        skipped, even if a crop window exists. This logic will be handled
        by the SvgRenderer.
        """
        mock_segment = MagicMock()
        mock_segment.crop_window_px = (100.0, 150.0, 200.0, 100.0)
        # This is the key: PassthroughSpec makes it a "vector"
        mock_segment.vectorization_spec = PassthroughSpec()

        mock_context = MagicMock()
        mock_context.data = b"render_data"
        mock_context.original_data = b"original_data"
        mock_context.source_pixel_dims = (1000, 800)

        target_size = (400, 200)

        spec = raster_renderer.compute_render_spec(
            mock_segment, target_size, mock_context
        )

        # Should return a simple passthrough spec because is_vector is true
        assert spec.width == 400
        assert spec.height == 200
        assert spec.data == b"render_data"
        assert spec.crop_rect is None
