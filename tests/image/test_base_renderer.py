from unittest.mock import MagicMock
from rayforge.image.base_renderer import RenderSpecification, Renderer


# --- Test Fixtures and Helpers ---
class ConcreteRenderer(Renderer):
    """A minimal concrete implementation of Renderer for testing the ABC."""

    def render_base_image(self, data, width, height, **kwargs):
        # This implementation is not needed for the tests in this suite.
        pass


# --- Test Suite ---
class TestRenderSpecification:
    """Tests for the RenderSpecification dataclass."""

    def test_initialization_with_defaults(self):
        """
        Tests that RenderSpecification initializes with correct default values
        for optional fields.
        """
        spec = RenderSpecification(width=100, height=50, data=b"test data")

        assert spec.width == 100
        assert spec.height == 50
        assert spec.data == b"test data"
        assert spec.kwargs == {}  # Should default to an empty dict
        assert spec.crop_rect is None  # Should default to None
        assert spec.apply_mask is True  # Should default to True

    def test_initialization_with_explicit_values(self):
        """
        Tests that RenderSpecification initializes correctly when all values
        are explicitly provided.
        """
        custom_kwargs = {"foo": "bar", "quality": 95}
        crop_rect = (10, 20, 30, 40)

        spec = RenderSpecification(
            width=200,
            height=150,
            data=b"explicit data",
            kwargs=custom_kwargs,
            crop_rect=crop_rect,
            apply_mask=False,
        )

        assert spec.width == 200
        assert spec.height == 150
        assert spec.data == b"explicit data"
        assert spec.kwargs == {"foo": "bar", "quality": 95}
        assert spec.crop_rect == (10, 20, 30, 40)
        assert spec.apply_mask is False


class TestRendererABC:
    """Tests for the Renderer abstract base class."""

    def test_default_compute_render_spec(self):
        """
        Tests the default implementation of Renderer.compute_render_spec to
        ensure it provides a simple, pass-through specification.
        """
        renderer = ConcreteRenderer()

        # Mock the complex inputs required by the method signature.
        # For the default implementation, their internal values don't matter.
        mock_segment = MagicMock(name="SourceAssetSegment")
        mock_context = MagicMock(name="RenderContext")
        mock_context.data = b"source data bytes"

        target_size = (800, 600)

        # Call the method under test
        spec = renderer.compute_render_spec(
            segment=mock_segment,
            target_size=target_size,
            source_context=mock_context,
        )

        # Verify the returned spec matches the default pass-through behavior
        assert isinstance(spec, RenderSpecification)
        assert spec.width == 800
        assert spec.height == 600
        assert spec.data == b"source data bytes"
        assert spec.apply_mask is True
        assert spec.crop_rect is None
        assert spec.kwargs == {}
