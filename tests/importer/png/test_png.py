import cairo
import pytest
from pathlib import Path
from typing import cast, Tuple
from unittest.mock import Mock
from rayforge.core.vectorization_config import TraceConfig
from rayforge.core.workpiece import WorkPiece
from rayforge.core.import_source import ImportSource
from rayforge.core.matrix import Matrix
from rayforge.importer.png.importer import PngImporter
from rayforge.importer.png.renderer import PNG_RENDERER
from rayforge.importer import renderer_by_name

# Assume the test PNGs are in the same directory as this test file
TEST_DATA_DIR = Path(__file__).parent


def load_png_data(filename: str) -> bytes:
    """Helper to load a PNG file from the test data directory."""
    file_path = TEST_DATA_DIR / filename
    assert file_path.exists(), f"Test file not found: {file_path}"
    return file_path.read_bytes()


def get_pixel_bgra(
    surface: cairo.ImageSurface, x: int, y: int
) -> Tuple[int, int, int, int]:
    """
    Samples a single pixel from a Cairo surface and returns its BGRA values.
    Cairo's FORMAT_ARGB32 is stored in memory as BGRA on little-endian systems.
    """
    stride = surface.get_stride()
    data = surface.get_data()
    offset = y * stride + x * 4
    # The bytes are in order B, G, R, A in the buffer
    b, g, r, a = data[offset : offset + 4]
    return b, g, r, a


def _setup_workpiece_with_context(
    importer: PngImporter, vector_config=None
) -> WorkPiece:
    """Helper to run importer and correctly link workpiece to its source."""
    payload = importer.get_doc_items(vector_config=vector_config)
    assert payload is not None and payload.items, (
        "Importer failed to produce a workpiece. Surface was likely blank."
    )
    source = payload.source
    wp = cast(WorkPiece, payload.items[0])

    mock_doc = Mock()
    mock_doc.import_sources = {source.uid: source}
    mock_doc.get_import_source_by_uid.side_effect = mock_doc.import_sources.get

    mock_parent = Mock()
    mock_parent.doc = mock_doc
    mock_parent.get_world_transform.return_value = Matrix.identity()
    wp.parent = mock_parent

    return wp


@pytest.fixture
def bilevel_png_data() -> bytes:
    """Fixture for the RGBA PNG that has bilevel characteristics."""
    return load_png_data("8-bit-with-1-bit-color.png")


@pytest.fixture
def color_png_data() -> bytes:
    """Fixture for a standard 8-bit/color RGBA PNG."""
    return load_png_data("color.png")


@pytest.fixture
def grayscale_png_data() -> bytes:
    """Fixture for an 8-bit grayscale PNG with an alpha channel."""
    return load_png_data("grayscale.png")


@pytest.fixture
def bilevel_workpiece(bilevel_png_data: bytes) -> WorkPiece:
    """A WorkPiece created from the bilevel PNG data."""
    importer = PngImporter(bilevel_png_data)
    return _setup_workpiece_with_context(importer, vector_config=TraceConfig())


@pytest.fixture
def color_workpiece(color_png_data: bytes) -> WorkPiece:
    """A WorkPiece created from the color PNG data."""
    importer = PngImporter(color_png_data)
    return _setup_workpiece_with_context(importer, vector_config=TraceConfig())


@pytest.fixture
def grayscale_workpiece(grayscale_png_data: bytes) -> WorkPiece:
    """A WorkPiece created from the grayscale PNG data."""
    importer = PngImporter(grayscale_png_data)
    return _setup_workpiece_with_context(importer, vector_config=TraceConfig())


class TestPngImporter:
    @pytest.mark.parametrize(
        "png_data_fixture, px_width, px_height",
        [
            ("bilevel_png_data", 243, 31),
            ("color_png_data", 300, 358),
            ("grayscale_png_data", 300, 358),
        ],
    )
    def test_importer_creates_workpiece_with_correct_size(
        self,
        png_data_fixture,
        px_width,
        px_height,
        request,
    ):
        """
        Tests that the importer creates a WorkPiece and correctly calculates
        its physical size in mm based on pixel dimensions and resolution.
        """
        png_data = request.getfixturevalue(png_data_fixture)
        importer = PngImporter(png_data)
        payload = importer.get_doc_items(vector_config=TraceConfig())

        assert payload and payload.items and len(payload.items) == 1
        assert isinstance(payload.source, ImportSource)

        wp = payload.items[0]
        assert isinstance(wp, WorkPiece)
        assert wp.import_source_uid == payload.source.uid
        assert payload.source.data == png_data

        # Verify that metadata was collected and attached (integration check)
        assert payload.source.metadata is not None
        assert payload.source.metadata["image_format"] == "PNG"
        assert payload.source.metadata["width"] == px_width

        # Verify physical size calculation
        expected_width_mm = px_width * (25.4 / 96.0)
        expected_height_mm = px_height * (25.4 / 96.0)

        assert wp.size[0] == pytest.approx(expected_width_mm, 5)
        assert wp.size[1] == pytest.approx(expected_height_mm, 5)

    def test_importer_requires_vector_config(self, color_png_data: bytes):
        """Importer returns None if no vector_config is provided."""
        importer = PngImporter(color_png_data)
        payload = importer.get_doc_items(vector_config=None)
        assert payload is None

    def test_importer_handles_invalid_data(self):
        """Tests the importer returns None for invalid PNG data."""
        importer = PngImporter(b"this is not a png")
        payload = importer.get_doc_items(vector_config=TraceConfig())
        assert payload is None

    def test_import_source_serialization_with_metadata(self):
        """Checks that metadata is correctly serialized and deserialized."""
        metadata = {
            "image_format": "PNG",
            "width": 100,
        }
        source = ImportSource(
            source_file=Path("test.png"),
            original_data=b"dummy",
            renderer=PNG_RENDERER,
            metadata=metadata,
        )

        # Serialize to dict
        state = source.to_dict()
        assert "metadata" in state
        assert state["metadata"] == metadata
        assert state["renderer_name"] == "PngRenderer"

        # Check that the real renderer is in the registry for from_dict to work
        assert "PngRenderer" in renderer_by_name

        # Deserialize from dict
        restored_source = ImportSource.from_dict(state)
        assert restored_source.metadata == metadata
        assert restored_source.renderer is PNG_RENDERER


class TestPngRenderer:
    def test_get_natural_size(self, bilevel_workpiece: WorkPiece):
        """Test natural size calculation on the renderer."""
        size = PNG_RENDERER.get_natural_size(bilevel_workpiece)
        assert size is not None
        width_mm, height_mm = size
        expected_width_mm = 243 * (25.4 / 96.0)
        assert width_mm == pytest.approx(expected_width_mm, 5)

    def test_render_to_pixels(self, bilevel_workpiece: WorkPiece):
        """Test rendering to a Cairo surface."""
        surface = PNG_RENDERER.render_to_pixels(
            bilevel_workpiece, width=200, height=26
        )
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 200

    def test_renderer_correct_colors_for_color_image(
        self, color_workpiece: WorkPiece
    ):
        """
        Checks for the R/B swap and red tint bugs by sampling known pixel
        colors. This implicitly tests that the premultiply path is taken.
        """
        surface = PNG_RENDERER.render_to_pixels(
            color_workpiece, width=300, height=358
        )
        assert surface is not None
        b, g, r, a = get_pixel_bgra(surface, x=150, y=50)
        assert (r, g, b, a) == (136, 189, 245, 255)

    def test_renderer_correct_rendering_for_bilevel_image(
        self, bilevel_workpiece: WorkPiece
    ):
        """
        Checks that the bilevel image is not blank and preserves transparency.
        This implicitly tests that the non-premultiply path is taken.
        """
        surface = PNG_RENDERER.render_to_pixels(
            bilevel_workpiece, width=243, height=31
        )
        assert surface is not None

        # Sample a pixel that should be part of the black text (opaque)
        b, g, r, a = get_pixel_bgra(surface, x=20, y=20)
        assert (r, g, b, a) == (0, 0, 0, 255)

        # Sample a pixel that should be in the background (transparent)
        b, g, r, a = get_pixel_bgra(surface, x=10, y=10)
        assert (r, g, b, a) == (0, 0, 0, 0), (
            "Background pixel should be fully transparent"
        )

    def test_renderer_correct_rendering_for_grayscale_image(
        self, grayscale_workpiece: WorkPiece
    ):
        """
        Checks that a grayscale image renders with R=G=B, not with a color
        tint.
        """
        surface = PNG_RENDERER.render_to_pixels(
            grayscale_workpiece, width=300, height=358
        )
        assert surface is not None

        # Sample a known gray pixel from the image
        b, g, r, a = get_pixel_bgra(surface, x=150, y=150)
        assert (r, g, b, a) == (22, 22, 22, 255)

    def test_renderer_handles_invalid_data_gracefully(self):
        """
        Test that the renderer does not raise exceptions for invalid data.
        """
        source = ImportSource(
            source_file=Path("nonexistent"),
            original_data=b"invalid data",
            renderer=PNG_RENDERER,
        )
        invalid_wp = WorkPiece(name="invalid")
        invalid_wp.import_source_uid = source.uid

        mock_doc = Mock()
        mock_doc.import_sources = {source.uid: source}
        mock_doc.get_import_source_by_uid.side_effect = (
            mock_doc.import_sources.get
        )
        mock_parent = Mock()
        mock_parent.doc = mock_doc
        mock_parent.get_world_transform.return_value = Matrix.identity()
        invalid_wp.parent = mock_parent

        assert PNG_RENDERER.get_natural_size(invalid_wp) is None
        assert PNG_RENDERER.render_to_pixels(invalid_wp, 100, 100) is None

        chunks = list(
            invalid_wp.render_chunk(
                pixels_per_mm_x=1.0,
                pixels_per_mm_y=1.0,
                max_chunk_width=50,
            )
        )
        assert len(chunks) == 0
