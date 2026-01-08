import cairo
import pytest
from pathlib import Path
from typing import cast, Tuple
from unittest.mock import Mock
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset import SourceAsset
from rayforge.core.matrix import Matrix
from rayforge.image.jpg.importer import JpgImporter
from rayforge.image.jpg.renderer import JPG_RENDERER
from rayforge.image import renderer_by_name
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.geo import Geometry

# Assume the test JPGs are in the same directory as this test file
TEST_DATA_DIR = Path(__file__).parent


def load_jpg_data(filename: str) -> bytes:
    """Helper to load a JPG file from the test data directory."""
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
    importer: JpgImporter, vectorization_spec=None
) -> WorkPiece:
    """Helper to run importer and correctly link workpiece to its source."""
    payload = importer.get_doc_items(vectorization_spec=vectorization_spec)
    assert payload is not None and payload.items, (
        "Importer failed to produce a workpiece. Image might be invalid."
    )
    source = payload.source
    wp = cast(WorkPiece, payload.items[0])

    mock_doc = Mock()
    mock_doc.source_assets = {source.uid: source}
    mock_doc.get_source_asset_by_uid.side_effect = mock_doc.source_assets.get

    mock_parent = Mock()
    mock_parent.doc = mock_doc
    mock_parent.get_world_transform.return_value = Matrix.identity()
    wp.parent = mock_parent

    return wp


@pytest.fixture
def color_jpg_data() -> bytes:
    """Fixture for a standard color JPG."""
    return load_jpg_data("color.jpg")


@pytest.fixture
def color_workpiece(color_jpg_data: bytes) -> WorkPiece:
    """A WorkPiece created from the color JPG data."""
    importer = JpgImporter(color_jpg_data)
    return _setup_workpiece_with_context(
        importer, vectorization_spec=TraceSpec()
    )


class TestJpgImporter:
    @pytest.mark.parametrize(
        "jpg_data_fixture, px_width, px_height",
        [
            ("color_jpg_data", 259, 194),
        ],
    )
    def test_importer_creates_workpiece_with_correct_size(
        self,
        jpg_data_fixture,
        px_width,
        px_height,
        request,
    ):
        """
        Tests that the importer creates a WorkPiece and correctly calculates
        its physical size in mm based on pixel dimensions and resolution.
        """
        jpg_data = request.getfixturevalue(jpg_data_fixture)
        importer = JpgImporter(jpg_data)
        payload = importer.get_doc_items(vectorization_spec=TraceSpec())

        assert payload and payload.items and len(payload.items) == 1
        assert isinstance(payload.source, SourceAsset)

        wp = payload.items[0]
        assert isinstance(wp, WorkPiece)
        assert wp.source_segment is not None
        assert wp.source_segment.source_asset_uid == payload.source.uid
        assert payload.source.original_data == jpg_data

        # Verify that metadata was collected and attached
        assert payload.source.metadata is not None
        assert payload.source.metadata["image_format"] == "JPEG"
        assert payload.source.metadata["width"] == px_width

        # Verify physical size calculation (assuming 96 DPI default)
        expected_width_mm = px_width * (25.4 / 96.0)
        expected_height_mm = px_height * (25.4 / 96.0)

        assert wp.size[0] == pytest.approx(expected_width_mm, 5)
        assert wp.size[1] == pytest.approx(expected_height_mm, 5)

    def test_importer_requires_vectorization_spec(self, color_jpg_data: bytes):
        """Importer returns None if no vectorization_spec is provided."""
        importer = JpgImporter(color_jpg_data)
        payload = importer.get_doc_items(vectorization_spec=None)
        assert payload is None

    def test_importer_handles_invalid_data(self):
        """Tests the importer returns None for invalid JPG data."""
        importer = JpgImporter(b"this is not a jpg")
        payload = importer.get_doc_items(vectorization_spec=TraceSpec())
        assert payload is None

    def test_source_asset_serialization_with_metadata(self):
        """Checks that metadata is correctly serialized and deserialized."""
        metadata = {
            "image_format": "JPEG",
            "width": 100,
            "jpeg-chroma-subsample": "4:4:4",
        }
        source = SourceAsset(
            source_file=Path("test.jpg"),
            original_data=b"dummy",
            renderer=JPG_RENDERER,
            metadata=metadata,
        )

        # Serialize to dict
        state = source.to_dict()
        assert "metadata" in state
        assert state["metadata"] == metadata
        assert state["renderer_name"] == "JpgRenderer"

        # Check that the real renderer is in the registry for from_dict to work
        assert "JpgRenderer" in renderer_by_name

        # Deserialize from dict
        restored_source = SourceAsset.from_dict(state)
        assert restored_source.metadata == metadata
        assert restored_source.renderer is JPG_RENDERER


class TestJpgRenderer:
    def test_get_natural_size(self, color_workpiece: WorkPiece):
        """Test natural size calculation on the renderer."""
        size = color_workpiece.natural_size
        assert size is not None
        width_mm, height_mm = size
        expected_width_mm = 300 * (25.4 / 96.0)
        assert width_mm == pytest.approx(expected_width_mm, 5)

    def test_render_to_pixels(self, color_workpiece: WorkPiece):
        """Test rendering to a Cairo surface."""
        surface = color_workpiece.render_to_pixels(width=150, height=179)
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 150

    def test_renderer_correct_colors_for_color_image(
        self, color_workpiece: WorkPiece
    ):
        """
        Checks that colors are rendered correctly by sampling a known pixel.
        """
        surface = color_workpiece.render_to_pixels(width=300, height=358)
        assert surface is not None
        # Sample a known blue pixel from the test image
        b, g, r, a = get_pixel_bgra(surface, x=150, y=50)
        # JPEG compression is lossy, so check for approximate values
        assert r == pytest.approx(107, 1)
        assert g == pytest.approx(180, 1)
        assert b == pytest.approx(65, 1)
        assert a == 255  # JPGs are always opaque

    def test_renderer_handles_invalid_data_gracefully(self):
        """
        Test that the renderer does not raise exceptions for invalid data.
        """
        source = SourceAsset(
            source_file=Path("nonexistent.jpg"),
            original_data=b"invalid data",
            renderer=JPG_RENDERER,
        )
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            pristine_geometry=Geometry(),
            vectorization_spec=TraceSpec(),
        )
        invalid_wp = WorkPiece(name="invalid", source_segment=gen_config)

        mock_doc = Mock()
        mock_doc.source_assets = {source.uid: source}
        mock_doc.get_source_asset_by_uid.side_effect = (
            mock_doc.source_assets.get
        )
        mock_parent = Mock()
        mock_parent.doc = mock_doc
        mock_parent.get_world_transform.return_value = Matrix.identity()
        invalid_wp.parent = mock_parent

        assert invalid_wp.natural_size == (0.0, 0.0)
        assert invalid_wp.render_to_pixels(100, 100) is None

        chunks = list(
            invalid_wp.render_chunk(
                pixels_per_mm_x=1.0,
                pixels_per_mm_y=1.0,
                max_chunk_width=50,
            )
        )
        assert len(chunks) == 0
