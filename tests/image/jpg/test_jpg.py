import cairo
import pytest
from pathlib import Path
from typing import cast, Tuple
from unittest.mock import Mock
from rayforge.core.vectorization_spec import TraceSpec, PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset import SourceAsset
from rayforge.core.matrix import Matrix
from rayforge.image.jpg.importer import JpgImporter
from rayforge.image.jpg.renderer import JPG_RENDERER
from rayforge.image import renderer_by_name
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.geo import Geometry
from rayforge.image.base_importer import ImporterFeature

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
    import_result = importer.get_doc_items(
        vectorization_spec=vectorization_spec
    )
    assert import_result is not None, "Importer returned None"
    payload = import_result.payload
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


class TestJpgImporterContract:
    """Tests for the Importer contract compliance of JpgImporter."""

    def test_class_attributes(self):
        """Tests that importer class has required attributes."""
        assert JpgImporter.label == "JPEG files"
        assert JpgImporter.mime_types == ("image/jpeg",)
        assert JpgImporter.extensions == (".jpg", ".jpeg")
        assert JpgImporter.features == {ImporterFeature.BITMAP_TRACING}

    def test_scan_returns_manifest(self, color_jpg_data: bytes):
        """Tests that scan() returns ImportManifest with correct data."""
        importer = JpgImporter(color_jpg_data, Path("test.jpg"))
        manifest = importer.scan()

        assert manifest.title == "test.jpg"
        assert manifest.natural_size_mm is not None
        width_mm, height_mm = manifest.natural_size_mm
        assert width_mm > 0
        assert height_mm > 0
        assert len(manifest.warnings) == 0
        assert len(manifest.errors) == 0

    def test_scan_handles_invalid_data(self):
        """Tests that scan() handles invalid JPEG data gracefully."""
        importer = JpgImporter(b"not a jpg", Path("invalid.jpg"))
        manifest = importer.scan()

        assert manifest.title == "invalid.jpg"
        assert manifest.natural_size_mm is None
        assert len(manifest.errors) > 0

    def test_parse_returns_parsing_result(self, color_jpg_data: bytes):
        """Tests that parse() returns ParsingResult with correct data."""
        importer = JpgImporter(color_jpg_data)
        parse_result = importer.parse()

        assert parse_result is not None
        x, y, width, height = parse_result.document_bounds
        assert x == 0.0
        assert y == 0.0
        assert width == 259.0
        assert height == 194.0
        assert parse_result.is_y_down is True
        assert parse_result.native_unit_to_mm == pytest.approx(25.4 / 72.0)
        assert len(parse_result.layers) == 1
        assert parse_result.layers[0].layer_id == "__default__"
        assert parse_result.layers[0].name == "__default__"
        assert parse_result.world_frame_of_reference is not None
        assert parse_result.background_world_transform is not None

    def test_parse_handles_invalid_data(self):
        """Tests that parse() returns None for invalid JPEG data."""
        importer = JpgImporter(b"not a jpg")
        parse_result = importer.parse()

        assert parse_result is None
        assert len(importer._errors) > 0

    def test_vectorize_returns_vectorization_result(
        self, color_jpg_data: bytes
    ):
        """Tests that vectorize() returns VectorizationResult."""
        importer = JpgImporter(color_jpg_data)
        parse_result = importer.parse()
        assert parse_result is not None

        spec = TraceSpec()
        vec_result = importer.vectorize(parse_result, spec)

        assert vec_result is not None
        assert vec_result.source_parse_result is parse_result
        assert None in vec_result.geometries_by_layer
        assert isinstance(vec_result.geometries_by_layer[None], Geometry)

    def test_vectorize_raises_type_error_for_wrong_spec(
        self, color_jpg_data: bytes
    ):
        """Tests that vectorize() raises TypeError for non-TraceSpec."""
        importer = JpgImporter(color_jpg_data)
        parse_result = importer.parse()
        assert parse_result is not None

        with pytest.raises(TypeError):
            importer.vectorize(parse_result, PassthroughSpec())

    def test_create_source_asset(self, color_jpg_data: bytes):
        """Tests that create_source_asset() returns SourceAsset."""
        importer = JpgImporter(color_jpg_data)
        parse_result = importer.parse()
        assert parse_result is not None

        source_asset = importer.create_source_asset(parse_result)

        assert isinstance(source_asset, SourceAsset)
        assert source_asset.renderer is JPG_RENDERER
        assert source_asset.original_data == color_jpg_data
        assert source_asset.width_px == 259
        assert source_asset.height_px == 194
        assert source_asset.metadata is not None
        assert source_asset.metadata["image_format"] == "JPEG"


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
        import_result = importer.get_doc_items(vectorization_spec=TraceSpec())
        assert import_result is not None
        payload = import_result.payload

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
        """Importer raises TypeError if wrong spec is provided."""
        importer = JpgImporter(color_jpg_data)
        with pytest.raises(TypeError):
            # Explicitly pass an invalid spec type to trigger the error.
            importer.get_doc_items(vectorization_spec=PassthroughSpec())

    def test_importer_handles_invalid_data(self):
        """Tests importer returns ImportResult with None payload on invalid."""
        importer = JpgImporter(b"this is not a jpg")
        import_result = importer.get_doc_items(vectorization_spec=TraceSpec())
        assert import_result is not None
        assert import_result.payload is None
        assert import_result.parse_result is None
        assert len(import_result.errors) > 0

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
        expected_width_mm = 259 * (25.4 / 96.0)
        assert width_mm == pytest.approx(expected_width_mm, 5)

    def test_render_to_pixels(self, color_workpiece: WorkPiece):
        """Test rendering to a Cairo surface."""
        surface = color_workpiece.render_to_pixels(width=150, height=112)
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 150

    def test_renderer_correct_colors_for_color_image(
        self, color_workpiece: WorkPiece
    ):
        """
        Checks that colors are rendered correctly by sampling a known pixel.
        """
        surface = color_workpiece.render_to_pixels(width=259, height=194)
        assert surface is not None
        # Sample a known blue pixel from the test image
        b, g, r, a = get_pixel_bgra(surface, x=150, y=50)
        # JPEG compression is lossy, so check for approximate values
        assert r == pytest.approx(107, abs=5)
        assert g == pytest.approx(180, abs=5)
        assert b == pytest.approx(65, abs=5)
        # Alpha channel for JPG is undefined, do not assert on it.

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
