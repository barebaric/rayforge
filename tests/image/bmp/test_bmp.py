import pytest
import cairo
import struct
from pathlib import Path
from typing import cast
from unittest.mock import Mock
from rayforge.image import import_file
from rayforge.image.base_importer import ImportPayload
from rayforge.image.bmp.renderer import BMP_RENDERER
from rayforge.image.bmp.parser import (
    parse_bmp,
    parse_dib_header,
    _validate_format,
    _get_row_offset,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.core.vectorization_spec import TraceSpec
from rayforge.core.source_asset import SourceAsset
from rayforge.core.matrix import Matrix
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.geo import Geometry

TEST_DATA_DIR = Path(__file__).parent


@pytest.fixture
def bmp_1bit_data() -> bytes:
    return (TEST_DATA_DIR / "img-1-bit.bmp").read_bytes()


@pytest.fixture
def bmp_24bit_data() -> bytes:
    return (TEST_DATA_DIR / "img-24-bit-gray.bmp").read_bytes()


@pytest.fixture
def bmp_32bit_data() -> bytes:
    return (TEST_DATA_DIR / "img-32-bit-color.bmp").read_bytes()


@pytest.fixture
def bmp_8bit_data() -> bytes:
    return (TEST_DATA_DIR / "img-8-bit-color.bmp").read_bytes()


@pytest.fixture
def bmp_8bit_gray_v5_header_data() -> bytes:
    return (TEST_DATA_DIR / "img-8-bit-gray-2.bmp").read_bytes()


@pytest.fixture
def bmp_core_header_data() -> bytes:
    """Generates a minimal BMP with a valid, padded BITMAPCOREHEADER."""
    width, height = 16, 16

    # Correctly calculate padded row size
    row_bytes_unpadded = (width + 7) // 8  # 2 bytes for 16 pixels
    row_bytes_padded = (row_bytes_unpadded + 3) & ~3  # Padded to 4 bytes

    palette_size = 2 * 3  # 2 entries, 3 bytes each (BGR) for CORE header
    image_data_size = row_bytes_padded * height  # 4 * 16 = 64 bytes

    pixel_data_start = 14 + 12 + palette_size  # 14+12+6 = 32
    file_size = pixel_data_start + image_data_size  # 32 + 64 = 96

    # File Header
    header = struct.pack("<2sIHHI", b"BM", file_size, 0, 0, pixel_data_start)
    # DIB Header (BITMAPCOREHEADER)
    dib_header = struct.pack("<IHHHH", 12, width, height, 1, 1)  # 1-bit
    # Palette (Black and White)
    palette = b"\x00\x00\x00" + b"\xff\xff\xff"

    # Generate pixel data WITH padding
    row_data = b"\xaa\x55"  # 2 bytes of checkerboard pattern
    padding = b"\x00" * (
        row_bytes_padded - row_bytes_unpadded
    )  # 2 bytes of padding
    pixels = (row_data + padding) * height

    return header + dib_header + palette + pixels


@pytest.fixture
def bmp_truncated_data(bmp_24bit_data: bytes) -> bytes:
    """Returns a valid BMP file with its pixel data cut short."""
    return bmp_24bit_data[:-100]


@pytest.fixture
def import_payload_helper():
    """A helper that wraps import_file for convenience."""

    def _import(
        data: bytes, source_file: Path = Path("test.bmp")
    ) -> ImportPayload:
        payload = import_file(
            source=data,
            mime_type="image/bmp",
            vectorization_spec=TraceSpec(),
        )
        assert payload is not None, "import_file failed to produce a payload"
        return payload

    return _import


def _setup_workpiece_with_context(payload: ImportPayload) -> WorkPiece:
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
def one_bit_workpiece(
    import_payload_helper, bmp_1bit_data: bytes
) -> WorkPiece:
    payload = import_payload_helper(bmp_1bit_data)
    return _setup_workpiece_with_context(payload)


@pytest.fixture
def eight_bit_workpiece(
    import_payload_helper, bmp_8bit_data: bytes
) -> WorkPiece:
    payload = import_payload_helper(bmp_8bit_data)
    return _setup_workpiece_with_context(payload)


class TestBmpParser:
    """High-level tests for the main parse_bmp function using real files."""

    def test_parse_1bit_format(self, bmp_1bit_data: bytes):
        """Verify parsing of a 1-bit BMP file."""
        header_info = parse_dib_header(bmp_1bit_data)
        assert header_info and header_info[2] == 1, "Test file is not 1-bit"
        parsed = parse_bmp(bmp_1bit_data)
        assert parsed and parsed[1] == 72 and parsed[2] == 48

    def test_parse_8bit_format(self, bmp_8bit_data: bytes):
        """Verify parsing of an 8-bit BMP file."""
        header_info = parse_dib_header(bmp_8bit_data)
        assert header_info and header_info[2] == 8, "Test file is not 8-bit"
        parsed = parse_bmp(bmp_8bit_data)
        assert parsed and parsed[1] == 72 and parsed[2] == 48

    def test_parse_8bit_v5_header_format(
        self, bmp_8bit_gray_v5_header_data: bytes
    ):
        """Verify parsing of an 8-bit BMP with a V5 header."""
        header_info = parse_dib_header(bmp_8bit_gray_v5_header_data)
        assert header_info and header_info[2] == 8, "Test file is not 8-bit"

        parsed = parse_bmp(bmp_8bit_gray_v5_header_data)
        assert parsed, "Parser failed to process 8-bit V5 header file"

        rgba_buffer, width, height, _, _ = parsed
        assert width == 300
        assert height == 358
        assert len(rgba_buffer) == 300 * 358 * 4

    def test_parse_24bit_format(self, bmp_24bit_data: bytes):
        """Verify parsing of a 24-bit BMP file."""
        header_info = parse_dib_header(bmp_24bit_data)
        assert header_info and header_info[2] == 24, "Test file is not 24-bit"
        parsed = parse_bmp(bmp_24bit_data)
        assert parsed and parsed[1] == 72 and parsed[2] == 48

    def test_parse_32bit_format(self, bmp_32bit_data: bytes):
        """Verify parsing of a 32-bit V5 BMP file."""
        header_info = parse_dib_header(bmp_32bit_data)
        assert header_info and header_info[2] == 32, "Test file is not 32-bit"
        parsed = parse_bmp(bmp_32bit_data)
        assert parsed and parsed[1] == 72 and parsed[2] == 48

    def test_parse_core_header_format(self, bmp_core_header_data: bytes):
        """Verify parsing of an old BITMAPCOREHEADER file."""
        header_info = parse_dib_header(bmp_core_header_data)
        assert header_info and header_info[2] == 1
        parsed = parse_bmp(bmp_core_header_data)
        assert parsed and parsed[1] == 16 and parsed[2] == 16

    def test_parse_invalid_data(self):
        """Tests that the parser returns None for non-BMP or malformed data."""
        assert parse_bmp(b"this is not a bmp") is None
        assert parse_bmp(b"BM" + b"\x00" * 50) is None

    def test_parse_truncated_file(self, bmp_truncated_data: bytes):
        """Tests that a file with incomplete pixel data fails gracefully."""
        assert parse_bmp(bmp_truncated_data) is None


class TestBmpParserHelpers:
    """
    Unit tests for individual helper functions and edge cases in the parser.
    """

    @pytest.mark.parametrize(
        "bpp, compression, expected",
        [
            (1, 0, True),
            (8, 0, True),
            (24, 0, True),
            (32, 0, True),
            (32, 3, True),  # BI_BITFIELDS
            (16, 0, False),  # Unsupported bpp
            (24, 1, False),  # Unsupported RLE compression
        ],
    )
    def test_validate_format(self, bpp, compression, expected):
        """Test the _validate_format helper with various inputs."""
        assert _validate_format(bpp, compression) is expected

    def test_get_row_offset_bottom_up(self):
        """Test row offset calculation for standard bottom-up images."""
        # For a 10px high image, row 0 is the last row in the data.
        assert (
            _get_row_offset(
                y=0, height=10, row_size=20, data_start=100, is_top_down=False
            )
            == 100 + 9 * 20
        )
        # Row 9 is the first row in the data.
        assert (
            _get_row_offset(
                y=9, height=10, row_size=20, data_start=100, is_top_down=False
            )
            == 100 + 0 * 20
        )

    def test_get_row_offset_top_down(self):
        """Test row offset calculation for top-down images."""
        # For a 10px high image, row 0 is the first row in the data.
        assert (
            _get_row_offset(
                y=0, height=10, row_size=20, data_start=100, is_top_down=True
            )
            == 100 + 0 * 20
        )
        # Row 9 is the last row in the data.
        assert (
            _get_row_offset(
                y=9, height=10, row_size=20, data_start=100, is_top_down=True
            )
            == 100 + 9 * 20
        )


class TestBmpImporter:
    """Tests the BmpImporter class."""

    @pytest.mark.parametrize(
        "bmp_data_fixture, expected_dims",
        [
            ("bmp_1bit_data", (72, 48)),
            ("bmp_8bit_data", (72, 48)),
            ("bmp_24bit_data", (72, 48)),
            ("bmp_32bit_data", (72, 48)),
            ("bmp_8bit_gray_v5_header_data", (300, 358)),
        ],
    )
    def test_importer_creates_workpiece_for_supported_types(
        self, bmp_data_fixture, expected_dims, request, import_payload_helper
    ):
        """Tests the importer creates a WorkPiece for all supported formats."""
        bmp_data = request.getfixturevalue(bmp_data_fixture)
        payload = import_payload_helper(bmp_data)

        assert payload and payload.items and len(payload.items) == 1
        assert isinstance(payload.source, SourceAsset)
        wp = payload.items[0]
        assert isinstance(wp, WorkPiece)
        assert wp.source_segment is not None
        assert wp.source_segment.source_asset_uid == payload.source.uid

        # The importer auto-crops to content. We check that the resulting
        # size is positive but no larger than the original canvas size.
        parsed_data = parse_bmp(bmp_data)
        assert parsed_data is not None
        _, _, _, dpi_x, dpi_y = parsed_data
        dpi_x = dpi_x or 96.0
        dpi_y = dpi_y or 96.0

        expected_w_px, expected_h_px = expected_dims
        max_width_mm = expected_w_px * 25.4 / dpi_x
        max_height_mm = expected_h_px * 25.4 / dpi_y

        wp_w, wp_h = wp.size
        assert 0 < wp_w <= max_width_mm + 1e-9  # Add tolerance
        assert 0 < wp_h <= max_height_mm + 1e-9

    def test_importer_handles_invalid_data(self):
        """Tests the importer returns None for malformed/invalid data."""
        assert (
            import_file(b"this is not a bmp file", mime_type="image/bmp")
            is None
        )


class TestBmpRenderer:
    """Tests the BmpRenderer class."""

    @pytest.mark.parametrize(
        "workpiece_fixture", ["one_bit_workpiece", "eight_bit_workpiece"]
    )
    def test_get_natural_size(self, workpiece_fixture, request):
        """Test natural size calculation on the renderer."""
        workpiece = request.getfixturevalue(workpiece_fixture)
        size = BMP_RENDERER.get_natural_size(workpiece)
        assert size is not None
        width_mm, height_mm = size

        # Natural size is the cropped content size from the generation
        # config. We check that it's positive and not larger than the canvas.
        max_width = 72 * 25.4 / 96.0
        max_height = 48 * 25.4 / 96.0
        assert 0 < width_mm <= max_width + 1e-9
        assert 0 < height_mm <= max_height + 1e-9

    @pytest.mark.parametrize(
        "workpiece_fixture", ["one_bit_workpiece", "eight_bit_workpiece"]
    )
    def test_render_to_pixels(self, workpiece_fixture, request):
        """Test rendering to a Cairo surface."""
        workpiece = request.getfixturevalue(workpiece_fixture)
        surface = BMP_RENDERER.render_to_pixels(
            workpiece, width=144, height=96
        )
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 144
        assert surface.get_height() == 96

    def test_renderer_handles_invalid_data_gracefully(self):
        """
        Test that the renderer does not raise exceptions for a WorkPiece
        with invalid data.
        """
        source = SourceAsset(
            source_file=Path("invalid.bmp"),
            original_data=b"not a valid bmp",
            renderer=BMP_RENDERER,
        )
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            segment_mask_geometry=Geometry(),
            vectorization_spec=TraceSpec(),
        )
        invalid_wp = WorkPiece(name="invalid", source_segment=gen_config)

        # Mock document context
        mock_doc = Mock()
        mock_doc.source_assets = {source.uid: source}
        mock_doc.get_source_asset_by_uid.side_effect = (
            mock_doc.source_assets.get
        )
        mock_parent = Mock()
        mock_parent.doc = mock_doc
        mock_parent.get_world_transform.return_value = Matrix.identity()
        invalid_wp.parent = mock_parent

        assert BMP_RENDERER.get_natural_size(invalid_wp) is None
        assert BMP_RENDERER.render_to_pixels(invalid_wp, 100, 100) is None
