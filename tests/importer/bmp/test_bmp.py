import pytest
import cairo
import struct
from pathlib import Path
from typing import cast

from rayforge.importer.bmp.importer import BmpImporter
from rayforge.importer.bmp.renderer import BMP_RENDERER
from rayforge.importer.bmp.parser import (
    parse_bmp,
    parse_dib_header,
    _validate_format,
    _get_row_offset,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.core.vectorization_config import TraceConfig

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
def unsupported_8bit_bmp_data() -> bytes:
    return (TEST_DATA_DIR / "img-8-bit-color.bmp").read_bytes()


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


# --- Workpiece Fixture ---


@pytest.fixture
def one_bit_workpiece(bmp_1bit_data: bytes) -> WorkPiece:
    importer = BmpImporter(bmp_1bit_data)
    doc_items = importer.get_doc_items(vector_config=TraceConfig())
    assert doc_items and len(doc_items) == 1
    return cast(WorkPiece, doc_items[0])


class TestBmpParser:
    """High-level tests for the main parse_bmp function using real files."""

    def test_parse_1bit_format(self, bmp_1bit_data: bytes):
        """Verify parsing of a 1-bit BMP file."""
        header_info = parse_dib_header(bmp_1bit_data)
        assert header_info and header_info[2] == 1, "Test file is not 1-bit"
        parsed = parse_bmp(bmp_1bit_data)
        assert parsed and parsed[1] == 72 and parsed[2] == 48

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

    def test_parse_unsupported_format(self, unsupported_8bit_bmp_data: bytes):
        """Tests that the parser returns None for 8-bit BMPs."""
        assert parse_bmp(unsupported_8bit_bmp_data) is None

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
            (24, 0, True),
            (32, 0, True),
            (32, 3, True),  # BI_BITFIELDS
            (8, 0, False),  # Unsupported bpp
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
        "bmp_data_fixture",
        ["bmp_1bit_data", "bmp_24bit_data", "bmp_32bit_data"],
    )
    def test_importer_creates_workpiece_for_supported_types(
        self, bmp_data_fixture, request
    ):
        """Tests the importer creates a WorkPiece for all supported formats."""
        bmp_data = request.getfixturevalue(bmp_data_fixture)
        importer = BmpImporter(bmp_data)
        doc_items = importer.get_doc_items(vector_config=TraceConfig())

        assert doc_items and len(doc_items) == 1
        wp = doc_items[0]
        assert isinstance(wp, WorkPiece)
        expected_width = 72 * 25.4 / 96.0
        expected_height = 48 * 25.4 / 96.0
        assert wp.size == pytest.approx((expected_width, expected_height))

    def test_importer_handles_unsupported_format(
        self, unsupported_8bit_bmp_data: bytes
    ):
        """Tests the importer returns None for an unsupported bit depth."""
        importer = BmpImporter(unsupported_8bit_bmp_data)
        assert importer.get_doc_items(vector_config=TraceConfig()) is None

    def test_importer_handles_invalid_data(self):
        """Tests the importer returns None for malformed/invalid data."""
        importer = BmpImporter(b"this is not a bmp file")
        assert importer.get_doc_items(vector_config=TraceConfig()) is None


class TestBmpRenderer:
    """Tests the BmpRenderer class."""

    def test_get_natural_size(self, one_bit_workpiece: WorkPiece):
        """Test natural size calculation on the renderer."""
        size = BMP_RENDERER.get_natural_size(one_bit_workpiece)
        assert size is not None
        width_mm, height_mm = size
        expected_width = 72 * 25.4 / 96.0
        expected_height = 48 * 25.4 / 96.0
        assert width_mm == pytest.approx(expected_width)
        assert height_mm == pytest.approx(expected_height)

    def test_render_to_pixels(self, one_bit_workpiece: WorkPiece):
        """Test rendering to a Cairo surface."""
        surface = BMP_RENDERER.render_to_pixels(
            one_bit_workpiece, width=144, height=96
        )
        assert isinstance(surface, cairo.ImageSurface)
        assert surface.get_width() == 144
        assert surface.get_height() == 96

    def test_renderer_handles_invalid_data_gracefully(
        self, unsupported_8bit_bmp_data: bytes
    ):
        """
        Test that the renderer does not raise exceptions for a WorkPiece
        with unsupported data.
        """
        invalid_wp = WorkPiece(
            source_file=Path("invalid.bmp"),
            data=unsupported_8bit_bmp_data,
            renderer=BMP_RENDERER,
        )
        assert BMP_RENDERER.get_natural_size(invalid_wp) is None
        assert BMP_RENDERER.render_to_pixels(invalid_wp, 100, 100) is None
