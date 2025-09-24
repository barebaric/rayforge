import cairo
import pytest
from pathlib import Path
from typing import Tuple
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
    from pyvips import GValue

from rayforge.image import image_util as util

# Assume the test PNGs are in the same directory as this test file
TEST_DATA_DIR = Path(__file__).parent / "png"


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


@pytest.fixture
def color_png_data() -> bytes:
    """Fixture for a standard 8-bit/color RGBA PNG."""
    return load_png_data("color.png")


@pytest.fixture
def grayscale_png_data() -> bytes:
    """Fixture for an 8-bit grayscale PNG with an alpha channel."""
    return load_png_data("grayscale.png")


class TestMetadataExtraction:
    """Tests for the `extract_vips_metadata` utility function."""

    def test_extract_basic_metadata(self, color_png_data: bytes):
        """Verify that basic image properties are correctly extracted."""
        image = pyvips.Image.pngload_buffer(color_png_data)
        metadata = util.extract_vips_metadata(image)
        assert metadata["image_format"] == "PNG"
        assert metadata["width"] == 300
        assert metadata["height"] == 358
        assert metadata["bands"] == 4
        assert metadata["format"] == "uchar"
        assert metadata["interpretation"] == "srgb"

    def test_extract_metadata_with_resolution(self):
        """Verify that resolution metadata (xres, yres) is extracted."""
        # Create an image with 10 pixels/mm resolution (approx 254 DPI)
        image = pyvips.Image.black(50, 50).copy(xres=10, yres=10)
        metadata = util.extract_vips_metadata(image)
        assert metadata["xres"] == 10.0
        assert metadata["yres"] == 10.0

    def test_extract_metadata_sanitizes_long_binary_data(self):
        """Tests that long binary data is sanitized to a placeholder string."""
        image = pyvips.Image.black(10, 10)
        long_data = b"\x00\x01\x02" * 1000
        image.set_type(GValue.blob_type, "custom-binary-data", long_data)
        metadata = util.extract_vips_metadata(image)
        assert "custom-binary-data" in metadata
        assert metadata["custom-binary-data"] == "<binary data, 3000 bytes>"

    def test_extract_metadata_sanitizes_icc_profile(self):
        """Tests that ICC profile data is specifically sanitized."""
        image = pyvips.Image.black(10, 10)
        icc_data = b"dummy icc profile data"
        image.set_type(GValue.blob_type, "icc-profile-data", icc_data)
        metadata = util.extract_vips_metadata(image)
        assert "icc-profile-data" in metadata
        assert (
            metadata["icc-profile-data"]
            == f"<ICC profile, {len(icc_data)} bytes>"
        )

    def test_extract_metadata_decodes_short_binary_data(self):
        """Tests that short, valid UTF-8 binary data is decoded to a string."""
        image = pyvips.Image.black(10, 10)
        comment_text = "A test comment."
        image.set_type(
            GValue.blob_type, "comment", comment_text.encode("utf-8")
        )
        metadata = util.extract_vips_metadata(image)
        assert metadata["comment"] == comment_text

    def test_extract_metadata_handles_non_decodable_short_binary(self):
        """Tests that short, non-UTF-8 binary data is sanitized."""
        image = pyvips.Image.black(10, 10)
        bad_data = b"\xff\xfe"  # Invalid start for UTF-8
        image.set_type(GValue.blob_type, "bad-data", bad_data)
        metadata = util.extract_vips_metadata(image)
        assert metadata["bad-data"] == f"<binary data, {len(bad_data)} bytes>"


class TestImageLoadingAndSizing:
    """Tests for `load_vips_image_from_data` and `get_physical_size_mm`."""

    def test_load_vips_image_from_valid_data(self, color_png_data: bytes):
        """Ensures valid PNG data loads correctly."""
        image = util.load_vips_image_from_data(color_png_data)
        assert isinstance(image, pyvips.Image)
        assert image
        assert image.width == 300

    def test_load_vips_image_from_invalid_data(self):
        """Ensures invalid data returns None without raising an exception."""
        image = util.load_vips_image_from_data(b"this is not a png")
        assert image is None

    def test_get_physical_size_with_resolution(self):
        """Test physical size calculation with resolution metadata present."""
        image = pyvips.Image.black(100, 200).copy(xres=10, yres=20)
        width_mm, height_mm = util.get_physical_size_mm(image)
        assert width_mm == pytest.approx(10.0)  # 100px / 10px/mm
        assert height_mm == pytest.approx(10.0)  # 200px / 20px/mm

    def test_get_physical_size_without_resolution(self):
        """Test physical size calculation falling back to a default DPI."""
        image = pyvips.Image.black(96, 192)  # No xres/yres
        width_mm, height_mm = util.get_physical_size_mm(image)
        assert width_mm == pytest.approx(25.4)  # 96px at 96 DPI = 1 inch
        assert height_mm == pytest.approx(50.8)  # 192px at 96 DPI = 2 inches


class TestImageNormalization:
    """Tests for `normalize_to_rgba`."""

    def test_normalize_already_rgba_image(self, color_png_data: bytes):
        """Test that an RGBA uchar image is passed through correctly."""
        image = pyvips.Image.pngload_buffer(color_png_data)
        normalized = util.normalize_to_rgba(image)
        assert normalized is not None
        assert normalized.bands == 4
        assert normalized.format == "uchar"
        assert normalized.interpretation == "srgb"

    def test_normalize_grayscale_alpha_image(self, grayscale_png_data: bytes):
        """Test that a grayscale+alpha image is converted to sRGB RGBA."""
        image = pyvips.Image.pngload_buffer(grayscale_png_data)
        assert image.interpretation == "b-w"
        assert image.bands == 2
        normalized = util.normalize_to_rgba(image)
        assert normalized is not None
        assert normalized.bands == 4
        assert normalized.format == "uchar"
        assert normalized.interpretation == "srgb"

    def test_normalize_rgb_image(self):
        """Test that an RGB image gets an alpha channel added."""
        image = pyvips.Image.black(10, 10, bands=3).copy(interpretation="srgb")
        assert not image.hasalpha()
        normalized = util.normalize_to_rgba(image)
        assert normalized is not None
        assert normalized.bands == 4
        assert normalized.hasalpha()

    def test_normalize_16bit_image(self):
        """Test that a 16-bit image is correctly cast down to 8-bit."""
        image = pyvips.Image.black(10, 10, bands=4).cast("ushort")
        assert image.format == "ushort"
        normalized = util.normalize_to_rgba(image)
        assert normalized is not None
        assert normalized.format == "uchar"

    def test_normalize_cmyk_image(self):
        """Test that a CMYK image is converted to sRGB."""
        image = pyvips.Image.black(10, 10, bands=4).copy(interpretation="cmyk")
        normalized = util.normalize_to_rgba(image)
        assert normalized is not None
        assert normalized.interpretation == "srgb"
        assert normalized.bands == 4  # Should gain an alpha channel


class TestCairoConversion:
    """Tests for `vips_rgba_to_cairo_surface`."""

    def test_conversion_preserves_size_and_format(self, color_png_data: bytes):
        """Verify the output surface has the correct dimensions and format."""
        image = util.load_vips_image_from_data(color_png_data)
        normalized = util.normalize_to_rgba(image)
        assert normalized is not None
        surface = util.vips_rgba_to_cairo_surface(normalized)
        assert isinstance(surface, cairo.ImageSurface)
        assert image
        assert surface.get_width() == image.width
        assert surface.get_height() == image.height
        assert surface.get_format() == cairo.FORMAT_ARGB32

    def test_conversion_shuffles_channels_to_bgra(self, color_png_data: bytes):
        """
        Verify the RGBA to BGRA channel shuffling by sampling a known pixel.
        """
        image = util.load_vips_image_from_data(color_png_data)
        normalized = util.normalize_to_rgba(image)
        assert normalized is not None
        surface = util.vips_rgba_to_cairo_surface(normalized)

        # The color at this coordinate in the source PNG is
        # (136, 189, 245, 255)
        expected_rgba = (136, 189, 245, 255)
        b, g, r, a = get_pixel_bgra(surface, x=150, y=50)

        # Check that the returned values match the expected RGBA values
        assert (r, g, b, a) == expected_rgba
