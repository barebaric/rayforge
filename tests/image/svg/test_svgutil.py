import pytest
from xml.etree import ElementTree as ET

# Assuming the file structure allows this import
from rayforge.image.svg.svgutil import (
    get_natural_size,
    trim_svg,
    MM_PER_PX,
)
from rayforge.image.util import parse_length


# --- Test Data Fixtures ---


@pytest.fixture
def svg_mm_data() -> bytes:
    return b"""<svg width="150mm" height="75.5mm">
                 <rect width="150" height="75.5" fill="red"/>
               </svg>"""


@pytest.fixture
def svg_px_data() -> bytes:
    return b"""<svg width="200px" height="100px">
                 <rect width="200" height="100" fill="blue"/>
               </svg>"""


@pytest.fixture
def svg_unitless_data() -> bytes:
    return b'<svg width="200" height="100"></svg>'


@pytest.fixture
def svg_with_margins_data() -> bytes:
    """SVG 200x200 with a 100x100 rect in the center."""
    return b"""<svg width="200px" height="200px" viewBox="0 0 200 200">
                 <rect x="50" y="50" width="100" height="100" fill="green"/>
               </svg>"""


@pytest.fixture
def svg_with_margins_and_mm_data() -> bytes:
    """SVG 80mm x 80mm with a 40x40 content rect in the center."""
    return b"""<svg width="80mm" height="80mm" viewBox="0 0 80 80">
                 <rect x="20" y="20" width="40" height="40" fill="green"/>
               </svg>"""


@pytest.fixture
def empty_svg_data() -> bytes:
    return b"<svg/>"


# --- Tests for get_natural_size ---


class TestGetNaturalSize:
    def test_get_size_with_mm(self, svg_mm_data: bytes):
        size = get_natural_size(svg_mm_data)
        assert size is not None
        width, height = size
        assert width == pytest.approx(150.0)
        assert height == pytest.approx(75.5)

    def test_get_size_with_px(self, svg_px_data: bytes):
        size = get_natural_size(svg_px_data)
        assert size is not None
        width, height = size
        assert width == pytest.approx(200.0 * MM_PER_PX)
        assert height == pytest.approx(100.0 * MM_PER_PX)

    def test_get_size_with_unitless(self, svg_unitless_data: bytes):
        # Unitless should default to px
        size = get_natural_size(svg_unitless_data)
        assert size is not None
        width, height = size
        assert width == pytest.approx(200.0 * MM_PER_PX)
        assert height == pytest.approx(100.0 * MM_PER_PX)

    def test_get_size_empty_svg(self, empty_svg_data: bytes):
        # No width/height attributes
        assert get_natural_size(empty_svg_data) is None

    def test_get_size_invalid_data(self):
        assert get_natural_size(b"not valid xml") is None
        assert get_natural_size(b"") is None


# --- Tests for trim_svg ---


class TestTrimSvg:
    def test_trim_svg_with_margins(self, svg_with_margins_data: bytes):
        """Test trimming an SVG with 25% margin on each side."""
        trimmed_data = trim_svg(svg_with_margins_data)
        assert trimmed_data != svg_with_margins_data

        root = ET.fromstring(trimmed_data)
        # Original was 200px, content is 100px. New size should be 100px.
        assert root.get("width") == "100.0px"
        assert root.get("height") == "100.0px"

        # ViewBox should be cropped
        # from '0 0 200 200' to '50.0 50.0 100.0 100.0'
        vb_str = root.get("viewBox")
        assert vb_str is not None
        vb_vals = [float(v) for v in vb_str.split()]
        assert vb_vals == pytest.approx([50.0, 50.0, 100.0, 100.0])

    def test_trim_svg_with_mm_units(self, svg_with_margins_and_mm_data: bytes):
        """Test trimming an SVG defined in mm."""
        trimmed_data = trim_svg(svg_with_margins_and_mm_data)
        assert trimmed_data != svg_with_margins_and_mm_data

        root = ET.fromstring(trimmed_data)
        # Original was 80mm, content is 40mm wide (20+40=60, margin is 20).
        # Margins are 20/80 = 0.25 on each side.
        # New size should be 80 * (1 - 0.25 - 0.25) = 40mm
        assert root.get("width") == "40.0mm"
        assert root.get("height") == "40.0mm"

        # ViewBox should be cropped from '0 0 80 80' to '20 20 40 40'
        vb_str = root.get("viewBox")
        assert vb_str is not None
        vb_vals = [float(v) for v in vb_str.split()]
        assert vb_vals == pytest.approx([20.0, 20.0, 40.0, 40.0])

    def test_no_trim_needed(self, svg_px_data: bytes):
        """
        Test that an SVG with no margins is semantically unchanged after
        running through trim_svg.
        """
        trimmed_data = trim_svg(svg_px_data)

        # A direct byte comparison is too brittle due to potential XML
        # serialization differences (whitespace, attribute order) and
        # floating point inaccuracies in margin calculation.
        # Instead, parse both and compare that the dimensions are unchanged.
        original_root = ET.fromstring(svg_px_data)
        trimmed_root = ET.fromstring(trimmed_data)

        orig_w_val, _ = parse_length(original_root.get("width"))
        trim_w_val, _ = parse_length(trimmed_root.get("width"))
        assert trim_w_val == pytest.approx(orig_w_val)

        orig_h_val, _ = parse_length(original_root.get("height"))
        trim_h_val, _ = parse_length(trimmed_root.get("height"))
        assert trim_h_val == pytest.approx(orig_h_val)

    def test_trim_empty_svg(self, empty_svg_data: bytes):
        """Test that an empty SVG is returned unchanged."""
        trimmed_data = trim_svg(empty_svg_data)
        assert trimmed_data == empty_svg_data
