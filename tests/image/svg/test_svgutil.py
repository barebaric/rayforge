from xml.etree import ElementTree as ET

import pytest

# Assuming the file structure allows this import
from rayforge.image.svg.svgutil import (
    MM_PER_PX,
    NO_LAYER_KEY,
    extract_layer_manifest,
    filter_svg_layers,
    get_natural_size,
    trim_svg,
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


@pytest.fixture
def svg_viewbox_only_data() -> bytes:
    """Unitless SVG carrying only a viewBox, no width/height attributes."""
    return b"""<svg viewBox="0 0 475 475">
                 <rect width="475" height="475" fill="red"/>
               </svg>"""


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

    def test_get_size_viewbox_only(self, svg_viewbox_only_data: bytes):
        # SVG with only a viewBox (no width/height) should fall back to the
        # viewBox size, treating user units as pixels at the requested ppi.
        size = get_natural_size(svg_viewbox_only_data)
        assert size is not None
        width, height = size
        assert width == pytest.approx(475.0 * MM_PER_PX)
        assert height == pytest.approx(475.0 * MM_PER_PX)

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


# --- Test data for orphan (root-level) geometry ---


@pytest.fixture
def svg_orphan_data() -> bytes:
    """Named layer plus root-level shapes outside any layer."""
    return b"""<svg width="100mm" height="100mm" viewBox="0 0 100 100"
                 xmlns="http://www.w3.org/2000/svg">
                 <defs><linearGradient id="grad"/></defs>
                 <g id="layer1"><rect x="0" y="0" width="10" height="10"/></g>
                 <rect x="20" y="20" width="30" height="5"/>
                 <g><rect x="60" y="20" width="30" height="5"/></g>
               </svg>"""


# --- Tests for extract_layer_manifest ---


class TestExtractLayerManifest:
    def test_orphan_geometry_listed_as_pseudo_layer(
        self, svg_orphan_data: bytes
    ):
        manifest = extract_layer_manifest(svg_orphan_data)
        by_id = {layer["id"]: layer for layer in manifest}
        assert set(by_id) == {"layer1", NO_LAYER_KEY}
        assert by_id[NO_LAYER_KEY]["count"] == 2
        assert by_id[NO_LAYER_KEY]["name"] == "Ungrouped"

    def test_no_pseudo_layer_without_orphans(self):
        data = b"""<svg xmlns="http://www.w3.org/2000/svg">
                     <g id="layer1"><rect width="10" height="10"/></g>
                   </svg>"""
        manifest = extract_layer_manifest(data)
        assert [layer["id"] for layer in manifest] == ["layer1"]

    def test_no_pseudo_layer_without_named_layers(self):
        data = b"""<svg xmlns="http://www.w3.org/2000/svg">
                     <rect width="10" height="10"/>
                   </svg>"""
        assert extract_layer_manifest(data) == []


# --- Tests for filter_svg_layers ---


class TestFilterSvgLayers:
    def test_named_layer_drops_orphan_geometry(self, svg_orphan_data: bytes):
        filtered = filter_svg_layers(svg_orphan_data, ["layer1"])
        root = ET.fromstring(filtered)
        tags = [child.tag.rsplit("}", 1)[-1] for child in root]
        assert "rect" not in tags
        assert tags.count("g") == 1
        assert "defs" in tags
        group = next(
            child for child in root if child.tag.rsplit("}", 1)[-1] == "g"
        )
        assert group.get("id") == "layer1"

    def test_pseudo_layer_keeps_only_orphan_geometry(
        self, svg_orphan_data: bytes
    ):
        filtered = filter_svg_layers(svg_orphan_data, [NO_LAYER_KEY])
        root = ET.fromstring(filtered)
        groups = [
            child for child in root if child.tag.rsplit("}", 1)[-1] == "g"
        ]
        assert all(not group.get("id") for group in groups)
        rects = [
            child for child in root if child.tag.rsplit("}", 1)[-1] == "rect"
        ]
        assert len(rects) == 1
        assert len(groups) == 1

    def test_named_layer_keeps_orphans_when_pseudo_selected(
        self, svg_orphan_data: bytes
    ):
        filtered = filter_svg_layers(svg_orphan_data, ["layer1", NO_LAYER_KEY])
        root = ET.fromstring(filtered)
        tags = [child.tag.rsplit("}", 1)[-1] for child in root]
        assert tags.count("g") == 2
        assert "rect" in tags
