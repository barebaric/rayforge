import pytest
from pathlib import Path
from typing import Optional
from rayforge.image.svg.svg_base import SvgImporterBase
from rayforge.image.structures import ParsingResult, VectorizationResult
from rayforge.core.vectorization_spec import VectorizationSpec
from xml.etree import ElementTree as ET

# --- Test Data ---

SVG_VALID_LAYERS = b"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">
  <g inkscape:groupmode="layer" id="layer1" inkscape:label="Cut">
    <rect x="0" y="0" width="10" height="10" />
  </g>
  <g inkscape:groupmode="layer" id="layer2" inkscape:label="Engrave">
    <circle cx="50" cy="50" r="10" />
  </g>
</svg>
"""

SVG_NO_LAYERS = b"""
<svg width="50px" height="50px" viewBox="0 0 50 50"
     xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="10" width="30" height="30"/>
</svg>"""

SVG_TRIM_TARGET = b"""
<svg width="200px" height="200px" viewBox="0 0 200 200"
     xmlns="http://www.w3.org/2000/svg">
    <rect x="50" y="50" width="10" height="10" fill="black"/>
</svg>
"""

SVG_INVALID = b"<svg>Not XML</svg"

# --- Test Harness ---


class ConcreteSvgImporter(SvgImporterBase):
    """Concrete implementation of SvgImporterBase for testing purposes."""

    label = "Concrete Test Importer"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

    def parse(self) -> Optional[ParsingResult]:
        return None

    def vectorize(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> VectorizationResult:
        raise NotImplementedError()


# --- Tests ---


def test_scan_finds_valid_layers():
    """Test scanning correctly identifies Inkscape layers and dimensions."""
    importer = ConcreteSvgImporter(SVG_VALID_LAYERS, Path("layers.svg"))
    manifest = importer.scan()

    assert manifest.title == "layers.svg"
    assert manifest.natural_size_mm is not None
    w, h = manifest.natural_size_mm
    assert w == pytest.approx(100.0, 0.1)
    assert h == pytest.approx(100.0, 0.1)

    assert len(manifest.layers) == 2
    assert manifest.layers[0].id == "layer1"
    assert manifest.layers[0].name == "Cut"
    assert manifest.layers[1].id == "layer2"
    assert manifest.layers[1].name == "Engrave"
    assert not manifest.warnings


def test_scan_no_layers_structure():
    """Test scanning a simple SVG with no explicit layer groups."""
    importer = ConcreteSvgImporter(SVG_NO_LAYERS, Path("simple.svg"))
    manifest = importer.scan()

    assert manifest.natural_size_mm is not None
    w, h = manifest.natural_size_mm
    assert w == pytest.approx(13.23, 0.1)
    assert len(manifest.layers) == 0


def test_scan_handles_corrupt_file_gracefully():
    """Test scanning a malformed file returns warnings instead of crashing."""
    importer = ConcreteSvgImporter(SVG_INVALID, Path("bad.svg"))
    manifest = importer.scan()

    assert manifest.title == "bad.svg"
    assert manifest.natural_size_mm is None
    assert len(manifest.warnings) > 0
    assert any("parse" in w.lower() for w in manifest.warnings)


def test_analytical_trim_logic():
    """Test that _analytical_trim correctly crops the SVG viewbox."""
    importer = ConcreteSvgImporter(SVG_TRIM_TARGET, Path("trim.svg"))
    trimmed_bytes = importer._analytical_trim(SVG_TRIM_TARGET)
    root = ET.fromstring(trimmed_bytes)

    # Content is 10x10. Padding is 10*0.01=0.1 on each side.
    # New width/height is 10 + 2*0.1 = 10.2
    # New x/y is 50 - 0.1 = 49.9
    assert root.get("viewBox") == "49.9 49.9 10.2 10.2"
    assert root.get("width") == "10.2000px"
    assert root.get("height") == "10.2000px"


def test_calculate_parsing_basics_success():
    """Test helper method extracts SVG object and standardized bounds."""
    importer = ConcreteSvgImporter(SVG_VALID_LAYERS)
    result = importer._calculate_parsing_basics()

    assert result is not None
    (
        svg_obj,
        document_bounds_units,
        unit_to_mm,
        untrimmed_bounds,
        world_frame,
    ) = result

    assert unit_to_mm == pytest.approx(1.0, 0.01)

    # Content bounds are 0,0 to 60,60. Padding is 60*0.01=0.6 on each side.
    # New viewbox is -0.6, -0.6, 61.2, 61.2
    px, py, pw, ph = document_bounds_units
    assert px == pytest.approx(-0.6)
    assert py == pytest.approx(-0.6)
    assert pw == pytest.approx(61.2)
    assert ph == pytest.approx(61.2)

    assert untrimmed_bounds is not None
    upx, upy, upw, uph = untrimmed_bounds
    assert upw == pytest.approx(100.0)
    assert uph == pytest.approx(100.0)


def test_calculate_parsing_basics_world_frame():
    """Test that the world_frame_of_reference is calculated correctly."""
    importer = ConcreteSvgImporter(SVG_VALID_LAYERS)
    result = importer._calculate_parsing_basics()

    assert result is not None
    _, _, unit_to_mm, untrimmed_bounds, world_frame = result

    assert untrimmed_bounds is not None
    # The world frame should be based on the untrimmed bounds.
    # untrimmed_bounds is in native units, which are mm for this SVG.
    ux, uy, uw, uh = untrimmed_bounds
    assert uw == pytest.approx(100.0)
    assert uh == pytest.approx(100.0)

    # World frame is (x_mm, y_mm, w_mm, h_mm), Y-Up
    fx, fy, fw, fh = world_frame
    assert fw == pytest.approx(uw * unit_to_mm)
    assert fh == pytest.approx(uh * unit_to_mm)
    assert fx == pytest.approx(ux * unit_to_mm)
    assert fy == pytest.approx(0.0)  # Y should be 0 for a Y-Up frame
