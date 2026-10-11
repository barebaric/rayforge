import gzip
from pathlib import Path

from rayforge.image import importer_registry
from rayforge.image.svg.importer import SvgImporter

SVG_WITH_LAYER = b"""<?xml version="1.0" encoding="UTF-8"?>
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg"
     xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape">
  <g inkscape:groupmode="layer" id="layer1" inkscape:label="Cut">
    <rect x="0" y="0" width="10" height="10" />
  </g>
</svg>
"""

CORRUPT_GZIP = b"\x1f\x8b" + b"not really gzip"


def test_registry_resolves_svgz():
    assert importer_registry.get_by_extension(".svgz") is SvgImporter


def test_extensions_include_svgz():
    assert ".svgz" in SvgImporter.extensions


def test_scan_svgz_matches_plain_svg():
    plain = SvgImporter(SVG_WITH_LAYER).scan()
    packed = SvgImporter(gzip.compress(SVG_WITH_LAYER)).scan()
    assert packed.errors == []
    assert packed.natural_size_mm == plain.natural_size_mm
    assert [lay.name for lay in packed.layers] == [
        lay.name for lay in plain.layers
    ]


def test_get_doc_items_from_svgz():
    result = SvgImporter(gzip.compress(SVG_WITH_LAYER)).get_doc_items()
    assert result is not None
    assert result.errors == []
    assert result.payload is not None
    assert len(result.payload.items) > 0


def test_plain_svg_is_passed_through_unchanged():
    importer = SvgImporter(SVG_WITH_LAYER)
    assert importer.raw_data == SVG_WITH_LAYER


def test_corrupt_gzip_reports_an_error():
    manifest = SvgImporter(CORRUPT_GZIP).scan()
    assert any(
        "gzip" in e.lower() or "svgz" in e.lower() for e in manifest.errors
    )


def test_imports_a_real_inkscape_svgz(tmp_path: Path):
    src = Path(__file__).parent / "circle.svg"
    gz = tmp_path / "circle.svgz"
    gz.write_bytes(gzip.compress(src.read_bytes()))
    importer = SvgImporter(gz.read_bytes(), gz)
    manifest = importer.scan()
    assert manifest.errors == []
    assert manifest.natural_size_mm is not None
