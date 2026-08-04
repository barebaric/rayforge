from pathlib import Path

import pytest
from raygeo.svg.color import ColorAttr

from rayforge.core.layer import Layer
from rayforge.core.vectorization_spec import (
    LayerImportMode,
    LayerSource,
    PassthroughSpec,
)
from rayforge.core.workpiece import WorkPiece
from rayforge.image.svg.importer import SvgImporter
from rayforge.image.svg.svg_vector import SvgVectorImporter

SVG_BASIC = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <rect x="10" y="10" width="20" height="20" />
</svg>
"""

SVG_GROUPS = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <g id="g1"><rect x="0" y="0" width="10" height="10"/></g>
    <g id="g2"><rect x="90" y="90" width="10" height="10"/></g>
</svg>
"""

SVG_COLORS = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="10" height="10" fill="#ff0000"/>
    <rect x="20" y="20" width="10" height="10" fill="#ff0000"/>
    <rect x="50" y="50" width="10" height="10" fill="#00ff00"/>
</svg>
"""

SVG_NO_COLOR = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <rect x="0" y="0" width="10" height="10" fill="#ff0000"/>
    <rect x="50" y="50" width="10" height="10"/>
</svg>
"""

SVG_TRANSFORM = b"""
<svg width="100mm" height="100mm" viewBox="0 0 100 100"
     xmlns="http://www.w3.org/2000/svg">
    <g transform="translate(10, 10)">
        <rect x="10" y="10" width="20" height="20" />
    </g>
</svg>
"""


@pytest.fixture
def vector_importer():
    return SvgVectorImporter(SVG_BASIC, Path("vector.svg"))


def test_parse_valid_vector_data(vector_importer):
    result = vector_importer.parse()
    assert result is not None

    # Content 20x20, padding 0.2. New viewbox is padded.
    px, py, pw, ph = result.document_bounds
    assert px == pytest.approx(9.8, abs=1e-3)
    assert py == pytest.approx(9.8, abs=1e-3)
    assert pw == pytest.approx(20.4, abs=1e-3)
    assert ph == pytest.approx(20.4, abs=1e-3)
    assert result.native_unit_to_mm == pytest.approx(1.0)
    assert result.is_y_down is True
    # The parse method now extracts actual layers, it doesn't create a default
    # for layerless SVGs.
    assert len(result.layers) == 0


def test_get_doc_items_alignment(vector_importer):
    """Verifies World Space alignment."""
    # Use a spec that forces merge, which is the default for layerless SVGs
    spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
    import_result = vector_importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload

    assert payload is not None
    item = payload.items[0]
    assert isinstance(item, WorkPiece)

    # Natural size should match padded content bounds
    assert item.natural_width_mm == pytest.approx(20.4)
    assert item.natural_height_mm == pytest.approx(20.4)

    # The workpiece is positioned at the top-left of the padded
    # bounds (9.8, 9.8), Y-inverted relative to the 100mm page.
    #   100 - (9.8 + 20.4) = 69.8
    wx, wy = item.matrix.transform_point((0, 0))
    assert wx == pytest.approx(9.8, abs=1e-3)
    assert wy == pytest.approx(69.8, abs=1e-3)


def test_layer_separation_and_positioning():
    """Test importing layers creates separate items positioned correctly."""
    importer = SvgVectorImporter(SVG_GROUPS, Path("groups.svg"))
    manifest = importer.scan()
    layer_ids = [layer.id for layer in manifest.layers]
    spec = PassthroughSpec(
        active_layer_ids=layer_ids,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
    )
    import_result = importer.get_doc_items(vectorization_spec=spec)
    assert import_result is not None
    payload = import_result.payload

    assert payload is not None
    assert len(payload.items) == 2

    # Find layers by name to make test robust to order
    l1 = next(item for item in payload.items if item.name == "g1")
    l2 = next(item for item in payload.items if item.name == "g2")

    assert isinstance(l1, Layer)
    wp1 = next(c for c in l1.children if isinstance(c, WorkPiece))
    # Position (0,0), Y-inverted: 100 - (0+10) = 90
    wx1, wy1 = wp1.get_world_transform().transform_point((0, 0))
    assert wx1 == pytest.approx(0.0, abs=1e-4)
    assert wy1 == pytest.approx(90.0, abs=1e-4)

    assert isinstance(l2, Layer)
    wp2 = next(c for c in l2.children if isinstance(c, WorkPiece))
    # Position (90,90), Y-inverted: 100 - (90+10) = 0
    wx2, wy2 = wp2.get_world_transform().transform_point((0, 0))
    assert wx2 == pytest.approx(90.0, abs=1e-4)
    assert wy2 == pytest.approx(0.0, abs=1e-4)


def test_nested_transforms_applied():
    """Test that group transforms are applied to the geometry."""
    importer = SvgVectorImporter(SVG_TRANSFORM, Path("trans.svg"))
    # Force merge
    spec = PassthroughSpec(layer_import_mode=LayerImportMode.FLATTEN)
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload

    assert payload is not None
    item = payload.items[0]
    # Content rect at (20,20). Padded bounds: 19.8, 19.8, 20.4, 20.4
    # Y-inverted: 100 - (19.8+20.4) = 59.8
    wx, wy = item.matrix.transform_point((0, 0))
    assert wx == pytest.approx(19.8)
    assert wy == pytest.approx(59.8)


def test_vectorize_handles_layerless_svg():
    """Ensure Vectorize handles SVGs without explicit layer groups."""
    importer = SvgVectorImporter(SVG_BASIC)
    parse_res = importer.parse()
    assert parse_res is not None
    # Parse will find no explicit layers
    assert len(parse_res.layers) == 0

    # Vectorize should find the geometry anyway via its fallback.
    result = importer.vectorize(parse_res, PassthroughSpec())
    assert result is not None
    # It should populate a single geometry under the 'None' key
    assert len(result.geometries_by_layer) == 1
    assert None in result.geometries_by_layer
    assert not result.geometries_by_layer[None].is_empty()


def test_scan_detects_color_layers():
    """Scan finds distinct colors and groups them as layers."""
    importer = SvgVectorImporter(SVG_COLORS, Path("colors.svg"))
    manifest = importer.scan()
    assert manifest is not None
    assert manifest.color_layers is not None
    by_id = {layer.id: layer for layer in manifest.color_layers}
    assert set(by_id) == {"#ff0000", "#00ff00"}
    assert by_id["#ff0000"].feature_count == 2
    assert by_id["#00ff00"].feature_count == 1
    # LayerInfo.color is an RGB 0-1 tuple
    assert by_id["#ff0000"].color == pytest.approx((1.0, 0.0, 0.0))
    assert by_id["#00ff00"].color == pytest.approx((0.0, 1.0, 0.0))


def test_scan_no_color_bucket():
    """Shapes without a color attribute go into the no-color bucket."""
    importer = SvgVectorImporter(SVG_NO_COLOR, Path("nocolor.svg"))
    manifest = importer.scan()
    assert manifest is not None
    by_id = {layer.id: layer for layer in manifest.color_layers}
    assert "#ff0000" in by_id
    assert "_no_color" in by_id
    assert by_id["_no_color"].color is None


def test_parse_by_color_sets_layer_color():
    """Parse with COLORS source returns layer geometries with colors."""
    importer = SvgVectorImporter(SVG_COLORS, Path("colors.svg"))
    import_result = importer.get_doc_items(
        PassthroughSpec(layer_source=LayerSource.COLORS)
    )
    assert import_result is not None
    assert import_result.parse_result is not None
    by_id = {
        layer.layer_id: layer for layer in import_result.parse_result.layers
    }
    assert set(by_id) == {"#ff0000", "#00ff00"}
    assert by_id["#ff0000"].color == "#ff0000"
    assert by_id["#00ff00"].color == "#00ff00"


def test_import_by_color_creates_colored_layers():
    """Importing by color creates one layer per color, tinted accordingly."""
    importer = SvgVectorImporter(SVG_COLORS, Path("colors.svg"))
    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None
    assert len(payload.items) == 2

    by_name = {item.name: item for item in payload.items}
    assert isinstance(by_name["Color #ff0000"], Layer)
    assert by_name["Color #ff0000"].color == "#ff0000"
    assert isinstance(by_name["Color #00ff00"], Layer)
    assert by_name["Color #00ff00"].color == "#00ff00"


def test_import_by_color_filters_deselected():
    """Only colors present in active_layer_ids are imported."""
    importer = SvgVectorImporter(SVG_COLORS, Path("colors.svg"))
    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
        active_layer_ids=["#ff0000"],
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None
    assert len(payload.items) == 1
    item = payload.items[0]
    assert isinstance(item, Layer)
    assert item.name == "Color #ff0000"
    assert item.color == "#ff0000"


def test_import_by_color_stroke_attribute():
    """Shapes are grouped by the chosen color attribute."""
    svg = b"""
    <svg width="100mm" height="100mm" viewBox="0 0 100 100"
         xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="10" height="10" stroke="#0000ff"/>
        <rect x="50" y="50" width="10" height="10" stroke="#00ff00"/>
    </svg>
    """
    importer = SvgVectorImporter(svg, Path("stroke.svg"))
    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        color_attr=ColorAttr.STROKE,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None
    by_name = {item.name: item for item in payload.items}
    assert set(by_name) == {"Color #0000ff", "Color #00ff00"}
    assert isinstance(by_name["Color #0000ff"], Layer)
    assert by_name["Color #0000ff"].color == "#0000ff"


def test_facade_imports_by_color():
    """The SvgImporter facade preserves the color layer source."""
    importer = SvgImporter(SVG_COLORS, Path("colors.svg"))
    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None
    by_name = {item.name: item for item in payload.items}
    assert set(by_name) == {"Color #ff0000", "Color #00ff00"}
    assert isinstance(by_name["Color #ff0000"], Layer)
    assert by_name["Color #ff0000"].color == "#ff0000"
    assert isinstance(by_name["Color #00ff00"], Layer)
    assert by_name["Color #00ff00"].color == "#00ff00"


def test_facade_default_spec_merges():
    """A default PassthroughSpec still merges everything into one item."""
    importer = SvgImporter(SVG_COLORS, Path("colors.svg"))
    import_result = importer.get_doc_items(PassthroughSpec())
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None
    assert len(payload.items) == 1


def test_any_mode_splits_fill_and_stroke():
    """Shapes with differing fill and stroke yield two color layers."""
    svg = b"""
    <svg width="100mm" height="100mm" viewBox="0 0 100 100"
         xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="10" height="10"
              fill="#ff0000" stroke="#0000ff"/>
    </svg>
    """
    importer = SvgVectorImporter(svg, Path("dual.svg"))
    manifest = importer.scan()
    by_id = {layer.id: layer for layer in manifest.color_layers}
    assert set(by_id) == {"#ff0000", "#0000ff"}

    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None
    by_name = {item.name: item for item in payload.items}
    assert set(by_name) == {"Color #ff0000", "Color #0000ff"}
    assert isinstance(by_name["Color #ff0000"], Layer)
    assert by_name["Color #ff0000"].color == "#ff0000"
    assert isinstance(by_name["Color #0000ff"], Layer)
    assert by_name["Color #0000ff"].color == "#0000ff"


def test_any_mode_uses_stroke_color_by_default():
    """Stroke colors are captured without selecting a color attribute."""
    svg = b"""
    <svg width="100mm" height="100mm" viewBox="0 0 100 100"
         xmlns="http://www.w3.org/2000/svg">
        <rect x="0" y="0" width="10" height="10" stroke="#0000ff"/>
        <rect x="50" y="50" width="10" height="10" stroke="#00ff00"/>
    </svg>
    """
    importer = SvgVectorImporter(svg, Path("stroke.svg"))
    manifest = importer.scan()
    by_id = {layer.id: layer for layer in manifest.color_layers}
    assert set(by_id) == {"#0000ff", "#00ff00"}

    spec = PassthroughSpec(
        layer_source=LayerSource.COLORS,
        layer_import_mode=LayerImportMode.NEW_LAYERS,
    )
    import_result = importer.get_doc_items(spec)
    assert import_result is not None
    payload = import_result.payload
    assert payload is not None
    by_name = {item.name: item for item in payload.items}
    assert set(by_name) == {"Color #0000ff", "Color #00ff00"}
