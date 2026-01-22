import pytest
from pathlib import Path
from rayforge.image.svg.svg_vector import SvgVectorImporter
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.core.workpiece import WorkPiece
from rayforge.core.layer import Layer

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
    assert px == pytest.approx(9.8)
    assert py == pytest.approx(9.8)
    assert pw == pytest.approx(20.4)
    assert ph == pytest.approx(20.4)
    assert result.native_unit_to_mm == pytest.approx(1.0)
    assert result.is_y_down is True
    # The parse method now extracts actual layers, it doesn't create a default
    # for layerless SVGs.
    assert len(result.layers) == 0


def test_get_doc_items_alignment(vector_importer):
    """Verifies World Space alignment."""
    # Use a spec that forces merge, which is the default for layerless SVGs
    spec = PassthroughSpec(create_new_layers=False)
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
    assert wx == pytest.approx(9.8)
    assert wy == pytest.approx(69.8)


def test_layer_separation_and_positioning():
    """Test importing layers creates separate items positioned correctly."""
    importer = SvgVectorImporter(SVG_GROUPS, Path("groups.svg"))
    manifest = importer.scan()
    layer_ids = [layer.id for layer in manifest.layers]
    spec = PassthroughSpec(active_layer_ids=layer_ids, create_new_layers=True)
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
    spec = PassthroughSpec(create_new_layers=False)
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
