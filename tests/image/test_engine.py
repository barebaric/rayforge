from typing import List, Tuple, Optional, Dict
import pytest

from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.image.engine import NormalizationEngine
from rayforge.image.structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
)
from rayforge.core.geo import Geometry


@pytest.fixture
def engine():
    return NormalizationEngine()


def create_vec_result(
    page_bounds: Tuple[float, float, float, float],
    layers: List[Tuple[str, Tuple[float, float, float, float]]],
    is_y_down: bool = True,
    unit_scale: float = 1.0,
) -> VectorizationResult:
    """
    Helper to construct a ParsingResult and wrap it in a
    VectorizationResult.
    """
    layer_geos = [
        LayerGeometry(layer_id=lid, content_bounds=bounds)
        for lid, bounds in layers
    ]
    parse_result = ParsingResult(
        page_bounds=page_bounds,
        native_unit_to_mm=unit_scale,
        is_y_down=is_y_down,
        layers=layer_geos,
    )
    # The engine currently doesn't use the geometry, so we can mock it.
    geometries: Dict[Optional[str], Geometry] = {
        layer.layer_id: Geometry() for layer in parse_result.layers
    }
    return VectorizationResult(
        geometries_by_layer=geometries, source_parse_result=parse_result
    )


def test_single_layer_y_down(engine):
    """
    Verifies standard behavior for SVG/Images (Y-Down).
    Content should be positioned relative to the bottom-left of the page
    in World Space.
    """
    # Page: 100x100
    # Content: 10x10 square at (10, 10) from top-left.
    # Bottom of content is at Y=20.
    # Distance from bottom of page = 100 - 20 = 80.
    vec_result = create_vec_result(
        page_bounds=(0, 0, 100, 100),
        layers=[("layer1", (10, 10, 10, 10))],
        is_y_down=True,
    )

    plan = engine.calculate_layout(vec_result, None)

    assert len(plan) == 1
    item = plan[0]

    # 1. Check Crop Window (should match content)
    assert item.crop_window == (10, 10, 10, 10)

    # 2. Check Normalization (Native -> 0-1)
    # Top-Left of content (10, 10) should map to (0, 0)
    p_norm = item.normalization_matrix.transform_point((10, 10))
    assert p_norm[0] == pytest.approx(0.0)
    assert p_norm[1] == pytest.approx(0.0)

    # Bottom-Right of content (20, 20) should map to (1, 1)
    p_norm_end = item.normalization_matrix.transform_point((20, 20))
    assert p_norm_end[0] == pytest.approx(1.0)
    assert p_norm_end[1] == pytest.approx(1.0)

    # 3. Check World Position (0-1 -> mm)
    # WorkPiece origin (0,0) in Rayforge is Bottom-Left.
    # Based on Y-Down logic, Y=80.
    # X is unchanged = 10.
    tx, ty = item.world_matrix.get_translation()
    assert tx == pytest.approx(10.0)
    assert ty == pytest.approx(80.0)

    # Check Size
    w, h = item.world_matrix.get_abs_scale()
    assert w == pytest.approx(10.0)
    assert h == pytest.approx(10.0)


def test_single_layer_y_up(engine):
    """
    Verifies standard behavior for DXF (Y-Up).
    Coordinates should map directly.
    """
    # Content: 10x10 square at (10, 10) from bottom-left.
    vec_result = create_vec_result(
        page_bounds=(0, 0, 100, 100),
        layers=[("layer1", (10, 10, 10, 10))],
        is_y_down=False,  # DXF style
    )

    plan = engine.calculate_layout(vec_result, None)
    item = plan[0]

    # Y position should be 10 directly
    tx, ty = item.world_matrix.get_translation()
    assert tx == pytest.approx(10.0)
    assert ty == pytest.approx(10.0)


def test_merge_layers_union(engine):
    """
    Verifies that multiple layers are merged into a single item
    representing their union bounding box by default.
    """
    # Layer A: (0, 0, 10, 10)
    # Layer B: (20, 20, 10, 10)
    # Union: (0, 0, 30, 30)
    vec_result = create_vec_result(
        page_bounds=(0, 0, 100, 100),
        layers=[
            ("A", (0, 0, 10, 10)),
            ("B", (20, 20, 10, 10)),
        ],
    )

    plan = engine.calculate_layout(vec_result, None)

    assert len(plan) == 1
    item = plan[0]

    # Check Union Crop Window
    assert item.crop_window == (0, 0, 30, 30)

    # Check World Size (30x30)
    w, h = item.world_matrix.get_abs_scale()
    assert w == pytest.approx(30.0)
    assert h == pytest.approx(30.0)


def test_split_layers_alignment(engine):
    """
    Verifies that splitting layers maintains their relative visual
    positions in the World.
    """
    # Page Height: 100
    # Layer Top: (10, 10, 10, 10). Bottom Y = 20. World Y = 80.
    # Layer Bottom: (10, 80, 10, 10). Bottom Y = 90. World Y = 10.
    # Visual distance: 70 units vertically.
    vec_result = create_vec_result(
        page_bounds=(0, 0, 100, 100),
        layers=[
            ("Top", (10, 10, 10, 10)),
            ("Bottom", (10, 80, 10, 10)),
        ],
    )

    spec = PassthroughSpec(active_layer_ids=["Top", "Bottom"])
    plan = engine.calculate_layout(vec_result, spec)

    assert len(plan) == 2

    # Identify items
    item_top = next(i for i in plan if i.layer_id == "Top")
    item_bottom = next(i for i in plan if i.layer_id == "Bottom")

    _, y_top = item_top.world_matrix.get_translation()
    _, y_bottom = item_bottom.world_matrix.get_translation()

    assert y_top == pytest.approx(80.0)
    assert y_bottom == pytest.approx(10.0)
    assert y_top - y_bottom == pytest.approx(70.0)


def test_unit_conversion(engine):
    """
    Verifies that native units are correctly converted to mm.
    """
    # 1 Native Unit = 2.0 mm
    vec_result = create_vec_result(
        page_bounds=(0, 0, 100, 100),
        layers=[("L1", (0, 0, 10, 10))],
        unit_scale=2.0,
    )

    plan = engine.calculate_layout(vec_result, None)
    item = plan[0]

    # Size should be 10 * 2.0 = 20mm
    w, h = item.world_matrix.get_abs_scale()
    assert w == pytest.approx(20.0)
    assert h == pytest.approx(20.0)


def test_empty_content_fallback(engine):
    """
    Verifies that if content bounds are missing/zero, it falls back
    to the page bounds.
    """
    vec_result = create_vec_result(
        page_bounds=(0, 0, 100, 100),
        layers=[],  # No layers
    )

    plan = engine.calculate_layout(vec_result, None)

    assert len(plan) == 1
    item = plan[0]
    # Should match page bounds
    assert item.crop_window == (0, 0, 100, 100)
