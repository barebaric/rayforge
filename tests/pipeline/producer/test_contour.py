import pytest
import cairo
from typing import List
from rayforge.core.workpiece import WorkPiece
from rayforge.core.geo import Geometry
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.pipeline.producer.contour import (
    ContourProducer,
    CutSide,
    CutOrder,
)
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.machine.models.laser import Laser
from rayforge.pipeline.artifact import WorkPieceArtifact


# --- Helpers ---


def create_rect(
    x: float, y: float, w: float, h: float, cw: bool = False
) -> Geometry:
    """Helper to create a rectangle geometry."""
    geo = Geometry()
    geo.move_to(x, y)
    if not cw:
        # CCW (Standard for outer boundaries)
        geo.line_to(x + w, y)
        geo.line_to(x + w, y + h)
        geo.line_to(x, y + h)
    else:
        # CW (Standard for holes, though the tracer should auto-normalize)
        geo.line_to(x, y + h)
        geo.line_to(x + w, y + h)
        geo.line_to(x + w, y)
    geo.close_path()
    return geo


def get_geo_from_artifact(artifact: WorkPieceArtifact) -> List[Geometry]:
    """
    Extracts individual contours from the generated Ops.
    Ops are flattened, so we reconstruct geometries based on MoveTo commands.
    """
    # Create a single geometry from all ops
    full_geo = Geometry.from_dict(artifact.ops.to_dict())
    # Split into individual closed loops (contours)
    return full_geo.split_into_contours()


# --- Fixtures ---


@pytest.fixture
def laser() -> Laser:
    laser_instance = Laser()
    # 0.1mm spot size implies default kerf is 0.1mm
    laser_instance.spot_size_mm = (0.1, 0.1)
    return laser_instance


@pytest.fixture
def dummy_surface() -> cairo.ImageSurface:
    """Vector producers still require a surface argument in the signature."""
    return cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)


@pytest.fixture
def vector_workpiece() -> WorkPiece:
    """
    Creates a workpiece setup for vector data (Passthrough).
    Physical Size: 20x20 mm.
    Internal Geometry: Normalized (0..1).
    """
    # Outer: (0,0) to (1,1) -> scales to 0..20
    outer = create_rect(0, 0, 1.0, 1.0, cw=False)
    # Inner: (0.25, 0.25) to (0.75, 0.75) -> scales to 5..15 (10x10 hole)
    inner = create_rect(0.25, 0.25, 0.5, 0.5, cw=True)  # CW hole

    combined = Geometry()
    combined.extend(outer)
    combined.extend(inner)

    segment = SourceAssetSegment(
        source_asset_uid="test",
        segment_mask_geometry=combined,
        # Note: This gets stored as Y-down, but for Passthrough logic in tests
        # we often mock the boundaries directly
        vectorization_spec=PassthroughSpec(),
    )

    wp = WorkPiece(name="test_geo", source_segment=segment)
    # We manually inject the boundaries into the cache to bypass the
    # SourceAssetSegment Y-flip logic for simpler testing coordinates
    # (0,0 bottom-left).
    wp._boundaries_cache = combined
    wp.set_size(20, 20)  # Matrix scale = 20

    return wp


# --- Tests ---


def test_serialization():
    """Test to_dict and from_dict for ContourProducer."""
    tracer = ContourProducer(
        remove_inner_paths=True,
        path_offset_mm=0.5,
        cut_side=CutSide.INSIDE,
        cut_order=CutOrder.OUTSIDE_INSIDE,
    )
    data = tracer.to_dict()
    recreated = OpsProducer.from_dict(data)

    assert isinstance(recreated, ContourProducer)
    assert recreated.remove_inner_paths is True
    assert recreated.path_offset_mm == 0.5
    assert recreated.cut_side == CutSide.INSIDE
    assert recreated.cut_order == CutOrder.OUTSIDE_INSIDE


def test_centerline_preserves_geometry(laser, dummy_surface, vector_workpiece):
    """Test CutSide.CENTERLINE (No offset)."""
    tracer = ContourProducer(cut_side=CutSide.CENTERLINE)

    artifact = tracer.run(
        laser, dummy_surface, (10, 10), workpiece=vector_workpiece
    )

    geoms = get_geo_from_artifact(artifact)

    # Should have 2 components (Outer and Inner)
    assert len(geoms) == 2

    # Bounds should match original exactly
    rects = sorted(
        [g.rect() for g in geoms], key=lambda r: r[2] - r[0]
    )  # Sort by width

    hole, outer = rects[0], rects[1]

    # Hole 10x10 at 5,5
    assert hole == pytest.approx((5.0, 5.0, 15.0, 15.0), abs=1e-5)
    # Outer 20x20 at 0,0
    assert outer == pytest.approx((0.0, 0.0, 20.0, 20.0), abs=1e-5)


def test_outside_cut_expands_outer_shrinks_hole(
    laser, dummy_surface, vector_workpiece
):
    """
    Test CutSide.OUTSIDE with Kerf.
    Logic:
    - Outer perimeter (Solid) should EXPAND.
    - Inner hole (Hole) should SHRINK.

    Kerf = 1.0mm (override in settings).
    Offset = Kerf / 2 = 0.5mm.
    """
    tracer = ContourProducer(cut_side=CutSide.OUTSIDE)
    settings = {"kerf_mm": 1.0}

    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        settings=settings,
    )
    geoms = get_geo_from_artifact(artifact)
    assert len(geoms) == 2

    rects = sorted([g.rect() for g in geoms], key=lambda r: r[2] - r[0])
    hole, outer = rects[0], rects[1]

    # Hole (Shrink by 0.5 on all sides): 5+0.5 to 15-0.5
    # -> 5.5 to 14.5 (Size 9x9)
    assert hole == pytest.approx((5.5, 5.5, 14.5, 14.5), abs=0.05)

    # Outer (Expand by 0.5 on all sides): 0-0.5 to 20+0.5
    # -> -0.5 to 20.5 (Size 21x21)
    assert outer == pytest.approx((-0.5, -0.5, 20.5, 20.5), abs=0.05)


def test_inside_cut_shrinks_outer_expands_hole(
    laser, dummy_surface, vector_workpiece
):
    """
    Test CutSide.INSIDE with Kerf.
    Logic:
    - Outer perimeter (Solid) should SHRINK.
    - Inner hole (Hole) should EXPAND.

    Kerf = 1.0mm. Offset = -0.5mm.
    """
    tracer = ContourProducer(cut_side=CutSide.INSIDE)
    settings = {"kerf_mm": 1.0}

    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        settings=settings,
    )
    geoms = get_geo_from_artifact(artifact)
    assert len(geoms) == 2

    rects = sorted([g.rect() for g in geoms], key=lambda r: r[2] - r[0])
    hole, outer = rects[0], rects[1]

    # Hole (Expand): 5-0.5 to 15+0.5 -> 4.5 to 15.5 (Size 11x11)
    assert hole == pytest.approx((4.5, 4.5, 15.5, 15.5), abs=0.05)

    # Outer (Shrink): 0+0.5 to 20-0.5 -> 0.5 to 19.5 (Size 19x19)
    assert outer == pytest.approx((0.5, 0.5, 19.5, 19.5), abs=0.05)


def test_manual_path_offset(laser, dummy_surface, vector_workpiece):
    """
    Test explicit path_offset_mm combined with CutSide.
    Offset = 2.0mm. CutSide=OUTSIDE. Kerf=0.
    Total Offset = +2.0mm.
    """
    tracer = ContourProducer(cut_side=CutSide.OUTSIDE, path_offset_mm=2.0)
    settings = {"kerf_mm": 0.0}

    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        settings=settings,
    )
    geoms = get_geo_from_artifact(artifact)

    rects = sorted([g.rect() for g in geoms], key=lambda r: r[2] - r[0])
    hole, outer = rects[0], rects[1]

    # Hole (Shrink by 2.0): 5+2 to 15-2 -> 7 to 13 (Size 6x6)
    assert hole == pytest.approx((7.0, 7.0, 13.0, 13.0), abs=0.05)

    # Outer (Expand by 2.0): -2 to 22 (Size 24x24)
    assert outer == pytest.approx((-2.0, -2.0, 22.0, 22.0), abs=0.05)


def test_remove_inner_paths(laser, dummy_surface, vector_workpiece):
    """Test that remove_inner_paths=True deletes the hole."""
    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE, remove_inner_paths=True
    )

    artifact = tracer.run(
        laser, dummy_surface, (10, 10), workpiece=vector_workpiece
    )
    geoms = get_geo_from_artifact(artifact)

    # Should only have 1 component (The outer 20x20)
    assert len(geoms) == 1
    assert geoms[0].rect() == pytest.approx((0.0, 0.0, 20.0, 20.0), abs=1e-5)


def test_cut_order_inside_outside(laser, dummy_surface, vector_workpiece):
    """
    Test CutOrder.INSIDE_OUTSIDE.
    The artifact ops are grouped. The first group of ops generated should
    be the hole.
    """
    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE, cut_order=CutOrder.INSIDE_OUTSIDE
    )
    artifact = tracer.run(
        laser, dummy_surface, (10, 10), workpiece=vector_workpiece
    )

    # Extract components based on command order.
    # The artifact.ops object contains sections. We need to convert to
    # geometries sequentially.

    full_geo = Geometry.from_dict(artifact.ops.to_dict())
    # split_into_contours preserves order of occurrence in command list
    contours = full_geo.split_into_contours()

    assert len(contours) == 2
    first, second = contours[0], contours[1]

    # First should be the hole (smaller area/bounds)
    # Hole area ~ 100, Outer area ~ 400
    assert first.area() < second.area()
    assert first.rect() == pytest.approx((5.0, 5.0, 15.0, 15.0), abs=1e-5)


def test_cut_order_outside_inside(laser, dummy_surface, vector_workpiece):
    """
    Test CutOrder.OUTSIDE_INSIDE.
    The first group of ops generated should be the outer perimeter.
    """
    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE, cut_order=CutOrder.OUTSIDE_INSIDE
    )
    artifact = tracer.run(
        laser, dummy_surface, (10, 10), workpiece=vector_workpiece
    )

    full_geo = Geometry.from_dict(artifact.ops.to_dict())
    contours = full_geo.split_into_contours()

    assert len(contours) == 2
    first, second = contours[0], contours[1]

    # First should be the outer perimeter (larger area)
    assert first.area() > second.area()
    assert first.rect() == pytest.approx((0.0, 0.0, 20.0, 20.0), abs=1e-5)


def test_winding_normalization_on_import(laser, dummy_surface):
    """
    If we supply a geometry where the hole has the WRONG winding (CCW, same
    as outer), CutSide.OUTSIDE would normally interpret it as a solid and
    expand it, potentially overlapping or confusing the logic.

    The fix in ContourProducer.run should call normalize_winding_orders,
    forcing the hole to CW before offset.
    """
    # Create a geometry with WRONG winding for the hole (CCW)
    # Normalized coordinates (0..1)
    outer = create_rect(0, 0, 1.0, 1.0, cw=False)  # CCW
    inner = create_rect(
        0.25, 0.25, 0.5, 0.5, cw=False
    )  # CCW (Wrong! Should be CW)

    bad_winding_geo = Geometry()
    bad_winding_geo.extend(outer)
    bad_winding_geo.extend(inner)

    wp = WorkPiece(name="bad_winding")
    wp._boundaries_cache = bad_winding_geo
    wp.set_size(20, 20)

    # CutSide.OUTSIDE + 1mm Kerf -> Offset +0.5mm
    tracer = ContourProducer(cut_side=CutSide.OUTSIDE)
    settings = {"kerf_mm": 1.0}

    artifact = tracer.run(
        laser, dummy_surface, (10, 10), workpiece=wp, settings=settings
    )
    geoms = get_geo_from_artifact(artifact)

    # If normalization works:
    # 1. Inner loop becomes CW.
    # 2. Offset +0.5mm on a CW loop (hole) means shrinking the hole.
    # 3. Result: Hole gets smaller.

    # If normalization FAILED:
    # 1. Inner loop stays CCW.
    # 2. Offset +0.5mm on a CCW loop (solid) means expanding.
    # 3. Result: Hole gets bigger.

    assert len(geoms) == 2
    rects = sorted([g.rect() for g in geoms], key=lambda r: r[2] - r[0])
    hole = rects[0]

    # Expect hole to shrink (10x10 -> 9x9)
    # Bounds: 5.5 to 14.5
    assert hole == pytest.approx((5.5, 5.5, 14.5, 14.5), abs=0.05)
