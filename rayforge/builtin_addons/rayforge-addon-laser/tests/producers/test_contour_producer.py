import pytest
import cairo
from typing import List
from raygeo import Geometry
from rayforge.core.workpiece import WorkPiece
from rayforge.core.source_asset_segment import SourceAssetSegment
from rayforge.core.vectorization_spec import PassthroughSpec
from rayforge.pipeline.producer.base import OpsProducer, CutSide
from rayforge.machine.models.laser import Laser
from rayforge.pipeline.artifact import WorkPieceArtifact
from laser_essentials.producers import ContourProducer, CutOrder


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
        pristine_geometry=combined,
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
    assert recreated.cut_side.value == CutSide.INSIDE.value
    assert recreated.cut_order.value == CutOrder.OUTSIDE_INSIDE.value


def test_centerline_preserves_geometry(laser, dummy_surface, vector_workpiece):
    """Test CutSide.CENTERLINE (No offset)."""
    tracer = ContourProducer(cut_side=CutSide.CENTERLINE)

    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
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
        generation_id=1,
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
        generation_id=1,
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
        generation_id=1,
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
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
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
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
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
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
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
        laser,
        dummy_surface,
        (10, 10),
        workpiece=wp,
        settings=settings,
        generation_id=1,
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


def test_overcut_extends_closed_contour(laser, dummy_surface):
    """Overcut extends a closed rectangle past its start point."""
    rect = create_rect(0, 0, 1.0, 1.0, cw=False)
    wp = WorkPiece(name="overcut_test")
    wp._boundaries_cache = rect
    wp.set_size(20, 20)

    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE,
        overcut=5.0,
    )

    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=wp,
        generation_id=1,
    )
    geoms = get_geo_from_artifact(artifact)
    assert len(geoms) == 1

    data = geoms[0].data
    assert data is not None
    last_x = data[-1, 1]
    last_y = data[-1, 2]
    # First side goes (0,0)→(20,0), 20mm.  Overcut 5mm → (5,0).
    assert last_x == pytest.approx(5.0, abs=0.01)
    assert last_y == pytest.approx(0.0, abs=0.01)


def test_overcut_zero_is_noop(laser, dummy_surface, vector_workpiece):
    """overcut=0 must not change the output."""
    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE,
        overcut=0.0,
    )
    artifact_a = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
    )

    tracer2 = ContourProducer(cut_side=CutSide.CENTERLINE)
    artifact_b = tracer2.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
    )

    geo_a = Geometry.from_dict(artifact_a.ops.to_dict())
    geo_b = Geometry.from_dict(artifact_b.ops.to_dict())
    assert geo_a == geo_b


def test_overcut_serialization():
    """overcut is preserved through serialization."""
    producer = ContourProducer(overcut=3.5)
    data = producer.to_dict()
    recreated = OpsProducer.from_dict(data)
    assert isinstance(recreated, ContourProducer)
    assert recreated.overcut == 3.5


def test_overcut_with_remove_inner_paths(
    laser, dummy_surface, vector_workpiece
):
    """Overcut works when remove_inner_paths=True."""
    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE,
        remove_inner_paths=True,
        overcut=5.0,
    )
    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
    )
    geoms = get_geo_from_artifact(artifact)
    assert len(geoms) == 1

    data = geoms[0].data
    assert data is not None
    last_x = data[-1, 1]
    last_y = data[-1, 2]
    assert last_x == pytest.approx(5.0, abs=0.01)
    assert last_y == pytest.approx(0.0, abs=0.01)


def test_overcut_applies_to_both_inner_and_outer(
    laser, dummy_surface, vector_workpiece
):
    """Overcut is applied to both outer and inner (hole) contours."""
    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE,
        overcut=5.0,
    )
    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=vector_workpiece,
        generation_id=1,
    )
    geoms = get_geo_from_artifact(artifact)
    assert len(geoms) == 2

    for geo in geoms:
        data = geo.data
        assert data is not None
        start_x, start_y = data[0, 1], data[0, 2]
        end_x, end_y = data[-1, 1], data[-1, 2]
        # With overcut the path no longer ends at the start point
        assert not (
            abs(end_x - start_x) < 0.01 and abs(end_y - start_y) < 0.01
        )


def test_overcut_larger_than_one_side(laser, dummy_surface):
    """Overcut that spans more than one segment still works."""
    rect = create_rect(0, 0, 1.0, 1.0, cw=False)
    wp = WorkPiece(name="overcut_large")
    wp._boundaries_cache = rect
    wp.set_size(20, 20)

    # First side is 20mm, so 25mm overcut goes past it.
    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE,
        overcut=25.0,
    )
    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=wp,
        generation_id=1,
    )
    geoms = get_geo_from_artifact(artifact)
    assert len(geoms) == 1

    data = geoms[0].data
    assert data is not None
    last_x = data[-1, 1]
    last_y = data[-1, 2]
    # 25mm: first side (20mm) fully traced, then 5mm into second side
    # Second side: (20,0)→(20,20).  5mm/20mm → t=0.25 → (20, 5)
    assert last_x == pytest.approx(20.0, abs=0.01)
    assert last_y == pytest.approx(5.0, abs=0.01)


def test_overcut_on_full_circle(laser, dummy_surface):
    """Overcut works on a full-circle arc (start == end)."""
    # Normalized circle: center (0.5,0.5), radius 0.5, start at (1,0.5)
    geo = Geometry()
    geo.move_to(1, 0.5)
    geo.arc_to(1, 0.5, i=-0.5, j=0, clockwise=False)

    wp = WorkPiece(name="circle_overcut")
    wp._boundaries_cache = geo
    wp.set_size(120, 120)

    tracer = ContourProducer(
        cut_side=CutSide.CENTERLINE,
        overcut=3.0,
    )
    artifact = tracer.run(
        laser,
        dummy_surface,
        (10, 10),
        workpiece=wp,
        generation_id=1,
    )
    geoms = get_geo_from_artifact(artifact)
    assert len(geoms) == 1

    data = geoms[0].data
    assert data is not None
    # After scaling: circle radius 60, center (60,60), start (120,60)
    # Overcut 3mm → angle = 3/60 = 0.05 rad CCW from angle 0.
    # x = 60 + 60·cos(0.05) ≈ 119.93, y = 60 + 60·sin(0.05) ≈ 63.0
    last_x = data[-1, 1]
    last_y = data[-1, 2]
    assert last_x == pytest.approx(119.93, abs=0.05)
    assert last_y == pytest.approx(63.0, abs=0.05)
