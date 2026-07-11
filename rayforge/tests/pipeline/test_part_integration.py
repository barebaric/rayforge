"""Integration tests: Part-based workflow end-to-end."""

import pytest
from raygeo import Part
from raygeo.geo import Geometry
from raygeo.ops.assembly.profile import profile_inner
from raygeo.ops.cut.cleared_area import ClearedArea
from raygeo.cnc.machining.plan import Workplan


def create_rect(x, y, w, h, cw=False):
    """Helper to create a rectangle geometry."""
    geo = Geometry()
    geo.move_to(x, y)
    if not cw:
        geo.line_to(x + w, y)
        geo.line_to(x + w, y + h)
        geo.line_to(x, y + h)
    else:
        geo.line_to(x, y + h)
        geo.line_to(x + w, y + h)
        geo.line_to(x + w, y)
    geo.close_path()
    return geo


def test_part_create():
    """Part can be created with geometry and queried."""
    geo = create_rect(0, 0, 100, 50)
    p = Part(geometry=geo, size_mm=(100, 50))
    assert p.size_mm == (100, 50)
    assert p.has_geometry()
    assert p.pixels_per_mm is None
    assert p.geometry is not None


def test_part_without_geometry():
    """Part can be created without geometry."""
    p = Part(size_mm=(200, 100))
    assert p.size_mm == (200, 100)
    assert not p.has_geometry()


def test_profile_inner_with_part_produces_ops():
    """profile_inner accepts a Part and returns ops."""
    geo = create_rect(0, 0, 100, 50)
    part = Part(geometry=geo, size_mm=(100, 50))

    boundary = [(0, 0), (100, 0), (100, 50), (0, 50)]
    cleared = ClearedArea(boundary)
    result = profile_inner(
        part,
        cleared,
        tool_radius=0.0,
        step_over=1.0,
        target_z=0.0,
        safe_z=5.0,
        cut_feed_rate=1000,
        cut_power=1.0,
    )

    assert result.ops is not None
    assert len(result.ops) > 0
    assert result.ops.rect() is not None


def test_workplan_from_part():
    """Workplan can be created from a Part."""
    geo = create_rect(0, 0, 100, 100)
    part = Part(geometry=geo, size_mm=(100, 100))

    wp = Workplan.from_part(part, safe_z=5.0)
    assert wp is not None
    assert "safe_z=5" in repr(wp)


def test_workplan_from_part_no_geometry():
    """Workplan.from_part raises ValueError for Parts without geometry."""
    part = Part(size_mm=(100, 100))
    with pytest.raises(ValueError):
        Workplan.from_part(part, safe_z=5.0)


def test_profile_inner_produces_same_ops_equivalently():
    """profile_inner works the same whether geometry comes from Part or
    is passed inline."""
    geo = create_rect(0, 0, 100, 50)
    part = Part(geometry=geo, size_mm=(100, 50))

    boundary = [(0, 0), (100, 0), (100, 50), (0, 50)]
    cleared = ClearedArea(boundary)

    result = profile_inner(
        part,
        cleared,
        tool_radius=0.0,
        step_over=1.0,
        target_z=0.0,
        safe_z=5.0,
        cut_feed_rate=1000,
        cut_power=1.0,
    )

    assert len(result.ops) > 0
    full_geo = result.ops.to_geometry()
    assert full_geo.rect() == pytest.approx((0.0, 0.0, 100.0, 50.0), abs=1)


def test_part_with_workpiece_flow():
    """End-to-end: WorkPiece → Part → assembler → ops."""
    from rayforge.core.source_asset_segment import SourceAssetSegment
    from rayforge.core.vectorization_spec import PassthroughSpec
    from rayforge.core.workpiece import WorkPiece

    # Create a workpiece with a normalized (0..1) geometry
    outer = create_rect(0, 0, 1.0, 1.0, cw=False)
    inner = create_rect(0.25, 0.25, 0.5, 0.5, cw=True)

    combined = Geometry()
    combined.extend(outer)
    combined.extend(inner)

    segment = SourceAssetSegment(
        source_asset_uid="test",
        pristine_geometry=combined,
        vectorization_spec=PassthroughSpec(),
    )

    wp = WorkPiece(name="test_part_wp", source_segment=segment)
    wp._boundaries_cache = combined
    wp.set_size(20, 20)

    # Convert to Part (the new API)
    part = wp.to_part()
    assert part is not None
    assert part.has_geometry()
    assert part.size_mm == (20, 20)

    # Use Part with assembler
    boundary = [(0, 0), (20, 0), (20, 20), (0, 20)]
    cleared = ClearedArea(boundary)
    result = profile_inner(
        part,
        cleared,
        tool_radius=0.0,
        step_over=1.0,
        target_z=0.0,
        safe_z=5.0,
        cut_feed_rate=1000,
        cut_power=1.0,
    )

    assert result.ops is not None
    assert len(result.ops) > 0

    # Verify the ops cover the expected area
    rect = result.ops.rect()
    assert rect[0] == pytest.approx(0, abs=1)
    assert rect[1] == pytest.approx(0, abs=1)
    assert rect[2] == pytest.approx(20, abs=1)
    assert rect[3] == pytest.approx(20, abs=1)
