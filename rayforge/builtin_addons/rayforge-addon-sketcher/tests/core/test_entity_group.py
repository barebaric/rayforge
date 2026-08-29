import math
from typing import Any

import pytest
from sketcher.core.arrays import InstancePlacement, PlacementKind
from sketcher.core.constraints import RadiusConstraint
from sketcher.core.entity_group import (
    EntityGroup,
    points_bbox_center,
    remap_point_refs,
)
from sketcher.core.registry import EntityRegistry


def get_entity(reg: EntityRegistry, eid: int) -> Any:
    """Returns the entity untyped so tests can assert concrete
    subclass attributes."""
    return reg.get_entity(eid)


class _ScalePlacement:
    """A minimal PlacementTransform: uniform scale about the origin."""

    def __init__(self, k: float):
        self.k = k

    def transform_point(self, x: float, y: float) -> tuple[float, float]:
        return (x * self.k, y * self.k)

    def transform_offset(self, dx: float, dy: float) -> tuple[float, float]:
        return (dx * self.k, dy * self.k)


def add_square(reg: EntityRegistry, ox=0.0, oy=0.0, size=10.0):
    """Adds a square as two lines sharing the diagonal-free corner."""
    p1 = reg.add_point(ox, oy)
    p2 = reg.add_point(ox + size, oy)
    shared = reg.add_point(ox + size, oy + size)
    p4 = reg.add_point(ox, oy + size)
    top = reg.add_line(p4, shared)
    right = reg.add_line(p2, shared)
    return p1, p2, shared, p4, top, right


# ----------------------------------------------------------------------
# Identity
# ----------------------------------------------------------------------


def test_point_ids_are_unique_and_stable():
    reg = EntityRegistry()
    _p1, p2, shared, p4, top, right = add_square(reg)
    group = EntityGroup(reg, [top, right, top])
    assert group.point_ids() == [p4, shared, p2]
    assert [p.id for p in group.points()] == [p4, shared, p2]
    assert [e.id for e in group.entities()] == [top, right, top]


def test_missing_entities_are_skipped():
    reg = EntityRegistry()
    p1 = reg.add_point(0.0, 0.0)
    p2 = reg.add_point(10.0, 0.0)
    line = reg.add_line(p1, p2)
    group = EntityGroup(reg, [999, line])
    assert [e.id for e in group.entities()] == [line]
    assert group.point_ids() == [p1, p2]


# ----------------------------------------------------------------------
# Center resolution
# ----------------------------------------------------------------------


def test_center_is_bbox_center_for_lines():
    reg = EntityRegistry()
    p1 = reg.add_point(0.0, 0.0)
    p2 = reg.add_point(10.0, 20.0)
    line = reg.add_line(p1, p2)
    group = EntityGroup(reg, [line])
    assert group.center() == (5.0, 10.0)


def test_center_prefers_explicit_circle_center():
    reg = EntityRegistry()
    c = reg.add_point(100.0, 50.0)
    r = reg.add_point(110.0, 50.0)
    circle = reg.add_circle(c, r)
    group = EntityGroup(reg, [circle])
    assert group.center() == (100.0, 50.0)


def test_center_prefers_explicit_ellipse_center():
    reg = EntityRegistry()
    c = reg.add_point(7.0, -3.0)
    rx = reg.add_point(9.0, -3.0)
    ry = reg.add_point(7.0, -1.0)
    ellipse = reg.add_ellipse(c, rx, ry)
    group = EntityGroup(reg, [ellipse])
    assert group.center() == (7.0, -3.0)


def test_center_falls_back_to_bbox_for_multiple_shapes():
    reg = EntityRegistry()
    c1 = reg.add_point(0.0, 0.0)
    r1 = reg.add_point(4.0, 0.0)
    c2 = reg.add_point(20.0, 0.0)
    r2 = reg.add_point(24.0, 0.0)
    circle1 = reg.add_circle(c1, r1)
    circle2 = reg.add_circle(c2, r2)
    group = EntityGroup(reg, [circle1, circle2])
    assert group.center() == (12.0, 0.0)


def test_points_bbox_center():
    reg = EntityRegistry()
    p1 = reg.add_point(-2.0, 8.0)
    p2 = reg.add_point(6.0, 12.0)
    p3 = reg.add_point(2.0, 4.0)
    assert points_bbox_center([reg.get_point(p) for p in (p1, p2, p3)]) == (
        2.0,
        8.0,
    )


# ----------------------------------------------------------------------
# Placement transforms
# ----------------------------------------------------------------------


def test_apply_placement_moves_shared_point_exactly_once():
    reg = EntityRegistry()
    _p1, _p2, shared, _p4, top, right = add_square(reg)
    group = EntityGroup(reg, [top, right])
    placement = InstancePlacement(
        kind=PlacementKind.ROTATION,
        angle=math.pi / 2,
        center=(0.0, 0.0),
    )
    group.apply_placement(placement)
    pt = reg.get_point(shared)
    assert pt.x == pytest.approx(-10.0)
    assert pt.y == pytest.approx(10.0)


def test_apply_placement_transforms_bezier_offsets():
    reg = EntityRegistry()
    start = reg.add_point(0.0, 0.0)
    end = reg.add_point(10.0, 0.0)
    bez = reg.add_bezier(start, end, cp1=(5.0, 2.0), cp2=(-1.0, 3.0))
    group = EntityGroup(reg, [bez])
    placement = InstancePlacement(
        kind=PlacementKind.ROTATION,
        angle=math.pi / 2,
        center=(0.0, 0.0),
    )
    group.apply_placement(placement)
    entity = get_entity(reg, bez)
    assert entity.cp1 == pytest.approx((-2.0, 5.0))
    assert entity.cp2 == pytest.approx((-3.0, -1.0))


def test_apply_placement_accepts_any_placement_transform():
    reg = EntityRegistry()
    p1 = reg.add_point(1.0, 2.0)
    p2 = reg.add_point(3.0, 2.0)
    line = reg.add_line(p1, p2)
    group = EntityGroup(reg, [line])
    group.apply_placement(_ScalePlacement(3.0))
    assert reg.get_point(p1).x == pytest.approx(3.0)
    assert reg.get_point(p1).y == pytest.approx(6.0)
    assert reg.get_point(p2).x == pytest.approx(9.0)
    assert reg.get_point(p2).y == pytest.approx(6.0)


def test_apply_rigid_motion_maps_center_onto_target():
    reg = EntityRegistry()
    a = reg.add_point(0.0, 0.0)
    b = reg.add_point(2.0, 0.0)
    line = reg.add_line(a, b)
    group = EntityGroup(reg, [line])
    motion = InstancePlacement(
        kind=PlacementKind.CURVE_ALIGNED,
        angle=math.pi / 2,
        center=(0.0, 0.0),
        target_center=(10.0, 0.0),
    )
    group.apply_rigid_motion(motion)
    assert reg.get_point(a).x == pytest.approx(10.0)
    assert reg.get_point(a).y == pytest.approx(0.0)
    assert reg.get_point(b).x == pytest.approx(10.0)
    assert reg.get_point(b).y == pytest.approx(2.0)


def test_translate_moves_shared_point_once():
    reg = EntityRegistry()
    _p1, _p2, shared, _p4, top, right = add_square(reg)
    group = EntityGroup(reg, [top, right])
    group.translate(5.0, -2.0)
    pt = reg.get_point(shared)
    assert pt.x == pytest.approx(15.0)
    assert pt.y == pytest.approx(8.0)


def test_radial_project_puts_center_on_circle():
    reg = EntityRegistry()
    c = reg.add_point(3.0, 4.0)
    r = reg.add_point(7.0, 4.0)
    circle = reg.add_circle(c, r)
    group = EntityGroup(reg, [circle])
    group.radial_project((0.0, 0.0), 10.0)
    cpt = reg.get_point(c)
    rpt = reg.get_point(r)
    assert math.hypot(cpt.x, cpt.y) == pytest.approx(10.0)
    assert math.hypot(rpt.x - cpt.x, rpt.y - cpt.y) == pytest.approx(4.0)


def test_radial_project_is_noop_when_center_already_on_circle():
    reg = EntityRegistry()
    c = reg.add_point(3.0, 4.0)
    r = reg.add_point(7.0, 4.0)
    circle = reg.add_circle(c, r)
    group = EntityGroup(reg, [circle])
    group.radial_project((0.0, 0.0), 5.0)
    assert reg.get_point(c).x == pytest.approx(3.0)
    assert reg.get_point(c).y == pytest.approx(4.0)
    assert reg.get_point(r).x == pytest.approx(7.0)


# ----------------------------------------------------------------------
# Snapshot / restore
# ----------------------------------------------------------------------


def test_snapshot_restore_round_trip():
    reg = EntityRegistry()
    p1 = reg.add_point(1.0, 1.0)
    p2 = reg.add_point(2.0, 5.0)
    line = reg.add_line(p1, p2)
    group = EntityGroup(reg, [line])
    snapshot = group.snapshot_positions()
    group.translate(10.0, -3.0)
    EntityGroup.restore_positions(snapshot)
    assert reg.get_point(p1).x == pytest.approx(1.0)
    assert reg.get_point(p1).y == pytest.approx(1.0)
    assert reg.get_point(p2).x == pytest.approx(2.0)
    assert reg.get_point(p2).y == pytest.approx(5.0)


# ----------------------------------------------------------------------
# Copy rewriting
# ----------------------------------------------------------------------


def test_rewrite_copy_from_pairs_positionally():
    reg = EntityRegistry()
    t1 = reg.add_point(0.0, 0.0)
    t2 = reg.add_point(10.0, 0.0)
    template = reg.add_line(t1, t2)
    c1 = reg.add_point(100.0, 100.0)
    c2 = reg.add_point(110.0, 100.0)
    copy = reg.add_line(c1, c2)
    placement = InstancePlacement(
        kind=PlacementKind.TRANSLATION, delta=(5.0, 7.0)
    )
    updates = EntityGroup(reg, [template]).rewrite_copy_from(
        EntityGroup(reg, [copy]), placement
    )
    assert reg.get_point(c1).x == pytest.approx(5.0)
    assert reg.get_point(c1).y == pytest.approx(7.0)
    assert reg.get_point(c2).x == pytest.approx(15.0)
    assert reg.get_point(c2).y == pytest.approx(7.0)
    assert [(pt.id, x, y) for pt, x, y in updates] == [
        (c1, 5.0, 7.0),
        (c2, 15.0, 7.0),
    ]


def test_rewrite_copy_from_takes_bezier_offsets_from_template():
    reg = EntityRegistry()
    ts = reg.add_point(0.0, 0.0)
    te = reg.add_point(10.0, 0.0)
    template = reg.add_bezier(ts, te, cp1=(1.0, 2.0), cp2=(3.0, 4.0))
    cs = reg.add_point(50.0, 0.0)
    ce = reg.add_point(60.0, 0.0)
    copy = reg.add_bezier(cs, ce, cp1=(9.0, 9.0), cp2=(8.0, 8.0))
    placement = InstancePlacement(
        kind=PlacementKind.ROTATION, angle=math.pi, center=(0.0, 0.0)
    )
    EntityGroup(reg, [template]).rewrite_copy_from(
        EntityGroup(reg, [copy]), placement
    )
    copy_entity = get_entity(reg, copy)
    assert copy_entity.cp1 == pytest.approx((-1.0, -2.0))
    assert copy_entity.cp2 == pytest.approx((-3.0, -4.0))


def test_rewrite_copy_from_keeps_copy_identity():
    reg = EntityRegistry()
    t1 = reg.add_point(0.0, 0.0)
    t2 = reg.add_point(10.0, 0.0)
    template = reg.add_line(t1, t2)
    c1 = reg.add_point(100.0, 100.0)
    c2 = reg.add_point(110.0, 100.0)
    copy = reg.add_line(c1, c2)
    placement = InstancePlacement(
        kind=PlacementKind.TRANSLATION, delta=(1.0, 1.0)
    )
    EntityGroup(reg, [template]).rewrite_copy_from(
        EntityGroup(reg, [copy]), placement
    )
    assert get_entity(reg, copy).id == copy
    assert reg.get_point(c1).id == c1
    assert reg.get_point(c2).id == c2


# ----------------------------------------------------------------------
# Membership semantics
# ----------------------------------------------------------------------


def test_helper_ids_collects_attached_construction_geometry():
    reg = EntityRegistry()
    c = reg.add_point(0.0, 0.0)
    rx = reg.add_point(5.0, 0.0)
    ry = reg.add_point(0.0, 5.0)
    ellipse = reg.add_ellipse(c, rx, ry)
    h1 = reg.add_line(c, rx, construction=True)
    h2 = reg.add_line(c, ry, construction=True)
    get_entity(reg, ellipse).helper_line_ids = [h1, h2]
    outside = reg.add_point(50.0, 50.0)
    reg.add_line(c, outside, construction=True)
    group = EntityGroup(reg, [ellipse])
    assert group.helper_ids() == [h1, h2]


def test_remap_point_refs_rewrites_group_entities():
    reg = EntityRegistry()
    p1 = reg.add_point(0.0, 0.0)
    p2 = reg.add_point(10.0, 0.0)
    line = reg.add_line(p1, p2)
    group = EntityGroup(reg, [line])
    group.remap_point_refs({p1: 77, p2: 88})
    entity = get_entity(reg, line)
    assert entity.p1_idx == 77
    assert entity.p2_idx == 88


def test_remap_point_refs_ignores_boolean_flags():
    reg = EntityRegistry()
    p1 = reg.add_point(0.0, 0.0)
    p2 = reg.add_point(10.0, 0.0)
    line = reg.add_line(p1, p2)
    entity = get_entity(reg, line)
    entity.invisible = True
    remap_point_refs(entity, {1: 99})
    assert entity.invisible is True
    assert entity.p2_idx == 99


def test_remap_point_refs_works_on_constraints():
    constr = RadiusConstraint(12, value=5.0)
    remap_point_refs(constr, {12: 34})
    assert constr.entity_id == 34
    assert constr.value == 5.0


# ----------------------------------------------------------------------
# Polylines
# ----------------------------------------------------------------------


def test_polylines_samples_entities():
    reg = EntityRegistry()
    p1 = reg.add_point(0.0, 0.0)
    p2 = reg.add_point(10.0, 0.0)
    p3 = reg.add_point(0.0, 10.0)
    line = reg.add_line(p1, p2)
    circle = reg.add_circle(p1, reg.add_point(5.0, 0.0))
    arc = reg.add_arc(p2, p3, p1)
    ellipse = reg.add_ellipse(
        p1, reg.add_point(4.0, 0.0), reg.add_point(0.0, 2.0)
    )
    bez = reg.add_bezier(p1, p2, cp1=(2.0, 5.0), cp2=(8.0, -5.0))
    group = EntityGroup(reg, [line, circle, arc, ellipse, bez])

    polylines = group.polylines()
    assert len(polylines) == 5
    assert polylines[0] == [(0.0, 0.0), (10.0, 0.0)]
    for polyline in polylines[1:]:
        assert len(polyline) > 8
    assert polylines[2][0] == (10.0, 0.0)
    assert polylines[2][-1] == pytest.approx((0.0, 10.0))
    assert polylines[4][0] == (0.0, 0.0)
    assert polylines[4][-1] == (10.0, 0.0)
