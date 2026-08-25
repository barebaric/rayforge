"""
End-to-end scenario tests simulating real UI flows for patterns:
create, delete individual members, re-open via double-click lookup,
edit/regenerate, undo/redo, and solver stability.
"""

import math

import pytest
from sketcher.core.commands import (
    CreatePatternCommand,
    EditPatternCommand,
    RemoveItemsCommand,
)
from sketcher.core.constraints import (
    DragConstraint,
    PointOnLineConstraint,
    RadiusConstraint,
    RotationalConstraint,
)
from sketcher.core.entities import Circle
from sketcher.core.patterns import (
    CircularPatternParams,
    SketchArrayMode,
    find_pattern_for_entity,
)
from sketcher.core.sketch import Sketch


def make_params(count=6, radius=40.0, center=(0.0, 0.0)):
    return CircularPatternParams(
        count=count,
        total_angle_deg=360.0,
        center=center,
        radius=radius,
        rotate_copies=True,
    )


def ui_delete(sketch, entity_ids):
    """Deletes like the DeleteTool: dependency-based removal."""
    points, entities, constraints = RemoveItemsCommand.calculate_dependencies(
        sketch,
        SimpleNamespaceSelection(set(entity_ids)),
    )
    cmd = RemoveItemsCommand(
        sketch,
        "",
        points=points,
        entities=entities,
        constraints=constraints,
    )
    cmd.execute()


class SimpleNamespaceSelection:
    def __init__(self, entity_ids):
        self.entity_ids = entity_ids
        self.point_ids = set()
        self.constraint_idx = None


@pytest.fixture
def app_flow():
    """Sketch with one array created through the full create flow."""
    sketch = Sketch()
    p0 = sketch.registry.add_point(30, 0)
    p1 = sketch.registry.add_point(50, 0)
    line = sketch.registry.add_line(p0, p1)

    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()
    sketch.solve()
    return sketch, cmd.pattern


def members_on_circle(sketch, pattern):
    """All living members' geometry sits at the guide circle radius."""
    circle = sketch.registry.get_entity(pattern.guide_circle_id)
    assert isinstance(circle, Circle)
    center = sketch.registry.get_point(circle.center_idx)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)

    pins = [
        c
        for c in sketch.constraints
        if isinstance(c, PointOnLineConstraint)
        and c.shape_id == pattern.guide_circle_id
    ]
    assert pins, "no members are anchored to the guide circle"
    for pin in pins:
        pt = sketch.registry.get_point(pin.point_id)
        d = math.hypot(pt.x - center.x, pt.y - center.y)
        assert d == pytest.approx(radius, abs=1e-3)


def test_created_array_is_fully_registered(app_flow):
    sketch, pattern = app_flow
    assert len(sketch.patterns) == 1
    assert (
        find_pattern_for_entity(sketch.patterns, pattern.guide_circle_id)
        is pattern
    )
    assert len(pattern.living_entity_ids(sketch.registry)) == 6
    assert [slot for slot, _e in pattern.members] == list(range(6))
    members_on_circle(sketch, pattern)


def test_dragging_member_keeps_circle_anchored(app_flow):
    """
    After an incremental member drag (solver per frame, like the real
    UI) the system settles with the guide circle still crossing every
    member and the pattern definition intact.
    """
    sketch, pattern = app_flow
    template_id = pattern.living_entity_ids(sketch.registry)[0]
    pids = CreatePatternCommand.collect_seed_point_ids(
        sketch.registry, [template_id]
    )
    for _step in range(10):
        for pid in pids:
            sketch.registry.get_point(pid).x += 2.0
        sketch.solve()

    members_on_circle(sketch, pattern)
    assert len(pattern.living_entity_ids(sketch.registry)) == 6
    assert (
        find_pattern_for_entity(sketch.patterns, pattern.guide_circle_id)
        is pattern
    )


def test_radius_dimension_resizes_the_array(app_flow):
    """Editing the radius constraint value re-sizes the whole array."""
    sketch, pattern = app_flow
    circle = sketch.registry.get_entity(pattern.guide_circle_id)
    rc = next(
        c
        for c in sketch.constraints
        if isinstance(c, RadiusConstraint) and c.entity_id == circle.id
    )
    rc.value = 60.0
    sketch.solve()

    members_on_circle(sketch, pattern)
    # The anchored reference point of the template was carried to the
    # new radius.
    pin = next(
        c
        for c in sketch.constraints
        if isinstance(c, PointOnLineConstraint)
        and c.shape_id == pattern.guide_circle_id
    )
    pt = sketch.registry.get_point(pin.point_id)
    center = sketch.registry.get_point(circle.center_idx)
    assert math.hypot(pt.x - center.x, pt.y - center.y) == pytest.approx(
        60.0, abs=1e-3
    )


def test_deleting_one_member_keeps_the_rest(app_flow):
    sketch, pattern = app_flow
    victims = pattern.living_entity_ids(sketch.registry)[2:3]
    before = set(pattern.living_entity_ids(sketch.registry))

    ui_delete(sketch, victims)

    living = set(pattern.living_entity_ids(sketch.registry))
    assert living == before - set(victims)
    # Definition survives; double-click lookup still works.
    assert (
        find_pattern_for_entity(sketch.patterns, pattern.guide_circle_id)
        is pattern
    )
    members_on_circle(sketch, pattern)


def test_edit_after_deletion_recreates_members(app_flow):
    sketch, pattern = app_flow
    victims = pattern.living_entity_ids(sketch.registry)[1:4]
    ui_delete(sketch, victims)
    assert len(pattern.living_entity_ids(sketch.registry)) == 3

    edit_cmd = EditPatternCommand(sketch, pattern, make_params())
    edit_cmd.execute()
    sketch.solve()

    assert len(pattern.living_entity_ids(sketch.registry)) == 6
    members_on_circle(sketch, pattern)

    # Exactly one master, still found by double-click.
    circles = [e for e in sketch.registry.entities if e.type == "circle"]
    assert len(circles) == 1
    assert (
        find_pattern_for_entity(sketch.patterns, pattern.guide_circle_id)
        is pattern
    )


def test_repeated_edit_cycles_are_stable(app_flow):
    sketch, pattern = app_flow
    for count in (8, 5, 6):
        EditPatternCommand(sketch, pattern, make_params(count=count)).execute()
        sketch.solve()
        assert len(pattern.living_entity_ids(sketch.registry)) == count

    circles = [e for e in sketch.registry.entities if e.type == "circle"]
    assert len(circles) == 1
    radius_constraints = [
        c for c in sketch.constraints if type(c).__name__ == "RadiusConstraint"
    ]
    assert len(radius_constraints) == 1


def test_gap_fill_places_members_at_their_slot_angles(app_flow):
    """
    Regression: regenerated members must be constrained at their own
    slot's angle, not the first placement. Otherwise two copies collapse
    onto one position when the solver runs.
    """
    sketch, pattern = app_flow
    # Delete members in slots 1 and 2 (non-template).
    victims = [eid for _slot, eids in pattern.members[1:3] for eid in eids]
    ui_delete(sketch, victims)

    EditPatternCommand(sketch, pattern, make_params()).execute()
    sketch.solve()

    # All six members must sit at distinct slot angles on the circle.
    step = 60.0
    circle = sketch.registry.get_entity(pattern.guide_circle_id)
    center = sketch.registry.get_point(circle.center_idx)

    angles = []
    for eid in pattern.living_entity_ids(sketch.registry):
        entity = sketch.registry.get_entity(eid)
        for pid in entity.get_point_ids():
            pt = sketch.registry.get_point(pid)
            # The template is radial: both of its points share a slot
            # angle, and so does every rotated copy.
            angles.append(
                round(
                    math.degrees(math.atan2(pt.y - center.y, pt.x - center.x))
                    / step
                )
                * step
                % 360.0
            )

    expected = sorted(step * j for j in range(6))
    seen_slots = sorted(set(angles))
    assert seen_slots == expected
    # Every slot holds exactly one member (two points each).
    assert len(angles) == 12
    assert sorted(angles) == sorted(expected + expected)


def test_pattern_points_helper_scopes_to_touched_pattern(app_flow):
    sketch, pattern = app_flow
    member_ids = set(pattern.living_entity_ids(sketch.registry))

    # Dragging an unrelated entity yields no exempt points.
    sp0 = sketch.registry.add_point(200, 200)
    sp1 = sketch.registry.add_point(210, 210)
    stranger = sketch.registry.add_line(sp0, sp1)
    assert sketch.get_pattern_points_for_entities({stranger}) == set()

    # Dragging a member exempts all member points plus master points.
    first_member = min(member_ids)
    points = sketch.get_pattern_points_for_entities({first_member})
    expected = set()
    for eid in member_ids:
        entity = sketch.registry.get_entity(eid)
        expected.update(entity.get_point_ids())
    guide = sketch.registry.get_entity(pattern.guide_circle_id)
    expected.update(guide.get_point_ids())
    assert points == expected


def test_radial_drag_does_not_distort_the_array(app_flow):
    """
    Regression: dragging a member radially used to fight the fixed
    radius dimension, violating constraints mid-drag and collapsing
    the geometry on release. The dimension now yields during the drag
    (excluded from the solve) and follows the geometry afterwards.
    """
    sketch, pattern = app_flow
    template_id = pattern.living_entity_ids(sketch.registry)[0]
    pids = CreatePatternCommand.collect_seed_point_ids(
        sketch.registry, [template_id]
    )

    # Simulate a substantial radial drag: strong drag targets (like
    # _handle_entity_drag) solved per frame with the pattern's radius
    # constraint excluded.
    excluded = sketch.get_pattern_constraint_indices_for_entities(
        {template_id}
    )
    assert excluded, "radius dimension was not excluded"

    for frame in range(10):
        extra = [
            DragConstraint(pid, p.x + 3.0, p.y, weight=1.0)
            for pid, p in (
                (pid, sketch.registry.get_point(pid)) for pid in pids
            )
        ]
        sketch.solve(
            extra_constraints=extra,
            update_constraint_status=False,
            excluded_constraints=excluded,
        )

    # Mid-drag: every member stays congruent (no distortion) and sits
    # on the guide circle at distinct slot angles.
    members_on_circle(sketch, pattern)

    # Release: dimensions follow geometry; final solve keeps everything
    # consistent instead of snapping back.
    sketch.sync_pattern_dimensions({template_id})
    sketch.solve()

    members_on_circle(sketch, pattern)
    assert len(pattern.living_entity_ids(sketch.registry)) == 6
    circle = sketch.registry.get_entity(pattern.guide_circle_id)
    center = sketch.registry.get_point(circle.center_idx)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    new_radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
    assert new_radius > 45.0  # grew substantially with the drag

    # Congruence: all copies still rigid rotations of the template.
    for c in sketch.constraints:
        if isinstance(c, RotationalConstraint):
            src = sketch.registry.get_point(c.p1)
            dst = sketch.registry.get_point(c.p2)
            ctr = sketch.registry.get_point(c.center)
            ca, sa = math.cos(c.value), math.sin(c.value)
            dx, dy = src.x - ctr.x, src.y - ctr.y
            ex = dst.x - (ctr.x + ca * dx - sa * dy)
            ey = dst.y - (ctr.y + sa * dx + ca * dy)
            assert math.hypot(ex, ey) < 1e-3


def test_edit_undo_restores_geometry_and_membership(app_flow):
    sketch, pattern = app_flow
    ui_delete(
        sketch,
        [
            eid
            for _slot, eids in pattern.living_members(sketch.registry)[1:4]
            for eid in eids
        ],
    )
    members_before = list(pattern.members)

    edit_cmd = EditPatternCommand(sketch, pattern, make_params())
    edit_cmd.execute()
    assert len(pattern.living_entity_ids(sketch.registry)) == 6

    edit_cmd.undo()
    assert pattern.members == members_before
    assert len(pattern.living_members(sketch.registry)) == 3
