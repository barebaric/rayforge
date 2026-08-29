"""
End-to-end scenario tests simulating real UI flows for arrays:
create, delete individual members, re-open via double-click lookup,
edit/regenerate, undo/redo, and solver stability.
"""

import math

import pytest
from sketcher.core.arrays import (
    CircularArrayStrategy,
    find_array_for_entity,
)
from sketcher.core.commands import (
    CreateArrayCommand,
    EditArrayCommand,
    RemoveItemsCommand,
)
from sketcher.core.constraints import (
    DragConstraint,
    RadiusConstraint,
)
from sketcher.core.entities import Circle
from sketcher.core.sketch import Sketch


def make_strategy(count=6, radius=40.0, center=(0.0, 0.0)):
    return CircularArrayStrategy(
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

    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()
    sketch.solve()
    return sketch, cmd.array


def members_consistent(sketch, array):
    """
    The template is the single source of truth: every copy is a rigid
    rotation of it about the guide circle's center (same distance
    profile, distinct slot angles). The guide circle itself is
    independent construction geometry whose radius is defined by its
    radius constraint.
    """
    registry = sketch.registry
    circle = registry.get_entity(array.guide_circle_id)
    assert isinstance(circle, Circle)
    center = registry.get_point(circle.center_idx)

    members = array.living_members(registry)
    assert members, "no living members"
    template_eids = members[0][1]

    def shape(eids):
        pts = []
        for eid in eids:
            entity = registry.get_entity(eid)
            assert entity is not None
            for pid in entity.get_point_ids():
                pt = registry.get_point(pid)
                pts.append((pt.x, pt.y))
        return pts

    def pairwise(points):
        return sorted(
            math.dist(p, q)
            for i, p in enumerate(points)
            for q in points[i + 1 :]
        )

    template_shape = shape(template_eids)
    first_angles = []
    for _slot, eids in members[1:]:
        copy_shape = shape(eids)
        # Rigid rotation of the template.
        assert pairwise(copy_shape) == pytest.approx(pairwise(template_shape))
        angle = math.degrees(
            math.atan2(
                copy_shape[0][1] - center.y, copy_shape[0][0] - center.x
            )
        )
        first_angles.append(round(angle, 3))
    # Copies sit at distinct angles around the center.
    assert len(set(first_angles)) == len(first_angles)


def test_created_array_is_fully_registered(app_flow):
    sketch, array = app_flow
    assert len(sketch.arrays) == 1
    assert find_array_for_entity(sketch.arrays, array.guide_circle_id) is array
    assert len(array.living_entity_ids(sketch.registry)) == 6
    assert [slot for slot, _e in array.members] == list(range(6))
    members_consistent(sketch, array)


def test_dragging_template_redistributes_copies(app_flow):
    """
    After an incremental template drag (solver per frame, like the
    real UI) the copies follow the template and the array definition
    stays intact. The guide circle's radius is untouched by template
    edits.
    """
    sketch, array = app_flow
    template_id = array.living_entity_ids(sketch.registry)[0]
    pids = CreateArrayCommand.collect_template_point_ids(
        sketch.registry, [template_id]
    )
    circle = sketch.registry.get_entity(array.guide_circle_id)
    center = sketch.registry.get_point(circle.center_idx)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    radius_before = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)

    for _step in range(10):
        for pid in pids:
            sketch.registry.get_point(pid).x += 2.0
        sketch.solve()

    members_consistent(sketch, array)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    radius_after = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
    assert radius_after == pytest.approx(radius_before)
    assert len(array.living_entity_ids(sketch.registry)) == 6
    assert find_array_for_entity(sketch.arrays, array.guide_circle_id) is array


def test_radius_dimension_redistributes_members(app_flow):
    """
    The dialog's radius is a hard constraint on the construction
    circle and the single source of truth for the orbit: editing the
    dimension redistributes the members onto the new radius. Each
    member keeps its shape and its angular position (a pure radial
    translation); the radius drives the member placement, never the
    other way around.
    """
    sketch, array = app_flow
    circle = sketch.registry.get_entity(array.guide_circle_id)
    rc = next(
        c
        for c in sketch.constraints
        if isinstance(c, RadiusConstraint) and c.entity_id == circle.id
    )
    registry = sketch.registry

    def member_shapes():
        return {
            slot: [
                (
                    registry.get_point(pid).x,
                    registry.get_point(pid).y,
                )
                for eid in eids
                for pid in registry.get_entity(eid).get_point_ids()
            ]
            for slot, eids in array.living_members(registry)
        }

    def pairwise_distances(points):
        return sorted(
            math.dist(p, q)
            for i, p in enumerate(points)
            for q in points[i + 1 :]
        )

    before = member_shapes()

    rc.value = 60.0
    sketch.solve()

    # The circle now has the new radius.
    center = registry.get_point(circle.center_idx)
    radius_pt = registry.get_point(circle.radius_pt_idx)
    assert math.hypot(
        radius_pt.x - center.x, radius_pt.y - center.y
    ) == pytest.approx(60.0, abs=1e-3)
    # Every member kept its shape and sits centered on the new radius.
    for slot, shape in member_shapes().items():
        assert pairwise_distances(shape) == pytest.approx(
            pairwise_distances(before[slot])
        )
        xs = [x for x, _y in shape]
        ys = [y for _x, y in shape]
        cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
        assert math.hypot(cx - center.x, cy - center.y) == pytest.approx(
            60.0, abs=1e-3
        )
    members_consistent(sketch, array)


def test_deleting_one_member_keeps_the_rest(app_flow):
    sketch, array = app_flow
    victims = array.living_entity_ids(sketch.registry)[2:3]
    before = set(array.living_entity_ids(sketch.registry))

    ui_delete(sketch, victims)

    living = set(array.living_entity_ids(sketch.registry))
    assert living == before - set(victims)
    # Definition survives; double-click lookup still works.
    assert find_array_for_entity(sketch.arrays, array.guide_circle_id) is array
    members_consistent(sketch, array)


def test_edit_after_deletion_recreates_members(app_flow):
    sketch, array = app_flow
    victims = array.living_entity_ids(sketch.registry)[1:4]
    ui_delete(sketch, victims)
    assert len(array.living_entity_ids(sketch.registry)) == 3

    edit_cmd = EditArrayCommand(sketch, array, make_strategy())
    edit_cmd.execute()
    sketch.solve()

    assert len(array.living_entity_ids(sketch.registry)) == 6
    members_consistent(sketch, array)

    # Exactly one master, still found by double-click.
    circles = [e for e in sketch.registry.entities if e.type == "circle"]
    assert len(circles) == 1
    assert find_array_for_entity(sketch.arrays, array.guide_circle_id) is array


def test_repeated_edit_cycles_are_stable(app_flow):
    sketch, array = app_flow
    for count in (8, 5, 6):
        EditArrayCommand(sketch, array, make_strategy(count=count)).execute()
        sketch.solve()
        assert len(array.living_entity_ids(sketch.registry)) == count

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
    sketch, array = app_flow
    # Delete members in slots 1 and 2 (non-template).
    victims = [eid for _slot, eids in array.members[1:3] for eid in eids]
    ui_delete(sketch, victims)

    EditArrayCommand(sketch, array, make_strategy()).execute()
    sketch.solve()

    # All six members must sit at distinct slot angles on the circle.
    step = 60.0
    circle = sketch.registry.get_entity(array.guide_circle_id)
    center = sketch.registry.get_point(circle.center_idx)

    angles = []
    for eid in array.living_entity_ids(sketch.registry):
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


def test_radial_drag_does_not_distort_the_array(app_flow):
    """
    Regression: dragging the template radially used to fight the fixed
    radius dimension, violating constraints mid-drag and collapsing
    the geometry on release. The template is now outside the fight:
    it is free geometry, the copies are re-derived from it, and the
    guide circle radius follows the template center.
    """
    sketch, array = app_flow
    template_id = array.living_entity_ids(sketch.registry)[0]
    pids = CreateArrayCommand.collect_template_point_ids(
        sketch.registry, [template_id]
    )

    # Simulate a substantial radial drag: strong drag targets (like
    # _handle_entity_drag) solved per frame, no exclusions.
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
        )

    # Mid-drag: every member stays congruent (no distortion) and sits
    # on the guide circle at distinct slot angles.
    members_consistent(sketch, array)

    # Release: a final solve keeps everything consistent instead of
    # snapping back.
    sketch.solve()

    members_consistent(sketch, array)
    assert len(array.living_entity_ids(sketch.registry)) == 6
    circle = sketch.registry.get_entity(array.guide_circle_id)
    center = sketch.registry.get_point(circle.center_idx)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    new_radius = math.hypot(radius_pt.x - center.x, radius_pt.y - center.y)
    assert new_radius == pytest.approx(40.0)  # circle unaffected


def test_edit_undo_restores_geometry_and_membership(app_flow):
    sketch, array = app_flow
    ui_delete(
        sketch,
        [
            eid
            for _slot, eids in array.living_members(sketch.registry)[1:4]
            for eid in eids
        ],
    )
    members_before = list(array.members)

    edit_cmd = EditArrayCommand(sketch, array, make_strategy())
    edit_cmd.execute()
    assert len(array.living_entity_ids(sketch.registry)) == 6

    edit_cmd.undo()
    assert array.members == members_before
    assert len(array.living_members(sketch.registry)) == 3
