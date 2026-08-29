import math

import pytest
from sketcher.core.arrays import CircularArrayStrategy
from sketcher.core.commands import CreateArrayCommand, EditArrayCommand
from sketcher.core.constraints import RadiusConstraint, RotationalConstraint
from sketcher.core.params import ParameterContext
from sketcher.core.sketch import Sketch


def make_strategy(count=6, radius=40.0, center=(0.0, 0.0), rotate=True):
    return CircularArrayStrategy(
        count=count,
        total_angle_deg=360.0,
        center=center,
        radius=radius,
        rotate_copies=rotate,
    )


@pytest.fixture
def sketch_with_array():
    sketch = Sketch()
    p0 = sketch.registry.add_point(30, 0)
    p1 = sketch.registry.add_point(50, 0)
    line = sketch.registry.add_line(p0, p1)
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()
    return sketch, cmd.array


def count_circles(sketch):
    return [e for e in sketch.registry.entities if e.type == "circle"]


def member_count(sketch, array):
    return len(array.living_members(sketch.registry))


def test_edit_fills_missing_slots_only(sketch_with_array):
    """With unchanged parameters, surviving members stay untouched."""
    sketch, array = sketch_with_array
    template_eids = array.living_members(sketch.registry)[0][1]

    # Delete two non-template members.
    victims = [eid for slot, eids in array.members[1:3] for eid in eids]
    sketch.registry.remove_entities_by_id(victims)
    sketch.prune_arrays()
    assert member_count(sketch, array) == 4
    survivors_before = [
        (slot, eids)
        for slot, eids in array.living_members(sketch.registry)[1:]
    ]

    edit_cmd = EditArrayCommand(sketch, array, make_strategy())
    edit_cmd.execute()

    living = array.living_members(sketch.registry)
    assert len(living) == 6
    # The template and surviving copies kept their entity IDs (they
    # were not regenerated).
    assert living[0][1] == template_eids
    surviving_flat = {eid for _s, eids in survivors_before for eid in eids}
    living_flat = {eid for _s, eids in living[1:] for eid in eids}
    assert surviving_flat <= living_flat

    # No duplicate masters were created.
    assert len(count_circles(sketch)) == 1
    radius_constraints = [
        c for c in sketch.constraints if isinstance(c, RadiusConstraint)
    ]
    assert len(radius_constraints) == 1


def test_edit_full_regen_when_template_deleted(sketch_with_array):
    """Without the template member, the array is fully re-distributed."""
    sketch, array = sketch_with_array
    first_two = [eid for slot, eids in array.members[:2] for eid in eids]
    sketch.registry.remove_entities_by_id(first_two)
    sketch.prune_arrays()
    assert member_count(sketch, array) == 4

    EditArrayCommand(sketch, array, make_strategy()).execute()

    living = array.living_members(sketch.registry)
    assert len(living) == 6
    assert [slot for slot, _e in living] == list(range(6))
    assert len(count_circles(sketch)) == 1


def test_edit_repeated_edits_do_not_duplicate_masters(sketch_with_array):
    sketch, array = sketch_with_array
    for i in range(3):
        edit_cmd = EditArrayCommand(sketch, array, make_strategy(count=6 + i))
        edit_cmd.execute()

    assert len(count_circles(sketch)) == 1
    radius_constraints = [
        c for c in sketch.constraints if isinstance(c, RadiusConstraint)
    ]
    assert len(radius_constraints) == 1
    assert member_count(sketch, array) == 8


def test_edit_updates_slots_consistently(sketch_with_array):
    sketch, array = sketch_with_array
    edit_cmd = EditArrayCommand(sketch, array, make_strategy(count=5))
    edit_cmd.execute()

    assert sorted(
        slot for slot, _e in array.living_members(sketch.registry)
    ) == list(range(5))
    assert len(array.members) == 5


def test_multi_entity_template_stays_grouped():
    """
    Regression: a multi-entity template must regenerate as whole shapes.
    Deleting fragments of one member must not turn its leftovers into
    a template that produces broken partial copies.
    """
    sketch = Sketch()
    # Square template: 4 entities.
    corners = [
        sketch.registry.add_point(x, y)
        for x, y in ((30, 0), (50, 0), (50, 20), (30, 20))
    ]
    template_ids = [
        sketch.registry.add_line(corners[i], corners[(i + 1) % 4])
        for i in range(4)
    ]
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4),
        list(template_ids),
    )
    cmd.execute()
    array = cmd.array
    assert array is not None

    # Every member is a 4-entity group.
    assert all(len(eids) == 4 for _s, eids in array.members)

    # Delete one full copy and ONE LINE of another copy (fragmentation).
    _slot1, group1 = array.members[1]
    full_victim = list(group1)
    _slot2, group2 = array.members[2]
    fragment_victim = group2[0]
    sketch.registry.remove_entities_by_id(full_victim + [fragment_victim])
    sketch.prune_arrays()
    assert len(array.members) == 3
    # The damaged group keeps its three surviving lines.
    damaged = next(eids for _s, eids in array.members if len(eids) == 3)
    assert set(damaged) == set(group2) - {fragment_victim}

    EditArrayCommand(sketch, array, make_strategy(count=4)).execute()

    living = array.living_members(sketch.registry)
    assert len(living) == 4
    # No fragmentation: every regenerated member is a complete copy of
    # whatever the template member now is (the damaged 3-line shape),
    # never a mix of single leftover lines.
    sizes = [len(eids) for _slot, eids in living]
    assert len(set(sizes)) == 1

    # No collapsed duplicates among MEMBER geometry: every member
    # occupies a distinct location.
    registry = sketch.registry
    member_points = set()
    for eid in array.living_entity_ids(registry):
        entity = registry.get_entity(eid)
        if entity is not None:
            member_points.update(entity.get_point_ids())
    positions = sorted(
        (
            round(registry.get_point(pid).x, 3),
            round(registry.get_point(pid).y, 3),
        )
        for pid in member_points
    )
    assert len(positions) == len(set(positions))


def test_edit_moves_master_geometry(sketch_with_array):
    sketch, array = sketch_with_array
    new_params = make_strategy(count=6, center=(10.0, 5.0), radius=30.0)
    edit_cmd = EditArrayCommand(sketch, array, new_params)
    edit_cmd.execute()

    circle = sketch.registry.get_entity(array.guide_circle_id)
    center = sketch.registry.get_point(circle.center_idx)
    radius_pt = sketch.registry.get_point(circle.radius_pt_idx)
    assert (center.x, center.y) == pytest.approx((10.0, 5.0))
    assert math.hypot(radius_pt.x - center.x, radius_pt.y - center.y) == (
        pytest.approx(30.0)
    )

    rc = next(
        c
        for c in sketch.constraints
        if isinstance(c, RadiusConstraint) and c.entity_id == circle.id
    )
    assert rc.value == pytest.approx(30.0)


def test_edit_undo_restores_previous_state(sketch_with_array):
    sketch, array = sketch_with_array
    members_before = list(array.members)
    points_before = sorted((p.x, p.y) for p in sketch.registry.points)

    edit_cmd = EditArrayCommand(
        sketch, array, make_strategy(count=8, center=(5.0, 5.0), radius=20.0)
    )
    edit_cmd.execute()
    assert member_count(sketch, array) == 8

    edit_cmd.undo()
    assert array.members == members_before
    assert member_count(sketch, array) == 6
    points_after = sorted((p.x, p.y) for p in sketch.registry.points)
    assert points_after == pytest.approx(points_before)


def test_edit_redo_reapplies_changes(sketch_with_array):
    sketch, array = sketch_with_array
    edit_cmd = EditArrayCommand(sketch, array, make_strategy(count=9))
    edit_cmd.execute()
    assert member_count(sketch, array) == 9

    edit_cmd.undo()
    assert member_count(sketch, array) == 6

    edit_cmd.execute()
    assert member_count(sketch, array) == 9
    # Still exactly one master after undo/redo cycles.
    assert len(count_circles(sketch)) == 1


def test_edit_keeps_master_alive_while_members_deleted(sketch_with_array):
    """Deleting all members must keep the definition (master survives)."""
    sketch, array = sketch_with_array
    all_ids = [eid for _slot, eids in array.members for eid in eids]
    sketch.registry.remove_entities_by_id(all_ids)
    sketch.prune_arrays()

    assert len(sketch.arrays) == 1
    assert array.living_members(sketch.registry) == []


def test_deleting_master_dissolves_array(sketch_with_array):
    sketch, array = sketch_with_array
    sketch.registry.remove_entities_by_id([array.guide_circle_id])
    sketch.prune_arrays()

    assert sketch.arrays == []


def test_edited_copies_are_static_baked(sketch_with_array):
    """Newly created copies are static: fixed points, no constraints
    referencing them."""
    sketch, array = sketch_with_array
    template_group = array.living_members(sketch.registry)[0][1]

    # Delete everything except the template, then regenerate.
    stale = [
        eid
        for _slot, eids in array.members
        for eid in eids
        if eids != template_group
    ]
    sketch.registry.remove_entities_by_id(stale)
    sketch.prune_arrays()

    EditArrayCommand(sketch, array, make_strategy()).execute()

    copy_pids = {
        pid
        for _slot, eids in array.living_members(sketch.registry)[1:]
        for eid in eids
        for pid in sketch.registry.get_entity(eid).get_point_ids()
    }
    for constr in sketch.constraints:
        assert not constr.get_referenced_point_ids() & copy_pids
    for pid in copy_pids:
        assert sketch.registry.get_point(pid).fixed
    # The template keeps its own constraints (unaffected by the
    # static copies).


def test_edit_leaves_zero_residual_before_any_solve(sketch_with_array):
    """
    Regression: editing used to teleport the master center while members
    stayed behind, leaving huge constraint residuals for the solver to
    "repair" (which collapsed the geometry). The command must leave the
    sketch fully consistent immediately after execute.
    """
    sketch, array = sketch_with_array
    edit_cmd = EditArrayCommand(
        sketch,
        array,
        make_strategy(count=9, center=(15.0, -10.0), radius=55.0),
    )
    edit_cmd.execute()

    ctx = ParameterContext()
    worst_rot = max(
        (
            max(abs(v) for v in c.error(sketch.registry, ctx))
            for c in sketch.constraints
            if isinstance(c, RotationalConstraint)
        ),
        default=0.0,
    )
    assert worst_rot < 1e-6

    pins = [
        c
        for c in sketch.constraints
        if type(c).__name__ == "PointOnLineConstraint"
    ]
    for pin in pins:
        err = pin.error(sketch.registry, ctx)
        err = err if isinstance(err, (list, tuple)) else [err]
        assert max(abs(v) for v in err) < 1e-6

    radius_constraints = [
        c for c in sketch.constraints if isinstance(c, RadiusConstraint)
    ]
    for rc in radius_constraints:
        assert abs(rc.error(sketch.registry, ctx)) < 1e-6


def test_empty_selection_creates_nothing():
    sketch = Sketch()
    cmd = CreateArrayCommand(sketch, make_strategy(), [])
    cmd.execute()
    assert cmd.created_entity_ids == []
    assert sketch.registry.entities == []
    assert sketch.arrays == []
