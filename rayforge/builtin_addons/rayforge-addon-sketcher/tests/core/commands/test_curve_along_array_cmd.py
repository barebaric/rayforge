import math

import pytest
from sketcher.core.arrays import (
    CurveAlongArray,
    CurveAlongArrayStrategy,
    find_array_for_entity,
)
from sketcher.core.commands import CreateArrayCommand, EditArrayCommand
from sketcher.core.constraints import (
    CoincidentConstraint,
    HorizontalConstraint,
    TangentConstraint,
    VerticalConstraint,
)
from sketcher.core.entities import Bezier, Ellipse, Line
from sketcher.core.sketch import Sketch


def make_strategy(count=4, path_id=1, align=True, offset=0.0):
    return CurveAlongArrayStrategy(
        count=count,
        path_entity_id=path_id,
        align_to_tangent=align,
        offset_to_start=offset,
        rotate_copies=True,
    )


@pytest.fixture
def sketch_with_template_and_path():
    """A horizontal line seed at the origin and a path line to the right."""
    sketch = Sketch()
    # Seed: a small vertical line at the origin (the member to repeat).
    sp0 = sketch.registry.add_point(0.0, -2.0)
    sp1 = sketch.registry.add_point(0.0, 2.0)
    seed = sketch.registry.add_line(sp0, sp1)
    # Guide path: a horizontal line from (0,0) to (30,0).
    pp0 = sketch.registry.add_point(0.0, 0.0)
    pp1 = sketch.registry.add_point(30.0, 0.0)
    path = sketch.registry.add_line(pp0, pp1)
    return sketch, seed, path


# ----------------------------------------------------------------------
# CreateArrayCommand
# ----------------------------------------------------------------------


def test_creates_static_copies_along_path(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()

    copies = [e for e in sketch.registry.entities if e.array_copy]
    # count=4 members: the template (the seed, left in place) plus
    # 3 copies on the path.
    assert len(copies) == 3
    array = cmd.array
    assert array is not None
    assert array.mode == "curve_along"
    # The guide entity is the pre-existing path, not a new circle.
    assert array.guide_circle_id == path
    circles = [e for e in sketch.registry.entities if e.type == "circle"]
    assert circles == []
    # The template member is the original seed entity, unmoved.
    assert array.living_members(sketch.registry)[0] == (0, [seed])
    seed_entity = sketch.registry.get_entity(seed)
    seed_p0 = sketch.registry.get_point(seed_entity.p1_idx)
    assert (seed_p0.x, seed_p0.y) == (0.0, -2.0)


def test_array_findable_by_path_entity(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=path),
        [seed],
    )
    cmd.execute()
    assert find_array_for_entity(sketch.arrays, path) is cmd.array


def test_copy_positions_span_the_path(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()

    # Each copy's bbox center should sit at 10, 20, 30 along x.
    array = cmd.array
    assert array is not None
    member_ids = array.living_entity_ids(sketch.registry)
    centers_x = []
    for eid in member_ids[1:]:  # skip template
        entity = sketch.registry.get_entity(eid)
        assert entity is not None
        pids = entity.get_point_ids()
        xs = [sketch.registry.get_point(p).x for p in pids]
        centers_x.append(sum(xs) / len(xs))
    assert sorted(centers_x) == pytest.approx([10.0, 20.0, 30.0])


def test_no_rotational_constraints_are_created(sketch_with_template_and_path):
    """Curve-along uses CurvePathConstraints, not RotationalConstraints."""
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    rot = [
        c
        for c in sketch.constraints
        if type(c).__name__ == "RotationalConstraint"
    ]
    assert rot == []


def test_no_linkage_constraints_are_created(sketch_with_template_and_path):
    """Curve-along copies are static: no solver linkage is created."""
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    assert sketch.constraints == []


def test_template_points_remain_editable(sketch_with_template_and_path):
    """Template points must NOT be fixed so the user can still edit
    the template shape."""
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None
    template_eid = array.living_entity_ids(sketch.registry)[0]
    entity = sketch.registry.get_entity(template_eid)
    assert entity is not None
    for pid in entity.get_point_ids():
        pt = sketch.registry.get_point(pid)
        assert pt.fixed is False


def test_dragging_path_redistributes_copies(sketch_with_template_and_path):
    """Moving the path endpoint and solving re-applies the array:
    copies move to new positions along the new path."""
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None

    copies_before = []
    for eid in array.living_entity_ids(sketch.registry)[1:]:
        entity = sketch.registry.get_entity(eid)
        assert entity is not None
        pids = entity.get_point_ids()
        copies_before.append(
            tuple(
                sorted(round(sketch.registry.get_point(p).x, 1) for p in pids)
            )
        )

    path_entity = sketch.registry.get_entity(path)
    end_pt = sketch.registry.get_point(path_entity.p2_idx)
    end_pt.x = 60.0
    sketch.solve()

    copies_after = []
    for eid in array.living_entity_ids(sketch.registry)[1:]:
        entity = sketch.registry.get_entity(eid)
        assert entity is not None
        pids = entity.get_point_ids()
        copies_after.append(
            tuple(
                sorted(round(sketch.registry.get_point(p).x, 1) for p in pids)
            )
        )
    assert copies_before != copies_after


def test_dragging_bezier_endpoint_redistributes_copies():
    """Dragging a Bezier path endpoint re-applies copies to new
    positions along the longer path."""
    sketch = Sketch()
    sp0 = sketch.registry.add_point(5.0, 0.0)
    sp1 = sketch.registry.add_point(10.0, 0.0)
    seed = sketch.registry.add_line(sp0, sp1)
    bp0 = sketch.registry.add_point(0.0, 0.0)
    bp1 = sketch.registry.add_point(40.0, 0.0)
    path = sketch.registry.add_bezier(bp0, bp1, cp1=(10, 30), cp2=(30, 30))
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None

    before = []
    for eid in array.living_entity_ids(sketch.registry)[1:]:
        entity = sketch.registry.get_entity(eid)
        assert entity is not None
        pids = entity.get_point_ids()
        before.append(
            tuple(
                sorted(round(sketch.registry.get_point(p).x, 1) for p in pids)
            )
        )

    path_entity = sketch.registry.get_entity(path)
    assert isinstance(path_entity, Bezier)
    end_pt = sketch.registry.get_point(path_entity.end_idx)
    end_pt.x = 80.0
    sketch.solve()

    after = []
    for eid in array.living_entity_ids(sketch.registry)[1:]:
        entity = sketch.registry.get_entity(eid)
        assert entity is not None
        pids = entity.get_point_ids()
        after.append(
            tuple(
                sorted(round(sketch.registry.get_point(p).x, 1) for p in pids)
            )
        )
    assert before != after
    assert sketch.registry.get_point(path_entity.end_idx).x == pytest.approx(
        80.0
    )


def test_undo_removes_copies_and_array(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    n_entities = len(sketch.registry.entities)
    n_points = len(sketch.registry.points)

    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    assert len(sketch.arrays) == 1

    cmd.undo()
    assert len(sketch.registry.entities) == n_entities
    assert len(sketch.registry.points) == n_points
    assert sketch.arrays == []


def test_redo_restores_array(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    cmd.undo()
    cmd.execute()
    # count=4 members: template (in place) + 3 copies.
    assert len([e for e in sketch.registry.entities if e.array_copy]) == 3
    assert len(sketch.arrays) == 1


def test_serialization_round_trip_preserves_curve_array(
    sketch_with_template_and_path,
):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=path),
        [seed],
    )
    cmd.execute()

    restored = Sketch.from_dict(sketch.to_dict())
    assert len(restored.arrays) == 1
    p = restored.arrays[0]
    assert isinstance(p, CurveAlongArray)
    assert p.mode == "curve_along"
    assert p.path_entity_id == path
    assert p.count == 3


def test_spacing_drives_count(sketch_with_template_and_path):
    """When spacing > 0, the count is derived from the usable path
    length, not the explicit count field."""
    sketch, seed, path = sketch_with_template_and_path
    # Path is 30 units; spacing 10 -> count 4.
    params = CurveAlongArrayStrategy(
        path_entity_id=path,
        spacing=10.0,
        count=99,
    )
    cmd = CreateArrayCommand(sketch, params, [seed])
    cmd.execute()
    array = cmd.array
    assert array is not None
    # 4 total members (template + 3 copies).
    assert len(array.living_members(sketch.registry)) == 4


def test_serialization_preserves_spacing(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    params = CurveAlongArrayStrategy(
        path_entity_id=path, spacing=7.5, offset_to_start=3.0
    )
    cmd = CreateArrayCommand(sketch, params, [seed])
    cmd.execute()
    restored = Sketch.from_dict(sketch.to_dict())
    p = restored.arrays[0]
    assert isinstance(p, CurveAlongArray)
    assert p.spacing == pytest.approx(7.5)
    assert p.offset_to_start == pytest.approx(3.0)


def test_empty_selection_creates_nothing(sketch_with_template_and_path):
    sketch, _seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(path_id=path),
        [],
    )
    cmd.execute()
    assert cmd.created_entity_ids == []
    assert sketch.arrays == []


def test_solver_keeps_geometry_stable_after_apply(
    sketch_with_template_and_path,
):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    before = [(p.id, p.x, p.y) for p in sketch.registry.points]
    sketch.solve()
    for pid, x, y in before:
        pt = sketch.registry.get_point(pid)
        assert pt.x == pytest.approx(x, abs=1e-3)
        assert pt.y == pytest.approx(y, abs=1e-3)


# ----------------------------------------------------------------------
# EditArrayCommand
# ----------------------------------------------------------------------


def test_edit_fills_missing_slots(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=6, path_id=path),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None

    # Delete two non-template members.
    victims = [eid for _slot, eids in array.members[1:3] for eid in eids]
    sketch.registry.remove_entities_by_id(victims)
    sketch.prune_arrays()
    assert len(array.living_members(sketch.registry)) == 4

    EditArrayCommand(
        sketch, array, make_strategy(count=6, path_id=path)
    ).execute()

    assert len(array.living_members(sketch.registry)) == 6


def test_edit_full_regen_when_count_changes(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None

    EditArrayCommand(
        sketch, array, make_strategy(count=8, path_id=path)
    ).execute()
    assert len(array.living_members(sketch.registry)) == 8


def test_edit_undo_restores_state(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None
    members_before = list(array.members)

    edit = EditArrayCommand(
        sketch, array, make_strategy(count=7, path_id=path)
    )
    edit.execute()
    assert len(array.living_members(sketch.registry)) == 7

    edit.undo()
    assert array.members == members_before
    assert len(array.living_members(sketch.registry)) == 4


def test_edit_redo_reapplies(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None

    edit = EditArrayCommand(
        sketch, array, make_strategy(count=9, path_id=path)
    )
    edit.execute()
    edit.undo()
    edit.execute()
    assert len(array.living_members(sketch.registry)) == 9


def test_deleting_path_dissolves_array(sketch_with_template_and_path):
    sketch, seed, path = sketch_with_template_and_path
    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=4, path_id=path),
        [seed],
    )
    cmd.execute()

    sketch.registry.remove_entities_by_id([path])
    sketch.prune_arrays()
    assert sketch.arrays == []


def test_edit_with_arc_path_aligns_copies():
    """Copies along an arc path rotate to follow the tangent."""
    sketch = Sketch()
    # Horizontal seed at the path start, pointing along the path's
    # initial tangent (+x) so we can observe rotation.
    sp0 = sketch.registry.add_point(10.0, 0.0)
    sp1 = sketch.registry.add_point(11.0, 0.0)
    seed = sketch.registry.add_line(sp0, sp1)
    c = sketch.registry.add_point(0.0, 0.0)
    s = sketch.registry.add_point(10.0, 0.0)
    e = sketch.registry.add_point(0.0, 10.0)
    arc = sketch.registry.add_arc(s, e, c, cw=False)

    cmd = CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=arc),
        [seed],
    )
    cmd.execute()
    array = cmd.array
    assert array is not None
    members = array.living_members(sketch.registry)
    assert len(members) == 3

    # The last copy sits near the arc end (0, 10), where the tangent
    # points in -x (angle ~pi). A horizontal seed rotated by ~180deg
    # is still horizontal, so dx stays larger than dy.
    last_eid = members[-1][1][0]
    last_entity = sketch.registry.get_entity(last_eid)
    assert last_entity is not None
    lp1, lp2 = last_entity.get_point_ids()
    p1 = sketch.registry.get_point(lp1)
    p2 = sketch.registry.get_point(lp2)
    dx = abs(p2.x - p1.x)
    dy = abs(p2.y - p1.y)
    assert dx > dy
    # Center near the arc end.
    cx = (p1.x + p2.x) / 2
    cy = (p1.y + p2.y) / 2
    assert math.hypot(cx - 0.0, cy - 10.0) < 1.5

    # The middle copy (slot 1) sits near (10*cos45, 10*sin45) and is
    # rotated ~45deg, so a horizontal 1-unit seed now spans roughly
    # equally in x and y.
    mid_eid = members[1][1][0]
    mid_entity = sketch.registry.get_entity(mid_eid)
    assert mid_entity is not None
    mp1, mp2 = mid_entity.get_point_ids()
    m1 = sketch.registry.get_point(mp1)
    m2 = sketch.registry.get_point(mp2)
    mdx = abs(m2.x - m1.x)
    mdy = abs(m2.y - m1.y)
    assert mdx == pytest.approx(mdy, abs=0.2)


# ----------------------------------------------------------------------
# Template extraction
# ----------------------------------------------------------------------


def test_template_extraction_severs_guide_links(sketch_with_template_and_path):
    """
    The template owns its geometry: a point shared with the guide is
    cloned, external constraints are erased, and applying the array
    never moves the guide.
    """
    from sketcher.core.constraints import DistanceConstraint

    sketch, seed, path = sketch_with_template_and_path
    registry = sketch.registry
    # Rebuild the seed so its first point IS the guide path's start
    # point (a shared pid), and tie its second point to a foreign
    # point.
    path_entity = registry.get_entity(path)
    shared_pid = path_entity.p1_idx
    registry.remove_entities_by_id([seed])
    seed_p2 = registry.add_point(10.0, 5.0)
    seed = registry.add_line(shared_pid, seed_p2)
    foreign = registry.add_point(30.0, 20.0)
    sketch.constraints.append(CoincidentConstraint(seed_p2, foreign))
    sketch.constraints.append(DistanceConstraint(seed_p2, foreign, 12.0))
    n_constraints = len(sketch.constraints)

    CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=path),
        [seed],
    ).execute()
    sketch.solve()

    # The guide is untouched.
    path_entity = registry.get_entity(path)
    start = registry.get_point(path_entity.p1_idx)
    end = registry.get_point(path_entity.p2_idx)
    assert (start.x, start.y) == (0.0, 0.0)
    assert (end.x, end.y) == (30.0, 0.0)
    # The foreign point is untouched and the template no longer
    # shares the guide's point.
    tpl_entity = registry.get_entity(seed)
    assert tpl_entity.p1_idx != shared_pid
    assert (registry.get_point(foreign).x, registry.get_point(foreign).y) == (
        30.0,
        20.0,
    )
    # External constraints were erased.
    assert len(sketch.constraints) < n_constraints
    # The solve reached a stable state: a second solve does not move
    # the template.
    tpl_p1 = registry.get_point(tpl_entity.p1_idx)
    pos = (tpl_p1.x, tpl_p1.y)
    sketch.solve()
    tpl_p1 = registry.get_point(tpl_entity.p1_idx)
    assert (tpl_p1.x, tpl_p1.y) == pos


def test_template_internal_constraints_survive_extraction():
    """The glue of a multi-entity template is kept and cloned onto
    every copy; only external constraints are erased."""
    sketch = Sketch()
    # Guide path.
    pp0 = sketch.registry.add_point(0.0, 0.0)
    pp1 = sketch.registry.add_point(30.0, 0.0)
    path = sketch.registry.add_line(pp0, pp1)
    # Template: two lines joined end-to-end by a coincident
    # constraint (internal), with the free end tied to a foreign
    # point (external).
    a0 = sketch.registry.add_point(0.0, -2.0)
    a1 = sketch.registry.add_point(0.0, 0.0)
    b0 = sketch.registry.add_point(0.0, 0.0)
    b1 = sketch.registry.add_point(0.0, 2.0)
    line_a = sketch.registry.add_line(a0, a1)
    line_b = sketch.registry.add_line(b0, b1)
    foreign = sketch.registry.add_point(50.0, 50.0)
    sketch.constraints.append(CoincidentConstraint(a1, b0))
    sketch.constraints.append(CoincidentConstraint(b1, foreign))

    CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=path),
        [line_a, line_b],
    ).execute()
    sketch.solve()

    array = sketch.arrays[0]
    registry = sketch.registry
    # Every member group is still glued: the two lines share (a
    # coincident copy of) the middle point.
    for slot, eids in array.living_members(registry):
        first = registry.get_entity(eids[0])
        second = registry.get_entity(eids[1])
        assert isinstance(first, Line)
        assert isinstance(second, Line)
        joined_a = registry.get_point(first.p2_idx)
        joined_b = registry.get_point(second.p1_idx)
        assert (joined_a.x, joined_a.y) == pytest.approx(
            (joined_b.x, joined_b.y)
        )
    # The external tie is gone: the foreign point is untouched.
    assert (registry.get_point(foreign).x, registry.get_point(foreign).y) == (
        50.0,
        50.0,
    )


def test_world_anchored_constraints_erased_tangents_kept():
    """
    Horizontal/vertical constraints pin the template to the world
    axes and would fight the array rotations, so they are erased at
    extraction. Entity-only constraints such as tangents stay
    internal and are cloned onto every member.
    """
    sketch = Sketch()
    pp0 = sketch.registry.add_point(0.0, 0.0)
    pp1 = sketch.registry.add_point(30.0, 0.0)
    path = sketch.registry.add_line(pp0, pp1)

    # Template: a horizontal line with an arc tangent to it.
    a = sketch.registry.add_point(5.0, 0.0)
    b = sketch.registry.add_point(15.0, 0.0)
    line = sketch.registry.add_line(a, b)
    a_start = sketch.registry.add_point(18.0, 0.0)
    a_end = sketch.registry.add_point(21.0, 3.0)
    a_center = sketch.registry.add_point(18.0, 3.0)
    arc = sketch.registry.add_arc(a_start, a_end, a_center)
    sketch.constraints.append(HorizontalConstraint(a, b))
    sketch.constraints.append(TangentConstraint(line, arc))

    CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=path),
        [line, arc],
    ).execute()
    sketch.solve()

    registry = sketch.registry
    array = sketch.arrays[0]
    assert not [
        c
        for c in sketch.constraints
        if isinstance(c, (HorizontalConstraint, VerticalConstraint))
    ]
    # The template keeps its tangent; the copies are static baked
    # geometry (no constraints), tangent to each other by
    # construction.
    template_eids = array.living_members(registry)[0][1]
    assert [
        c
        for c in sketch.constraints
        if isinstance(c, TangentConstraint)
        and c.line_id in template_eids
        and c.shape_id in template_eids
    ]

    def member_shape(eids):
        shape = []
        for eid in eids:
            member_entity = registry.get_entity(eid)
            assert member_entity is not None
            for pid in member_entity.get_point_ids():
                pt = registry.get_point(pid)
                shape.append((pt.x, pt.y))
        return shape

    template_shape = member_shape(template_eids)

    def congruent(a, b):
        da = sorted(
            math.dist(p, q) for i, p in enumerate(a) for q in a[i + 1 :]
        )
        db = sorted(
            math.dist(p, q) for i, p in enumerate(b) for q in b[i + 1 :]
        )
        return da == pytest.approx(db)

    for slot, eids in array.living_members(registry):
        # Baked copies are congruent to the (tangent-preserving)
        # template shape.
        assert congruent(member_shape(eids), template_shape)
        # Static copies are outside the solver: their points are
        # fixed and carry no constraints.
        if slot != 0:
            for eid in eids:
                copy_entity = registry.get_entity(eid)
                assert copy_entity is not None
                for pid in copy_entity.get_point_ids():
                    assert registry.get_point(pid).fixed


def test_reapply_does_not_leak_fixed_copy_points():
    """
    Copies are static baked geometry: their points are fixed and
    carry no constraints (the solver never touches them). Re-applies
    must collect those fixed points — the general dependency
    calculation refuses to delete fixed points — and repeated
    re-applies must not accumulate geometry.
    """
    sketch = Sketch()
    pp0 = sketch.registry.add_point(0.0, 0.0)
    pp1 = sketch.registry.add_point(30.0, 0.0)
    path = sketch.registry.add_line(pp0, pp1)
    sp0 = sketch.registry.add_point(0.0, -2.0)
    sp1 = sketch.registry.add_point(0.0, 2.0)
    template = sketch.registry.add_line(sp0, sp1)

    CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=path),
        [template],
    ).execute()
    sketch.solve()

    array = sketch.arrays[0]
    registry = sketch.registry
    n_points = len(registry.points)
    n_entities = len(registry.entities)

    # Copies are fixed and unconstrained.
    for slot, eids in array.living_members(registry)[1:]:
        for eid in eids:
            copy_entity = registry.get_entity(eid)
            assert copy_entity is not None
            for pid in copy_entity.get_point_ids():
                assert registry.get_point(pid).fixed

    # Drag the template and re-apply via the solver sync several
    # times: geometry must stay bounded.
    tpl_entity = registry.get_entity(array.living_members(registry)[0][1][0])
    assert isinstance(tpl_entity, Line)
    dragged = registry.get_point(tpl_entity.p1_idx)
    for _ in range(10):
        dragged.x += 0.5
        sketch.solve()
        assert len(registry.points) == n_points
        assert len(registry.entities) == n_entities


def test_ellipse_helper_lines_follow_the_template():
    """
    An ellipse's construction lines share its points and belong to
    the template: extraction must keep them attached (they share the
    extracted points) and the placement must carry them along.
    """
    sketch = Sketch()
    c = sketch.registry.add_point(40.0, 20.0)
    rx = sketch.registry.add_point(50.0, 20.0)
    ry = sketch.registry.add_point(40.0, 30.0)
    eid = sketch.registry.add_ellipse(c, rx, ry)
    registry = sketch.registry
    h1 = sketch.registry.add_line(c, rx)
    h2 = sketch.registry.add_line(c, ry)
    ellipse = registry.get_entity(eid)
    assert isinstance(ellipse, Ellipse)
    ellipse.helper_line_ids = [h1, h2]

    # Guide path far from the drawn ellipse.
    pp0 = sketch.registry.add_point(0.0, 0.0)
    pp1 = sketch.registry.add_point(60.0, 0.0)
    path = sketch.registry.add_line(pp0, pp1)

    CreateArrayCommand(
        sketch,
        make_strategy(count=3, path_id=path),
        [eid],
    ).execute()
    sketch.solve()

    registry = sketch.registry
    array = sketch.arrays[0]
    template = registry.get_entity(array.living_members(registry)[0][1][0])
    assert isinstance(template, Ellipse)
    assert sorted(template.helper_line_ids) == sorted([h1, h2])
    for hid in template.helper_line_ids:
        helper = registry.get_entity(hid)
        assert isinstance(helper, Line)
        # The helpers share the template's (extracted) points, so
        # they follow it onto the guide.
        assert helper.get_point_ids()[0] == template.center_idx
