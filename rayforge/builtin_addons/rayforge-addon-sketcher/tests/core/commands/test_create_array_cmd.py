import math

import pytest
from sketcher.core.arrays import CircularArrayStrategy
from sketcher.core.commands import CreateArrayCommand
from sketcher.core.constraints import RadiusConstraint
from sketcher.core.sketch import Sketch


@pytest.fixture
def sketch_with_template():
    sketch = Sketch()
    p0 = sketch.registry.add_point(0, 0)
    p1 = sketch.registry.add_point(20, 0)
    line = sketch.registry.add_line(p0, p1)
    return sketch, line, p0, p1


def make_strategy(count=4, radius=20.0, rotate=True):
    return CircularArrayStrategy(
        count=count,
        total_angle_deg=360.0,
        center=(0.0, 0.0),
        radius=radius,
        rotate_copies=rotate,
    )


def test_creates_copies_and_master(sketch_with_template):
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()

    entities = sketch.registry.entities
    assert len(entities) == 5

    copies = [e for e in entities if e.array_copy]
    assert len(copies) == 3

    # The master is a construction circle with a radius constraint.
    master = next(e for e in entities if e.type == "circle")
    assert master.construction is True

    radius_constraints = [
        c for c in sketch.constraints if isinstance(c, RadiusConstraint)
    ]
    assert len(radius_constraints) == 1
    assert radius_constraints[0].entity_id == master.id


def test_members_are_static_baked_copies(sketch_with_template):
    """Copies are static baked geometry: no constraints reference
    their points and the points are fixed (outside the solver)."""
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()

    copy_pids = {
        pid
        for eid in cmd.created_entity_ids
        for pid in sketch.registry.get_entity(eid).get_point_ids()
    }
    for constr in sketch.constraints:
        assert not constr.get_referenced_point_ids() & copy_pids
    for pid in copy_pids:
        assert sketch.registry.get_point(pid).fixed


def test_deleting_a_member_keeps_others_in_place(sketch_with_template):
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()
    array = cmd.array
    assert array is not None
    positions_before = sorted(
        (round(p.x, 3), round(p.y, 3)) for p in sketch.registry.points
    )

    victim = array.living_entity_ids(sketch.registry)[3]
    sketch.registry.remove_entities_by_id([victim])
    sketch.prune_arrays()

    assert len(array.living_entity_ids(sketch.registry)) == 3
    # The emptied member group is dropped by pruning.
    assert len(array.members) == 3
    positions_after = sorted(
        (round(p.x, 3), round(p.y, 3)) for p in sketch.registry.points
    )
    # Survivors did not move: only the deleted member is gone.
    removed = set(positions_before) - set(positions_after)
    assert len(removed) <= 2


def test_registers_array_definition(sketch_with_template):
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(count=5), [line])
    cmd.execute()

    assert len(sketch.arrays) == 1
    array = sketch.arrays[0]
    assert array.count == 5
    assert [slot for slot, _eids in array.members] == list(range(5))
    assert all(len(eids) == 1 for _s, eids in array.members)
    flat = [eid for _s, eids in array.members for eid in eids]
    assert cmd.guide_circle_id == array.guide_circle_id
    assert registry_has_all(sketch, flat)


def registry_has_all(sketch, entity_ids):
    return all(
        sketch.registry.get_entity(eid) is not None for eid in entity_ids
    )


def test_copy_positions_are_rotated(sketch_with_template):
    """
    The template is placed onto the guide circle (radial translation,
    shape preserved) and the copies are rigid rotations of it about
    the center. Template (0,0)-(20,0) has its center at distance 10,
    so it is projected onto the radius-20 circle: (10,0)-(30,0).
    """
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()

    positions = sorted(
        (round(p.x, 6), round(p.y, 6)) for p in sketch.registry.points
    )
    assert (10.0, 0.0) in positions
    assert (30.0, 0.0) in positions
    assert (0.0, 10.0) in positions
    assert (0.0, 30.0) in positions
    assert (-10.0, 0.0) in positions
    assert (-30.0, 0.0) in positions
    assert (0.0, -10.0) in positions
    assert (0.0, -30.0) in positions


def test_apply_places_template_on_the_guide(sketch_with_template):
    """Creating the array places the template onto the guide circle:
    its points end up at the circle radius, shape preserved."""
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()

    assert cmd.array is not None
    circle = next(e for e in sketch.registry.entities if e.type == "circle")
    center = sketch.registry.get_point(circle.center_idx)
    eids = cmd.array.living_members(sketch.registry)[0][1]
    pids = CreateArrayCommand.collect_template_point_ids(sketch.registry, eids)
    pts = [sketch.registry.get_point(pid) for pid in pids]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    assert math.hypot(cx - center.x, cy - center.y) == pytest.approx(20.0)
    # Shape preserved: the member is only translated.
    xs = [p.x for p in pts]
    assert max(xs) - min(xs) == pytest.approx(20.0)


def test_translate_mode_bakes_static_copies(sketch_with_template):
    sketch, line, _p0, _p1 = sketch_with_template
    strategy = CircularArrayStrategy(
        count=3,
        total_angle_deg=180.0,
        center=(0.0, 0.0),
        rotate_copies=False,
    )
    cmd = CreateArrayCommand(sketch, strategy, [line])
    cmd.execute()
    assert len([e for e in sketch.registry.entities if e.array_copy]) == 2


def test_undo_removes_everything_including_array(sketch_with_template):
    sketch, line, _p0, _p1 = sketch_with_template
    num_points_before = len(sketch.registry.points)
    num_entities_before = len(sketch.registry.entities)

    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()
    assert len(sketch.registry.entities) == num_entities_before + 4
    assert len(sketch.arrays) == 1

    cmd.undo()
    assert len(sketch.registry.entities) == num_entities_before
    assert len(sketch.registry.points) == num_points_before
    assert sketch.constraints == []
    assert sketch.arrays == []


def test_redo_restores_array(sketch_with_template):
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()
    cmd.undo()
    cmd.execute()

    assert len([e for e in sketch.registry.entities if e.array_copy]) == 3
    assert len(sketch.arrays) == 1


def test_serialization_round_trip_preserves_array(sketch_with_template):
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(count=3), [line])
    cmd.execute()

    restored = Sketch.from_dict(sketch.to_dict())
    copies = [e for e in restored.registry.entities if e.array_copy]
    assert len(copies) >= 2
    assert len(restored.arrays) == 1
    assert restored.arrays[0].count == 3


def test_empty_selection_creates_nothing():
    sketch = Sketch()
    cmd = CreateArrayCommand(sketch, make_strategy(), [])
    cmd.execute()
    assert cmd.created_entity_ids == []
    assert sketch.registry.entities == []
    assert sketch.arrays == []


def test_solver_keeps_geometry_stable_after_apply(sketch_with_template):
    """Solving after creation must not visibly move geometry."""
    sketch, line, _p0, _p1 = sketch_with_template
    cmd = CreateArrayCommand(sketch, make_strategy(), [line])
    cmd.execute()

    before = [(p.id, p.x, p.y) for p in sketch.registry.points]
    sketch.solve()
    for pid, x, y in before:
        pt = sketch.registry.get_point(pid)
        assert pt.x == pytest.approx(x, abs=1e-3)
        assert pt.y == pytest.approx(y, abs=1e-3)
