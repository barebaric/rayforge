import pytest
from sketcher.core.commands import CreatePatternCommand
from sketcher.core.constraints import RadiusConstraint, RotationalConstraint
from sketcher.core.patterns import CircularPatternParams, SketchArrayMode
from sketcher.core.sketch import Sketch


@pytest.fixture
def sketch_with_seed():
    sketch = Sketch()
    p0 = sketch.registry.add_point(0, 0)
    p1 = sketch.registry.add_point(20, 0)
    line = sketch.registry.add_line(p0, p1)
    return sketch, line, p0, p1


def make_params(count=4, radius=20.0, rotate=True):
    return CircularPatternParams(
        count=count,
        total_angle_deg=360.0,
        center=(0.0, 0.0),
        radius=radius,
        rotate_copies=rotate,
    )


def test_creates_copies_and_master(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()

    entities = sketch.registry.entities
    assert len(entities) == 5

    copies = [e for e in entities if e.pattern_copy]
    assert len(copies) == 3

    # The master is a construction circle with a radius constraint.
    master = next(e for e in entities if e.type == "circle")
    assert master.construction is True

    radius_constraints = [
        c for c in sketch.constraints if isinstance(c, RadiusConstraint)
    ]
    assert len(radius_constraints) == 1
    assert radius_constraints[0].entity_id == master.id


def test_members_linked_to_template(sketch_with_seed):
    """Copies carry rotational constraints back to the template member."""
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()

    rotational = [
        c for c in sketch.constraints if isinstance(c, RotationalConstraint)
    ]
    # Two template points x three copies.
    assert len(rotational) == 6

    template_pids = set(
        CreatePatternCommand.collect_seed_point_ids(sketch.registry, [line])
    )
    master = next(e for e in sketch.registry.entities if e.type == "circle")
    for c in rotational:
        assert c.p1 in template_pids
        assert c.center == master.center_idx


def test_deleting_a_member_keeps_others_in_place(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()
    pattern = cmd.pattern
    assert pattern is not None
    positions_before = sorted(
        (round(p.x, 3), round(p.y, 3)) for p in sketch.registry.points
    )

    victim = pattern.living_entity_ids(sketch.registry)[3]
    sketch.registry.remove_entities_by_id([victim])
    sketch.prune_patterns()

    assert len(pattern.living_entity_ids(sketch.registry)) == 3
    # The emptied member group is dropped by pruning.
    assert len(pattern.members) == 3
    positions_after = sorted(
        (round(p.x, 3), round(p.y, 3)) for p in sketch.registry.points
    )
    # Survivors did not move: only the deleted member is gone.
    removed = set(positions_before) - set(positions_after)
    assert len(removed) <= 2


def test_registers_pattern_definition(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(count=5), [line]
    )
    cmd.execute()

    assert len(sketch.patterns) == 1
    pattern = sketch.patterns[0]
    assert pattern.count == 5
    assert [slot for slot, _eids in pattern.members] == list(range(5))
    assert all(len(eids) == 1 for _s, eids in pattern.members)
    flat = [eid for _s, eids in pattern.members for eid in eids]
    assert cmd.guide_circle_id == pattern.guide_circle_id
    assert registry_has_all(sketch, flat)


def registry_has_all(sketch, entity_ids):
    return all(
        sketch.registry.get_entity(eid) is not None for eid in entity_ids
    )


def test_copy_positions_are_rotated(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()

    positions = sorted(
        (round(p.x, 6), round(p.y, 6)) for p in sketch.registry.points
    )
    assert (20.0, 0.0) in positions
    assert (0.0, 20.0) in positions
    assert (-20.0, 0.0) in positions
    assert (round(-0.0, 6), -20.0) in positions or (0.0, -20.0) in positions


def test_apply_does_not_move_geometry(sketch_with_seed):
    """Creating the pattern must not shift existing geometry."""
    sketch, line, _p0, _p1 = sketch_with_seed
    before = [(p.id, p.x, p.y) for p in sketch.registry.points]
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()

    for pid, x, y in before:
        pt = sketch.registry.get_point(pid)
        assert (pt.x, pt.y) == (x, y)


def test_translate_mode_bakes_static_copies(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    params = CircularPatternParams(
        count=3,
        total_angle_deg=180.0,
        center=(0.0, 0.0),
        rotate_copies=False,
    )
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, params, [line]
    )
    cmd.execute()
    assert len([e for e in sketch.registry.entities if e.pattern_copy]) == 2


def test_undo_removes_everything_including_pattern(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    num_points_before = len(sketch.registry.points)
    num_entities_before = len(sketch.registry.entities)

    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()
    assert len(sketch.registry.entities) == num_entities_before + 4
    assert len(sketch.patterns) == 1

    cmd.undo()
    assert len(sketch.registry.entities) == num_entities_before
    assert len(sketch.registry.points) == num_points_before
    assert sketch.constraints == []
    assert sketch.patterns == []


def test_redo_restores_pattern(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()
    cmd.undo()
    cmd.execute()

    assert len([e for e in sketch.registry.entities if e.pattern_copy]) == 3
    assert len(sketch.patterns) == 1


def test_serialization_round_trip_preserves_pattern(sketch_with_seed):
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(count=3), [line]
    )
    cmd.execute()

    restored = Sketch.from_dict(sketch.to_dict())
    copies = [e for e in restored.registry.entities if e.pattern_copy]
    assert len(copies) >= 2
    assert len(restored.patterns) == 1
    assert restored.patterns[0].count == 3


def test_empty_selection_creates_nothing():
    sketch = Sketch()
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), []
    )
    cmd.execute()
    assert cmd.created_entity_ids == []
    assert sketch.registry.entities == []
    assert sketch.patterns == []


def test_solver_keeps_geometry_stable_after_apply(sketch_with_seed):
    """Solving after creation must not visibly move geometry."""
    sketch, line, _p0, _p1 = sketch_with_seed
    cmd = CreatePatternCommand(
        sketch, SketchArrayMode.CIRCULAR, make_params(), [line]
    )
    cmd.execute()

    before = [(p.id, p.x, p.y) for p in sketch.registry.points]
    sketch.solve()
    for pid, x, y in before:
        pt = sketch.registry.get_point(pid)
        assert pt.x == pytest.approx(x, abs=1e-3)
        assert pt.y == pytest.approx(y, abs=1e-3)
