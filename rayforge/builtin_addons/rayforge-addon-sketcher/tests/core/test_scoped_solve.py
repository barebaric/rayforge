import pytest
from sketcher.core.components import (
    compute_constraint_components,
    get_referenced_points,
)
from sketcher.core.constraints import (
    Constraint,
    DragConstraint,
    TangentConstraint,
)
from sketcher.core.params import ParameterContext
from sketcher.core.registry import EntityRegistry
from sketcher.core.sketch import Sketch
from sketcher.core.solver import Solver


def add_rect(sketch, x, y, w, h, anchored=True):
    """Adds a constrained rectangle and returns its point IDs.

    Anchored rectangles are fully constrained; unanchored ones can
    translate, which makes them draggable.
    """
    p1 = sketch.add_point(x, y)
    p2 = sketch.add_point(x + w, y)
    p3 = sketch.add_point(x + w, y + h)
    p4 = sketch.add_point(x, y + h)
    sketch.add_line(p1, p2)
    sketch.add_line(p2, p3)
    sketch.add_line(p3, p4)
    sketch.add_line(p4, p1)
    sketch.constrain_horizontal(p1, p2)
    sketch.constrain_horizontal(p4, p3)
    sketch.constrain_vertical(p1, p4)
    sketch.constrain_vertical(p2, p3)
    if anchored:
        sketch.constrain_distance(p1, p2, w)
        sketch.constrain_distance(p1, p4, h)
        sketch.registry.get_point(p1).fixed = True
    return [p1, p2, p3, p4]


def positions(sketch, pids):
    return {
        pid: (
            sketch.registry.get_point(pid).x,
            sketch.registry.get_point(pid).y,
        )
        for pid in pids
    }


def drag_scope(sketch, dragged_points):
    """Mirrors SelectTool._compute_drag_scope."""
    scope = set(dragged_points)
    for component in compute_constraint_components(
        sketch.registry, sketch.constraints
    ):
        if not component.isdisjoint(scope):
            scope |= component
    return scope


def test_components_two_islands():
    sketch = Sketch("test")
    rect_a = add_rect(sketch, 0, 0, 10, 5)
    rect_b = add_rect(sketch, 50, 0, 10, 5)

    components = compute_constraint_components(
        sketch.registry, sketch.constraints
    )

    assert len(components) == 2
    assert set(rect_a) in components
    assert set(rect_b) in components


def test_components_merge_via_shared_constraint():
    sketch = Sketch("test")
    rect_a = add_rect(sketch, 0, 0, 10, 5)
    rect_b = add_rect(sketch, 50, 0, 10, 5)
    sketch.constrain_distance(rect_a[0], rect_b[0], 50.0)

    components = compute_constraint_components(
        sketch.registry, sketch.constraints
    )

    assert len(components) == 1
    assert components[0] == set(rect_a) | set(rect_b)


def test_entity_level_constraint_couples_points():
    registry = EntityRegistry()
    center = registry.add_point(0, 0)
    radius_pt = registry.add_point(5, 0)
    lp1 = registry.add_point(10, 10)
    lp2 = registry.add_point(20, 10)
    circle = registry.add_circle(center, radius_pt)
    line = registry.add_line(lp1, lp2)
    constraint = TangentConstraint(line, circle)

    assert get_referenced_points(registry, constraint) == {
        center,
        radius_pt,
        lp1,
        lp2,
    }
    components = compute_constraint_components(registry, [constraint])
    assert components == [{center, radius_pt, lp1, lp2}]


def test_solver_point_filter_ignores_other_points():
    registry = EntityRegistry()
    params = ParameterContext()
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(10, 0)

    solver = Solver(registry, params, [], point_filter={p1})
    assert solver.solve(update_dof=False) is True

    assert registry.get_point(p1).x == pytest.approx(0.0)
    assert registry.get_point(p1).y == pytest.approx(0.0)
    assert registry.get_point(p2).x == 10.0

    solver = Solver(
        registry, params, [DragConstraint(p1, 5.0, 5.0)], point_filter={p1}
    )
    assert solver.solve(update_dof=False) is True

    assert registry.get_point(p1).x == pytest.approx(5.0, abs=1e-4)
    assert registry.get_point(p2).x == 10.0


def test_scoped_solve_moves_dragged_island_only():
    sketch = Sketch("test")
    rect_a = add_rect(sketch, 0, 0, 10, 5, anchored=False)
    rect_b = add_rect(sketch, 50, 0, 10, 5)
    before_a = positions(sketch, rect_a)
    before_b = positions(sketch, rect_b)

    delta = 5.0
    drag_constraints: list[Constraint] = [
        DragConstraint(pid, x + delta, y, weight=1.0)
        for pid, (x, y) in before_a.items()
    ]
    success = sketch.solve(
        extra_constraints=drag_constraints,
        update_constraint_status=False,
        point_scope=set(rect_a),
    )

    assert success is True
    assert positions(sketch, rect_b) == before_b
    for pid, (x, y) in positions(sketch, rect_a).items():
        assert x == pytest.approx(before_a[pid][0] + delta, abs=1e-3)
        assert y == pytest.approx(before_a[pid][1], abs=1e-3)


def test_scoped_solve_matches_global_solve():
    delta = (3.0, 2.0)

    def build():
        sketch = Sketch("test")
        rect_a = add_rect(sketch, 0, 0, 10, 5, anchored=False)
        rect_b = add_rect(sketch, 50, 0, 10, 5)
        return sketch, rect_a, rect_b

    sketch_g, rect_a_g, rect_b_g = build()
    sketch_s, rect_a_s, rect_b_s = build()

    start_a = positions(sketch_g, rect_a_g)
    global_constraints: list[Constraint] = [
        DragConstraint(pid, x + delta[0], y + delta[1], weight=1.0)
        for pid, (x, y) in start_a.items()
    ]
    global_constraints += [
        DragConstraint(pid, x, y, weight=0.01)
        for pid, (x, y) in positions(sketch_g, rect_b_g).items()
        if not sketch_g.registry.get_point(pid).fixed
    ]
    sketch_g.solve(
        extra_constraints=global_constraints,
        update_constraint_status=False,
    )

    scope = drag_scope(sketch_s, rect_a_s)
    assert scope == set(rect_a_s)
    scoped_constraints: list[Constraint] = [
        DragConstraint(pid, x + delta[0], y + delta[1], weight=1.0)
        for pid, (x, y) in positions(sketch_s, rect_a_s).items()
    ]
    sketch_s.solve(
        extra_constraints=scoped_constraints,
        update_constraint_status=False,
        point_scope=scope,
    )

    for pid_g, pid_s in zip(rect_a_g, rect_a_s):
        pg = sketch_g.registry.get_point(pid_g)
        ps = sketch_s.registry.get_point(pid_s)
        assert pg.x == pytest.approx(ps.x, abs=1e-4)
        assert pg.y == pytest.approx(ps.y, abs=1e-4)
    for pid_g, pid_s in zip(rect_b_g, rect_b_s):
        pg = sketch_g.registry.get_point(pid_g)
        ps = sketch_s.registry.get_point(pid_s)
        assert pg.x == pytest.approx(ps.x, abs=1e-6)
        assert pg.y == pytest.approx(ps.y, abs=1e-6)


def test_scoped_solve_falls_back_on_stale_scope():
    sketch = Sketch("test")
    add_rect(sketch, 0, 0, 10, 5, anchored=False)
    p_far = sketch.add_point(200, 200)

    drag_constraints: list[Constraint] = [
        DragConstraint(p_far, 300.0, 200.0, weight=1.0)
    ]
    sketch.solve(
        extra_constraints=drag_constraints,
        update_constraint_status=False,
        point_scope={p_far, 99999},
    )

    p = sketch.registry.get_point(p_far)
    assert p.x == pytest.approx(300.0, abs=1e-3)
    assert p.y == pytest.approx(200.0, abs=1e-3)


def test_tool_style_drag_keeps_other_islands_frozen():
    sketch = Sketch("test")
    rect_a = add_rect(sketch, 0, 0, 10, 5, anchored=False)
    rect_b = add_rect(sketch, 50, 0, 10, 5)
    rect_c = add_rect(sketch, 0, 50, 10, 5)
    frozen_before = {
        pid: pos
        for rect in (rect_b, rect_c)
        for pid, pos in positions(sketch, rect).items()
    }

    dragged = set(rect_a)
    scope = drag_scope(sketch, dragged)
    start_a = positions(sketch, dragged)

    for step in range(1, 11):
        drag_constraints: list[Constraint] = [
            DragConstraint(pid, x + step, y, weight=1.0)
            for pid, (x, y) in start_a.items()
        ]
        drag_constraints += [
            DragConstraint(pid, x, y, weight=0.01)
            for pid, (x, y) in positions(sketch, scope).items()
            if pid not in dragged
        ]
        sketch.solve(
            extra_constraints=drag_constraints,
            update_constraint_status=False,
            point_scope=scope,
        )

    frozen_after = {
        pid: pos
        for rect in (rect_b, rect_c)
        for pid, pos in positions(sketch, rect).items()
    }
    assert frozen_after == frozen_before
    for pid, (x, y) in positions(sketch, dragged).items():
        assert x == pytest.approx(start_a[pid][0] + 10, abs=1e-2)
        assert y == pytest.approx(start_a[pid][1], abs=1e-2)

    sketch.solve()
    assert {
        pid: pos
        for rect in (rect_b, rect_c)
        for pid, pos in positions(sketch, rect).items()
    } == frozen_before
