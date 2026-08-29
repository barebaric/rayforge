"""Regression tests for dragging the guide of a curve array.

Simulates the UI drag loop: guide geometry changes, then solve()
(sync_arrays re-applies the array). Members must stay rigid copies of
the template and the template's internal constraints must stay
satisfied after every frame.
"""

import math

import pytest
from sketcher.core.arrays import (
    CurveAlongArrayStrategy,
    resolve_template_center,
)
from sketcher.core.commands import CreateArrayCommand
from sketcher.core.constraints import CoincidentConstraint
from sketcher.core.params import ParameterContext
from sketcher.core.sketch import Sketch


def _build_sketch():
    """Bezier guide + a line/arc/line template with coincidences and a
    radius dimension, like a hand-drawn bump shape."""
    sketch = Sketch()
    # Guide: bezier from (0,0) to (40,0).
    g0 = sketch.registry.add_point(0.0, 0.0)
    g1 = sketch.registry.add_point(40.0, 0.0)
    guide = sketch.registry.add_bezier(
        g0, g1, cp1=(0.0, 15.0), cp2=(40.0, -15.0)
    )
    # Template: line + arc + line, endpoints tied by coincidences
    # (distinct registry points joined by constraints, as the solver
    # sees them in a hand-drawn shape).
    p0 = sketch.registry.add_point(-6.0, -2.0)
    p1 = sketch.registry.add_point(-3.0, -2.0)
    a0 = sketch.registry.add_point(-3.0, -2.0)
    a1 = sketch.registry.add_point(3.0, -2.0)
    center = sketch.registry.add_point(0.0, -4.0)
    p2 = sketch.registry.add_point(3.0, -2.0)
    p3 = sketch.registry.add_point(6.0, -2.0)
    line1 = sketch.registry.add_line(p0, p1)
    arc = sketch.registry.add_arc(a0, a1, center, cw=False)
    line2 = sketch.registry.add_line(p2, p3)
    sketch.constrain_coincident(p1, a0)
    sketch.constrain_coincident(p2, a1)
    sketch.constrain_radius(arc, 2.0)
    template = [line1, arc, line2]
    return sketch, guide, template


def _coincident_residual(sketch):
    ctx = ParameterContext()
    worst = 0.0
    for constr in sketch.constraints:
        if not isinstance(constr, CoincidentConstraint):
            continue
        err = constr.error(sketch.registry, ctx)
        err = err if isinstance(err, (list, tuple)) else [err]
        worst = max(worst, max(abs(v) for v in err))
    return worst


def _member_points(sketch, array, slot):
    eids = next(eids for s, eids in array.members if s == slot)
    pts = []
    for eid in eids:
        entity = sketch.registry.get_entity(eid)
        for pid in entity.get_point_ids():
            pt = sketch.registry.get_point(pid)
            pts.append((pt.x, pt.y))
    return pts


def _shape(points):
    """Pairwise distance multiset: rigid motions preserve it exactly."""
    dists = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dists.append(round(math.dist(points[i], points[j]), 9))
    return sorted(dists)


def _shapes_match(points_a, points_b, tol=1e-6):
    """Rigid-equivalence up to solver noise: the pairwise distance
    multisets must agree within the tolerance."""
    dists_a = _shape(points_a)
    dists_b = _shape(points_b)
    if len(dists_a) != len(dists_b):
        return False, float("inf")
    deviation = max(abs(a - b) for a, b in zip(dists_a, dists_b))
    return deviation <= tol, deviation


@pytest.fixture
def array_flow():
    sketch, guide, template = _build_sketch()
    strategy = CurveAlongArrayStrategy(
        count=6, path_entity_id=guide, align_to_tangent=True
    )
    cmd = CreateArrayCommand(sketch, strategy, list(template))
    cmd.execute()
    sketch.solve()
    array = sketch.arrays[0]
    return sketch, guide, array


def test_create_leaves_constraints_satisfied(array_flow):
    sketch, _guide, array = array_flow
    assert _coincident_residual(sketch) < 1e-6
    template_points = _member_points(sketch, array, 0)
    for slot in range(1, 6):
        rigid, deviation = _shapes_match(
            _member_points(sketch, array, slot), template_points
        )
        assert rigid, f"slot {slot}: shape deviation {deviation}"


def _drag_template(sketch, array, dx, dy):
    """Simulates the user dragging the whole template by (dx, dy)."""
    _slot, template_eids = array.members[0]
    pids: dict[int, None] = {}
    for eid in template_eids:
        entity = sketch.registry.get_entity(eid)
        for pid in entity.get_point_ids():
            pids.setdefault(pid)
    for pid in pids:
        pt = sketch.registry.get_point(pid)
        pt.x += dx
        pt.y += dy
    sketch.solve()


def _assert_members_follow(sketch, array, frame):
    """Every copy must be a rigid copy of the CURRENT template (the
    template accumulates ~1e-9 solver noise between frames, so the
    comparison runs against its present shape)."""
    template_points = _member_points(sketch, array, 0)
    for slot in range(1, 6):
        rigid, deviation = _shapes_match(
            _member_points(sketch, array, slot), template_points
        )
        assert rigid, (
            f"frame {frame}: slot {slot} deviates from template by {deviation}"
        )


def _template_center(sketch, array):
    """The template's logical center, exactly as the reanchor pins it
    (resolve_template_center: a lone Circle/Ellipse's center point,
    else the bbox center of the defining points). For Circle/Ellipse
    templates the pin is exact; for bbox-center templates the center
    wobbles by up to ~|dtangent| x extent under rotation, so pin
    assertions on such templates need a loose tolerance."""
    eids = array.members[0][1]
    pts = []
    for eid in eids:
        entity = sketch.registry.get_entity(eid)
        for pid in entity.get_point_ids():
            pt = sketch.registry.get_point(pid)
            if pt is not None:
                pts.append(pt)
    return resolve_template_center(sketch.registry, list(eids), pts)


def _guide_start(sketch, guide):
    bezier = sketch.registry.get_entity(guide)
    pt = sketch.registry.get_point(bezier.start_idx)
    return (pt.x, pt.y)


def test_user_repro_drag_template_away_and_back_then_guide(array_flow):
    """The user's repro: create, drag the template away, drag it back
    onto the start, then drag the guide -> flickering started.

    With the template's position guide-owned, the template is pinned
    to the guide's start: a template drag is undone by the next sync,
    and guide drags keep every frame consistent."""
    sketch, guide, array = array_flow
    bezier = sketch.registry.get_entity(guide)

    # Drag the template away: the next sync snaps it back onto the
    # guide's start point (the template's position is guide-owned).
    _drag_template(sketch, array, 10.0, 10.0)
    assert _template_center(sketch, array) == pytest.approx(
        _guide_start(sketch, guide), abs=0.6
    )
    assert _coincident_residual(sketch) < 1e-6
    _drag_template(sketch, array, -10.0, -10.0)
    assert _coincident_residual(sketch) < 1e-6

    # Now drag the guide control point over several frames.
    for frame in range(4):
        bezier.cp1 = (bezier.cp1[0] + 3.0, bezier.cp1[1] - 3.0)
        sketch.solve()
        residual = _coincident_residual(sketch)
        assert residual < 1e-6, f"frame {frame}: residual {residual}"
        assert _template_center(sketch, array) == pytest.approx(
            _guide_start(sketch, guide), abs=0.6
        )
        _assert_members_follow(sketch, array, frame)


def test_dragging_guide_endpoint_moves_template_with_it(array_flow):
    """Dragging the guide's start endpoint (a solver point) carries
    the pinned template and the copies along rigidly."""
    sketch, guide, array = array_flow
    bezier = sketch.registry.get_entity(guide)

    for frame in range(4):
        start_pt = sketch.registry.get_point(bezier.start_idx)
        start_pt.x += 3.0
        start_pt.y += 2.0
        sketch.solve()

        residual = _coincident_residual(sketch)
        assert residual < 1e-6, (
            f"frame {frame}: coincident residual {residual}"
        )
        assert _template_center(sketch, array) == pytest.approx(
            _guide_start(sketch, guide), abs=0.6
        )
        _assert_members_follow(sketch, array, frame)


def test_dragging_guide_cp_keeps_members_rigid(array_flow):
    """Dragging a bezier control point changes the guide signature but
    moves no solver point: every sync frame must leave the template's
    coincidences exactly satisfied and the copies rigid."""
    sketch, guide, array = array_flow
    bezier = sketch.registry.get_entity(guide)

    for frame in range(4):
        bezier.cp1 = (bezier.cp1[0] + 3.0, bezier.cp1[1] - 3.0)
        sketch.solve()

        residual = _coincident_residual(sketch)
        assert residual < 1e-6, (
            f"frame {frame}: coincident residual {residual}"
        )
        _assert_members_follow(sketch, array, frame)
