"""Regression tests for standalone points in array members.

A member shape can carry points referenced by no entity — e.g. a
rectangle's symmetry center, held only by a SymmetryConstraint between
two corners. Such points are array member data: they are baked into
every copy, moved by the same placements/motions as the member's
entity points, tracked on the Array definition (``standalone_pids``),
and hit-testing treats them as derived geometry.
"""

import pytest
from sketcher.core.arrays import CurveAlongArrayStrategy
from sketcher.core.commands import (
    CreateArrayCommand,
    EditArrayCommand,
    RemoveItemsCommand,
)
from sketcher.core.constraints import SymmetryConstraint
from sketcher.core.params import ParameterContext
from sketcher.core.sketch import Sketch


def build_rect_array_sketch(count=4):
    """Bezier guide + rectangle template with a symmetry-center
    standalone point, arrayed along the curve."""
    sketch = Sketch()
    g0 = sketch.registry.add_point(0.0, 0.0)
    g1 = sketch.registry.add_point(40.0, 0.0)
    guide = sketch.registry.add_bezier(
        g0, g1, cp1=(0.0, 15.0), cp2=(40.0, -15.0)
    )

    corners = [
        sketch.registry.add_point(x, y)
        for x, y in (
            (7.0, 8.0),
            (13.0, 8.0),
            (13.0, 12.0),
            (7.0, 12.0),
        )
    ]
    center = sketch.registry.add_point(10.0, 10.0)
    lines = [
        sketch.registry.add_line(corners[i], corners[(i + 1) % 4])
        for i in range(4)
    ]
    # The center is tied to the corners by point symmetry only.
    sketch.constrain_symmetry([center, corners[0], corners[2]], [])
    sketch.constrain_symmetry([center, corners[1], corners[3]], [])

    strategy = CurveAlongArrayStrategy(
        count=count, path_entity_id=guide, align_to_tangent=True
    )
    cmd = CreateArrayCommand(sketch, strategy, lines)
    cmd.execute()
    sketch.solve()
    return sketch, guide, sketch.arrays[0], center, cmd


def member_entity_points(sketch, array, slot):
    eids = next(eids for s, eids in array.members if s == slot)
    pts = []
    for eid in eids:
        entity = sketch.registry.get_entity(eid)
        for pid in entity.get_point_ids():
            pt = sketch.registry.get_point(pid)
            pts.append((pt.x, pt.y))
    return pts


def member_center(sketch, array, slot):
    pids = array.standalone_pids[slot]
    assert len(pids) == 1
    pt = sketch.registry.get_point(pids[0])
    return (pt.x, pt.y)


def assert_center_is_corner_midpoint(sketch, array, slot):
    pts = member_entity_points(sketch, array, slot)
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    assert member_center(sketch, array, slot) == pytest.approx(
        (cx, cy), abs=1e-6
    )


def symmetry_residual(sketch):
    ctx = ParameterContext()
    worst = 0.0
    for constr in sketch.constraints:
        if not isinstance(constr, SymmetryConstraint):
            continue
        err = constr.error(sketch.registry, ctx)
        err = err if isinstance(err, (list, tuple)) else [err]
        worst = max(worst, max(abs(v) for v in err))
    return worst


@pytest.fixture
def rect_array():
    return build_rect_array_sketch()


def test_create_tracks_standalone_points_per_member(rect_array):
    sketch, _guide, array, center_pid, _cmd = rect_array

    assert array.standalone_pids[0] == [center_pid]
    for slot in range(1, array.count):
        assert len(array.standalone_pids[slot]) == 1
        assert_center_is_corner_midpoint(sketch, array, slot)
    assert symmetry_residual(sketch) < 1e-6


def test_standalone_points_are_derived_geometry(rect_array):
    """The members' standalone points must lose hittests against user
    geometry (e.g. the guide endpoint under the template center)."""
    sketch, guide, array, center_pid, _cmd = rect_array
    derived = sketch.get_derived_point_ids()
    assert center_pid in derived
    for slot in range(1, array.count):
        assert set(array.standalone_pids[slot]) <= derived

    bezier = sketch.registry.get_entity(guide)
    assert bezier.start_idx not in derived


def test_guide_drag_keeps_copy_centers_with_copies(rect_array):
    sketch, guide, array, _center_pid, _cmd = rect_array
    bezier = sketch.registry.get_entity(guide)

    for frame in range(4):
        start_pt = sketch.registry.get_point(bezier.start_idx)
        start_pt.x += 3.0
        start_pt.y -= 2.0
        sketch.solve()

        assert symmetry_residual(sketch) < 1e-6
        for slot in range(array.count):
            assert_center_is_corner_midpoint(sketch, array, slot)


def test_gap_fill_creates_standalone_points(rect_array):
    """Copies regenerated after a deletion must carry their own
    standalone points, positioned with the copy."""
    sketch, _guide, array, _center_pid, _cmd = rect_array
    victims = [eid for _s, eids in array.members[1:2] for eid in eids]
    points, entities, constraints = RemoveItemsCommand.calculate_dependencies(
        sketch,
        _Selection(victims),
    )
    RemoveItemsCommand(
        sketch,
        "",
        points=points,
        entities=entities,
        constraints=constraints,
    ).execute()

    EditArrayCommand(
        sketch, array, array.make_strategy(sketch.registry)
    ).execute()
    sketch.solve()

    for slot in range(array.count):
        assert slot in array.standalone_pids
        assert_center_is_corner_midpoint(sketch, array, slot)


def test_undo_create_restores_template_center(rect_array):
    sketch, _guide, _array, center_pid, cmd = rect_array
    placed = sketch.registry.get_point(center_pid)
    assert (placed.x, placed.y) != pytest.approx((10.0, 10.0))

    cmd.undo()

    restored = sketch.registry.get_point(center_pid)
    assert (restored.x, restored.y) == pytest.approx((10.0, 10.0))


class _Selection:
    def __init__(self, entity_ids):
        self.entity_ids = set(entity_ids)
        self.point_ids = set()
        self.constraint_idx = None
