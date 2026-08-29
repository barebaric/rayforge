from typing import cast

from raygeo.geo import Matrix
from sketcher.core import Sketch
from sketcher.core.arrays import CurveAlongArray
from sketcher.core.entities import Bezier, Line
from sketcher.ui_gtk.hittest import SketchHitTester


class FakeCanvas:
    def __init__(self):
        self.view_transform = Matrix.translation(0, 0)


class FakeElement:
    def __init__(self, sketch):
        self.sketch = sketch
        self.canvas = FakeCanvas()
        self.content_transform = Matrix.translation(0, 0)
        self.point_radius = 5.0

    def get_world_transform(self):
        return Matrix.translation(0, 0)


class TestEntityZOrder:
    def test_topmost_entity_wins_on_overlap(self):
        """Later-drawn entities are picked first when they overlap."""
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        first = s.add_line(p1, p2)
        q1 = s.add_point(0, 0)
        q2 = s.add_point(100, 0)
        second = s.add_line(q1, q2)

        element = FakeElement(s)
        tester = SketchHitTester()
        hit = tester._hit_test_entities(50, 0, element)

        assert hit is not None
        assert hit.id == second
        assert hit.id != first

    def test_single_entity_still_hit(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2)

        element = FakeElement(s)
        tester = SketchHitTester()
        hit = tester._hit_test_entities(50, 0, element)

        assert hit is not None
        assert hit.id == line_id

    def test_miss_returns_none(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        s.add_line(p1, p2)

        element = FakeElement(s)
        tester = SketchHitTester()
        assert tester._hit_test_entities(50, 500, element) is None


class TestPointZOrder:
    def test_later_point_wins_tie(self):
        """Coincident points resolve to the one drawn last."""
        s = Sketch()
        first = s.add_point(100, 100)
        second = s.add_point(100, 100)

        element = FakeElement(s)
        tester = SketchHitTester()
        hit = tester._hit_test_points(100, 100, element)

        assert hit is not None
        assert hit == second
        assert hit != first

    def test_nearest_point_wins(self):
        s = Sketch()
        far = s.add_point(106, 100)
        near = s.add_point(101, 101)

        element = FakeElement(s)
        tester = SketchHitTester()
        hit = tester._hit_test_points(100, 100, element)

        assert hit == near
        assert hit != far


def _build_managed_sketch():
    """Guide bezier plus an array whose template start point sits
    exactly on the guide's start point, as the user sees it: one
    point, two registry entries."""
    s = Sketch()
    g0 = s.registry.add_point(0.0, 0.0)
    g1 = s.registry.add_point(100.0, 0.0)
    guide = s.registry.add_bezier(g0, g1, cp1=(30.0, 0.0), cp2=(70.0, 0.0))

    t0 = s.registry.add_point(0.0, 0.0)
    t1 = s.registry.add_point(12.0, 5.0)
    template = s.registry.add_line(t0, t1)

    c0 = s.registry.add_point(50.0, 0.0)
    c1 = s.registry.add_point(62.0, 5.0)
    copy = s.registry.add_line(c0, c1)

    array = CurveAlongArray(
        uid="hittest-array",
        guide_circle_id=guide,
        members=[(0, [template]), (1, [copy])],
    )
    s.arrays.append(array)
    return s, guide, array


class TestDerivedPointPicking:
    def test_guide_endpoint_wins_at_shared_location(self):
        """Template point and guide endpoint coincide: the pick must
        resolve to the guide's endpoint. Derived points lose even
        though they were created later (old z-order tie-break)."""
        s, guide, _array = _build_managed_sketch()
        bezier = cast(Bezier, s.registry.get_entity(guide))

        element = FakeElement(s)
        tester = SketchHitTester()
        assert tester._hit_test_points(0.0, 0.0, element) == bezier.start_idx

    def test_template_point_pickable_in_isolation(self):
        """Away from user geometry, derived points are still picked
        (template reshaping stays possible)."""
        s, _guide, array = _build_managed_sketch()
        _slot, template_eids = array.members[0]
        line = cast(Line, s.registry.get_entity(template_eids[0]))
        template_pt = s.registry.get_point(line.p2_idx)

        element = FakeElement(s)
        tester = SketchHitTester()
        assert tester._hit_test_points(12.0, 5.0, element) == template_pt.id

    def test_member_point_ids_include_template_and_copies(self):
        s, guide, array = _build_managed_sketch()
        derived = s.get_derived_point_ids()

        bezier = cast(Bezier, s.registry.get_entity(guide))
        assert bezier.start_idx not in derived
        assert bezier.end_idx not in derived

        member_pids = set()
        for _slot, eids in array.members:
            for eid in eids:
                entity = cast(Line, s.registry.get_entity(eid))
                member_pids.update(entity.get_point_ids())
        assert member_pids
        assert member_pids <= derived
