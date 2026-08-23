from raygeo.geo import Matrix
from sketcher.core import Sketch
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
