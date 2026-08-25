import pytest
from sketcher.core import Sketch
from sketcher.core.commands import MirrorAxis, MirrorCommand, MirrorDirection
from sketcher.core.constraints import (
    AngleConstraint,
    DistanceConstraint,
    TangentConstraint,
)
from sketcher.core.entities import Arc, Bezier
from sketcher.core.selection import SketchSelection


@pytest.fixture
def sketch_with_l():
    """Two lines forming an L at (0,0)."""
    s = Sketch()
    p1 = s.add_point(-100, 0)  # left of corner
    corner = s.add_point(0, 0)
    p3 = s.add_point(0, 100)  # above corner
    line1 = s.add_line(p1, corner)
    line2 = s.add_line(corner, p3)
    return s, [line1, line2], [p1, corner, p3]


class TestMirrorPrepare:
    def test_empty_selection_returns_none(self):
        s = Sketch()

        sel = SketchSelection()
        result = MirrorCommand.prepare(s, sel, MirrorDirection.VERTICAL)
        assert result is None

    def test_line_mirror_vertical(self, sketch_with_l):
        s, entity_ids, point_ids = sketch_with_l

        sel = SketchSelection()
        sel.entity_ids = entity_ids
        result = MirrorCommand.prepare(s, sel, MirrorDirection.VERTICAL)
        assert result is not None

        ent_ids, pt_ids, dropped, axis = result
        # bbox of points: x in [-100, 0], y in [0, 100]
        # VERTICAL flips Y, axis at y = (0 + 100) / 2 = 50
        assert axis.direction == MirrorDirection.VERTICAL
        assert axis.position == pytest.approx(50.0)
        assert set(ent_ids) == set(entity_ids)
        assert set(pt_ids) == set(point_ids)
        assert dropped == []

    def test_line_mirror_horizontal(self, sketch_with_l):
        s, entity_ids, _point_ids = sketch_with_l

        sel = SketchSelection()
        sel.entity_ids = entity_ids
        result = MirrorCommand.prepare(s, sel, MirrorDirection.HORIZONTAL)
        assert result is not None

        _, _, _, axis = result
        # HORIZONTAL flips X, axis at x = (-100 + 0) / 2 = -50
        assert axis.direction == MirrorDirection.HORIZONTAL
        assert axis.position == pytest.approx(-50.0)

    def test_external_constraint_dropped(self):
        """TangentConstraint with unselected shape is dropped."""
        s = Sketch()
        # Line (selected) tangent to circle (not selected)
        lp1 = s.add_point(0, 50)
        lp2 = s.add_point(100, 50)
        line_id = s.add_line(lp1, lp2)

        cc = s.add_point(50, 0)
        cr = s.add_point(50, 40)
        circle_id = s.add_circle(cc, cr)

        s.constraints.append(TangentConstraint(line_id, circle_id))

        sel = SketchSelection()
        sel.entity_ids = [line_id]
        result = MirrorCommand.prepare(s, sel, MirrorDirection.VERTICAL)
        assert result is not None
        _, _, dropped, _ = result
        assert len(dropped) == 1
        assert isinstance(dropped[0], TangentConstraint)

    def test_internal_constraint_kept(self):
        """DistanceConstraint between two selected points is kept."""
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2)

        s.constraints.append(DistanceConstraint(p1, p2, 100.0))

        sel = SketchSelection()
        sel.entity_ids = [line_id]
        result = MirrorCommand.prepare(s, sel, MirrorDirection.VERTICAL)
        assert result is not None
        _, _, dropped, _ = result
        assert dropped == []

    def test_expression_angle_constraint_dropped(self):
        """Expression-based AngleConstraint is not mirror-compatible."""
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        p3 = s.add_point(100, 100)
        # Two lines forming an angle at p2
        l1 = s.add_line(p1, p2)
        l2 = s.add_line(p2, p3)

        s.constraints.append(
            AngleConstraint(l1, l2, value=45.0, expression="45")
        )

        sel = SketchSelection()
        sel.entity_ids = [l1, l2]
        result = MirrorCommand.prepare(s, sel, MirrorDirection.VERTICAL)
        assert result is not None
        _, _, dropped, _ = result
        assert len(dropped) == 1
        assert isinstance(dropped[0], AngleConstraint)

    def test_numeric_angle_constraint_kept(self):
        """Plain numeric AngleConstraint is mirror-compatible."""
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        p3 = s.add_point(100, 100)
        l1 = s.add_line(p1, p2)
        l2 = s.add_line(p2, p3)

        s.constraints.append(AngleConstraint(l1, l2, value=45.0))

        sel = SketchSelection()
        sel.entity_ids = [l1, l2]
        result = MirrorCommand.prepare(s, sel, MirrorDirection.VERTICAL)
        assert result is not None
        _, _, dropped, _ = result
        assert dropped == []


class TestMirrorAxis:
    def test_vertical_flips_y(self):
        axis = MirrorAxis(MirrorDirection.VERTICAL, 50.0)
        assert axis.apply(10, 0) == (10, 100)
        assert axis.apply(10, 100) == (10, 0)
        assert axis.apply(10, 50) == (10, 50)

    def test_horizontal_flips_x(self):
        axis = MirrorAxis(MirrorDirection.HORIZONTAL, 50.0)
        assert axis.apply(0, 10) == (100, 10)
        assert axis.apply(100, 10) == (0, 10)
        assert axis.apply(50, 10) == (50, 10)

    def test_flip_offset_vertical(self):
        axis = MirrorAxis(MirrorDirection.VERTICAL, 0.0)
        assert axis.flip_offset((30, 40)) == (30, -40)

    def test_flip_offset_horizontal(self):
        axis = MirrorAxis(MirrorDirection.HORIZONTAL, 0.0)
        assert axis.flip_offset((30, 40)) == (-30, 40)


class TestMirrorExecute:
    def test_line_points_mirrored_in_place(self, sketch_with_l):
        s, entity_ids, point_ids = sketch_with_l

        sel = SketchSelection()
        sel.entity_ids = entity_ids

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        s.notify_update = lambda: None  # avoid solver for unit test
        cmd.capture_snapshot()
        cmd._do_execute()

        p1 = s.registry.get_point(point_ids[0])
        corner = s.registry.get_point(point_ids[1])
        p3 = s.registry.get_point(point_ids[2])

        # Original: p1=(-100,0), corner=(0,0), p3=(0,100)
        # VERTICAL flips Y, axis at y=50. Mirrored:
        # p1: x=-100, y=2*50-0=100 -> (-100, 100)
        # corner: x=0, y=2*50-0=100 -> (0, 100)
        # p3: x=0, y=2*50-100=0 -> (0, 0)
        assert p1.x == pytest.approx(-100)
        assert p1.y == pytest.approx(100)
        assert corner.x == pytest.approx(0)
        assert corner.y == pytest.approx(100)
        assert p3.x == pytest.approx(0)
        assert p3.y == pytest.approx(0)

    def test_no_new_points_created(self, sketch_with_l):
        s, entity_ids, _point_ids = sketch_with_l
        original_point_count = len(s.registry.points)

        sel = SketchSelection()
        sel.entity_ids = entity_ids

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(s.registry.points) == original_point_count

    def test_arc_clockwise_toggled(self):
        s = Sketch()
        start = s.add_point(100, 0)
        end = s.add_point(0, 100)
        center = s.add_point(0, 0)
        arc_id = s.add_arc(start, end, center, clockwise=True)

        sel = SketchSelection()
        sel.entity_ids = [arc_id]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.capture_snapshot()
        cmd._do_execute()

        arc = s.registry.get_entity(arc_id)
        assert isinstance(arc, Arc)
        assert arc.clockwise is False

    def test_bezier_cp_deltas_flipped(self):
        s = Sketch()
        start = s.add_point(0, 0)
        end = s.add_point(100, 0)
        bezier_id = s.add_bezier(start, end, cp1=(30, 40), cp2=(-20, -10))

        sel = SketchSelection()
        sel.entity_ids = [bezier_id]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.capture_snapshot()
        cmd._do_execute()

        bezier = s.registry.get_entity(bezier_id)
        assert isinstance(bezier, Bezier)
        # Vertical mirror flips dy component of cp deltas
        assert bezier.cp1 == (30, -40)
        assert bezier.cp2 == (-20, 10)

    def test_external_constraint_removed(self):
        """TangentConstraint to unselected circle is removed on execute."""
        s = Sketch()
        lp1 = s.add_point(0, 50)
        lp2 = s.add_point(100, 50)
        line_id = s.add_line(lp1, lp2)
        cc = s.add_point(50, 0)
        cr = s.add_point(50, 40)
        circle_id = s.add_circle(cc, cr)
        s.constraints.append(TangentConstraint(line_id, circle_id))

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(s.constraints) == 0

    def test_internal_distance_constraint_kept(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2)
        s.constraints.append(DistanceConstraint(p1, p2, 100.0))

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(s.constraints) == 1
        assert isinstance(s.constraints[0], DistanceConstraint)

    def test_numeric_angle_value_negated(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        p3 = s.add_point(100, 100)
        l1 = s.add_line(p1, p2)
        l2 = s.add_line(p2, p3)
        s.constraints.append(AngleConstraint(l1, l2, value=45.0))

        sel = SketchSelection()
        sel.entity_ids = [l1, l2]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(s.constraints) == 1
        assert s.constraints[0].value == pytest.approx(-45.0)


class TestMirrorUndo:
    def test_undo_restores_points(self, sketch_with_l):
        s, entity_ids, point_ids = sketch_with_l

        sel = SketchSelection()
        sel.entity_ids = entity_ids

        # Save original positions
        originals = {}
        for pid in point_ids:
            p = s.registry.get_point(pid)
            originals[pid] = (p.x, p.y)

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.execute()

        # Verify points moved (VERTICAL flips Y)
        moved = any(
            s.registry.get_point(pid).y != originals[pid][1]
            for pid in point_ids
        )
        assert moved

        cmd.undo()

        for pid in point_ids:
            p = s.registry.get_point(pid)
            assert p.x == pytest.approx(originals[pid][0])
            assert p.y == pytest.approx(originals[pid][1])

    def test_undo_restores_arc_clockwise(self):
        s = Sketch()
        start = s.add_point(100, 0)
        end = s.add_point(0, 100)
        center = s.add_point(0, 0)
        arc_id = s.add_arc(start, end, center, clockwise=True)

        sel = SketchSelection()
        sel.entity_ids = [arc_id]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.execute()

        arc = s.registry.get_entity(arc_id)
        assert isinstance(arc, Arc)
        assert arc.clockwise is False

        cmd.undo()

        arc = s.registry.get_entity(arc_id)
        assert isinstance(arc, Arc)
        assert arc.clockwise is True

    def test_undo_restores_bezier_cps(self):
        s = Sketch()
        start = s.add_point(0, 0)
        end = s.add_point(100, 0)
        bezier_id = s.add_bezier(start, end, cp1=(30, 40), cp2=(-20, -10))

        sel = SketchSelection()
        sel.entity_ids = [bezier_id]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.execute()

        bezier = s.registry.get_entity(bezier_id)
        assert isinstance(bezier, Bezier)
        assert bezier.cp1 == (30, -40)

        cmd.undo()

        bezier = s.registry.get_entity(bezier_id)
        assert isinstance(bezier, Bezier)
        assert bezier.cp1 == (30, 40)
        assert bezier.cp2 == (-20, -10)

    def test_undo_re_adds_dropped_constraints(self):
        s = Sketch()
        lp1 = s.add_point(0, 50)
        lp2 = s.add_point(100, 50)
        line_id = s.add_line(lp1, lp2)
        cc = s.add_point(50, 0)
        cr = s.add_point(50, 40)
        circle_id = s.add_circle(cc, cr)
        s.constraints.append(TangentConstraint(line_id, circle_id))

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.execute()
        assert len(s.constraints) == 0

        cmd.undo()
        assert len(s.constraints) == 1
        assert isinstance(s.constraints[0], TangentConstraint)

    def test_undo_restores_angle_value(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        p3 = s.add_point(100, 100)
        l1 = s.add_line(p1, p2)
        l2 = s.add_line(p2, p3)
        s.constraints.append(AngleConstraint(l1, l2, value=45.0))

        sel = SketchSelection()
        sel.entity_ids = [l1, l2]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.execute()
        assert s.constraints[0].value == pytest.approx(-45.0)

        cmd.undo()
        assert s.constraints[0].value == pytest.approx(45.0)


class TestMirrorBarePoints:
    def test_bare_point_selection_mirrors_points(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        p3 = s.add_point(50, 100)

        sel = SketchSelection()
        sel.point_ids = [p1, p2, p3]

        cmd = MirrorCommand(s, sel, MirrorDirection.VERTICAL)
        cmd.capture_snapshot()
        cmd._do_execute()

        # bbox y: [0, 100], VERTICAL flips Y, axis at y=50
        # p1: y=2*50-0=100, p2: y=2*50-0=100, p3: y=2*50-100=0
        assert s.registry.get_point(p1).y == pytest.approx(100)
        assert s.registry.get_point(p2).y == pytest.approx(100)
        assert s.registry.get_point(p3).y == pytest.approx(0)
