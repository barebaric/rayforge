import pytest
from sketcher.core import Sketch
from sketcher.core.commands import DuplicateCommand
from sketcher.core.constraints import (
    DistanceConstraint,
    TangentConstraint,
)
from sketcher.core.entities import Arc, Line
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


class TestDuplicatePrepare:
    def test_empty_selection_returns_none(self):
        s = Sketch()

        sel = SketchSelection()
        result = DuplicateCommand.prepare(s, sel)
        assert result is None

    def test_line_selection_resolved(self, sketch_with_l):
        s, entity_ids, point_ids = sketch_with_l

        sel = SketchSelection()
        sel.entity_ids = entity_ids
        result = DuplicateCommand.prepare(s, sel)
        assert result is not None

        ent_ids, pt_ids, constraints = result
        assert set(ent_ids) == set(entity_ids)
        assert set(pt_ids) == set(point_ids)
        assert constraints == []

    def test_internal_constraint_included(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2)
        s.constraints.append(DistanceConstraint(p1, p2, 100.0))

        sel = SketchSelection()
        sel.entity_ids = [line_id]
        result = DuplicateCommand.prepare(s, sel)
        assert result is not None
        _, _, constraints = result
        assert len(constraints) == 1
        assert isinstance(constraints[0], DistanceConstraint)

    def test_external_constraint_excluded(self):
        """TangentConstraint with unselected shape is not duplicated."""
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
        result = DuplicateCommand.prepare(s, sel)
        assert result is not None
        _, _, constraints = result
        assert constraints == []

    def test_bare_points_only(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)

        sel = SketchSelection()
        sel.point_ids = [p1, p2]
        result = DuplicateCommand.prepare(s, sel)
        assert result is not None
        ent_ids, pt_ids, _ = result
        assert ent_ids == []
        assert set(pt_ids) == {p1, p2}


class TestDuplicateExecute:
    def test_creates_copies_at_same_position(self, sketch_with_l):
        s, entity_ids, point_ids = sketch_with_l
        num_points = len(s.registry.points)
        num_entities = len(s.registry.entities)

        sel = SketchSelection()
        sel.entity_ids = entity_ids

        result = DuplicateCommand.prepare(s, sel)
        assert result is not None
        _, dup_point_ids, _ = result

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(s.registry.points) == num_points + len(dup_point_ids)
        assert len(s.registry.entities) == 2 * num_entities
        assert len(cmd.new_entity_ids) == 2

        for eid in cmd.new_entity_ids:
            clone = s.registry.get_entity(eid)
            assert clone is not None
            assert clone.id not in entity_ids
            for pid in clone.get_point_ids():
                assert pid not in point_ids

    def test_originals_unchanged(self, sketch_with_l):
        s, entity_ids, point_ids = sketch_with_l

        sel = SketchSelection()
        sel.entity_ids = entity_ids

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        for pid in point_ids:
            p = s.registry.get_point(pid)
            assert p.x == pytest.approx([-100, 0, 0][point_ids.index(pid)])
            assert p.y == pytest.approx([0, 0, 100][point_ids.index(pid)])
        assert all(e.id in entity_ids for e in s.registry.entities[:2])

    def test_internal_constraint_duplicated_and_remapped(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2)
        s.constraints.append(DistanceConstraint(p1, p2, 100.0))

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(s.constraints) == 2
        clone = s.constraints[1]
        assert isinstance(clone, DistanceConstraint)
        assert clone is not s.constraints[0]
        assert clone.p1 != p1
        assert clone.p2 != p2
        assert clone.value == pytest.approx(100.0)

        clone_line = s.registry.get_entity(cmd.new_entity_ids[0])
        assert isinstance(clone_line, Line)
        assert set(clone_line.get_point_ids()) == {clone.p1, clone.p2}

    def test_external_constraint_not_duplicated(self):
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

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(s.constraints) == 1
        assert isinstance(s.constraints[0], TangentConstraint)
        assert s.constraints[0].shape_id == circle_id

    def test_arc_state_copied(self):
        s = Sketch()
        start = s.add_point(100, 0)
        end = s.add_point(0, 100)
        center = s.add_point(0, 0)
        arc_id = s.add_arc(start, end, center, clockwise=True)

        sel = SketchSelection()
        sel.entity_ids = [arc_id]

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        clone = s.registry.get_entity(cmd.new_entity_ids[0])
        assert isinstance(clone, Arc)
        assert clone.clockwise is True

    def test_fixed_point_duplicated_as_movable(self):
        s = Sketch()
        origin = s.add_point(0, 0, fixed=True)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(origin, p2)

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        clone_line = s.registry.get_entity(cmd.new_entity_ids[0])
        assert isinstance(clone_line, Line)
        clone_origin = s.registry.get_point(clone_line.p1_idx)
        assert clone_origin.fixed is False
        assert clone_origin.x == pytest.approx(0)
        assert clone_origin.y == pytest.approx(0)

    def test_new_point_ids_track_selected_bare_points(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)

        sel = SketchSelection()
        sel.point_ids = [p1, p2]

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        assert len(cmd.new_entity_ids) == 0
        assert len(cmd.new_point_ids) == 2
        for new_pid in cmd.new_point_ids:
            p = s.registry.get_point(new_pid)
            assert p.x in (pytest.approx(0), pytest.approx(100))

    def test_construction_flag_preserved(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2, construction=True)

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = DuplicateCommand(s, sel)
        cmd.capture_snapshot()
        cmd._do_execute()

        clone = s.registry.get_entity(cmd.new_entity_ids[0])
        assert isinstance(clone, Line)
        assert clone.construction is True


class TestDuplicateUndo:
    def test_undo_removes_duplicates(self, sketch_with_l):
        s, entity_ids, _point_ids = sketch_with_l
        num_points = len(s.registry.points)
        num_entities = len(s.registry.entities)

        sel = SketchSelection()
        sel.entity_ids = entity_ids

        cmd = DuplicateCommand(s, sel)
        cmd.execute()
        assert len(s.registry.points) == num_points + len(cmd.created_points)
        assert len(s.registry.entities) == 2 * num_entities

        cmd.undo()

        assert len(s.registry.points) == num_points
        assert len(s.registry.entities) == num_entities
        for eid in cmd.new_entity_ids:
            assert s.registry.get_entity(eid) is None

    def test_undo_removes_duplicated_constraints(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2)
        s.constraints.append(DistanceConstraint(p1, p2, 100.0))

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = DuplicateCommand(s, sel)
        cmd.execute()
        assert len(s.constraints) == 2

        cmd.undo()
        assert len(s.constraints) == 1

    def test_undo_restores_original_geometry(self):
        s = Sketch()
        p1 = s.add_point(0, 0)
        p2 = s.add_point(100, 0)
        line_id = s.add_line(p1, p2)

        sel = SketchSelection()
        sel.entity_ids = [line_id]

        cmd = DuplicateCommand(s, sel)
        cmd.execute()

        # Move the duplicate to prove undo restores the full state
        clone_line = s.registry.get_entity(cmd.new_entity_ids[0])
        assert isinstance(clone_line, Line)
        s.registry.get_point(clone_line.p1_idx).x = 500

        cmd.undo()

        assert s.registry.get_point(p1).x == pytest.approx(0)
        assert s.registry.get_point(p2).x == pytest.approx(100)
