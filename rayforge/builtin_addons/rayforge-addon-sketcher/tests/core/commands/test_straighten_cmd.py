import pytest

from sketcher.core import Sketch
from sketcher.core.commands import StraightenBezierCommand
from sketcher.core.entities import Bezier, Line


@pytest.fixture
def sketch():
    return Sketch()


@pytest.fixture
def curved_bezier(sketch):
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 10.0)
    bezier_id = sketch.registry.add_bezier(p1, p2)
    bezier = sketch.registry.get_entity(bezier_id)
    bezier.cp1 = (5.0, 0.0)
    bezier.cp2 = (5.0, 0.0)
    return bezier_id, p1, p2


def test_straighten_converts_bezier_to_line(sketch, curved_bezier):
    bezier_id, p1, p2 = curved_bezier
    cmd = StraightenBezierCommand(sketch, bezier_id)
    cmd.execute()

    entity = sketch.registry.get_entity(bezier_id)
    assert isinstance(entity, Line)
    assert entity.p1_idx == p1
    assert entity.p2_idx == p2


def test_straighten_preserves_construction_flag(sketch, curved_bezier):
    bezier_id, p1, p2 = curved_bezier
    bezier = sketch.registry.get_entity(bezier_id)
    bezier.construction = True

    cmd = StraightenBezierCommand(sketch, bezier_id)
    cmd.execute()

    line = sketch.registry.get_entity(bezier_id)
    assert isinstance(line, Line)
    assert line.construction is True


def test_straighten_undo_restores_bezier(sketch, curved_bezier):
    bezier_id, p1, p2 = curved_bezier
    original_bezier = sketch.registry.get_entity(bezier_id)
    original_cp1 = original_bezier.cp1
    original_cp2 = original_bezier.cp2

    cmd = StraightenBezierCommand(sketch, bezier_id)
    cmd.execute()

    assert isinstance(sketch.registry.get_entity(bezier_id), Line)

    cmd.undo()

    restored = sketch.registry.get_entity(bezier_id)
    assert isinstance(restored, Bezier)
    assert restored.cp1 == original_cp1
    assert restored.cp2 == original_cp2


def test_straighten_with_invalid_entity_id(sketch):
    cmd = StraightenBezierCommand(sketch, 9999)
    cmd.execute()

    assert cmd._old_bezier is None


def test_straighten_with_line_entity_does_nothing(sketch):
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(10.0, 10.0)
    line_id = sketch.add_line(p1, p2)

    cmd = StraightenBezierCommand(sketch, line_id)
    cmd.execute()

    entity = sketch.registry.get_entity(line_id)
    assert isinstance(entity, Line)
