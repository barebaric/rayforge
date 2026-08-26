from unittest.mock import Mock

from sketcher.core import Sketch
from sketcher.core.commands import BezierCommand, BezierPreviewState
from sketcher.core.entities import Bezier


def test_bezier_command_preserves_control_points():
    """A committed standalone bezier must keep its control points exactly
    as previewed; tangency smoothing must not pick up the new entity."""
    sketch = Sketch()
    start_pid = sketch.add_point(100.0, 100.0)

    cmd = BezierCommand(
        sketch,
        start_id=start_pid,
        end_pos=(200.0, 100.0),
        is_line=False,
        cp1=(30.0, 40.0),
        cp2=(-30.0, -40.0),
    )
    cmd.execute()

    bezier = next(e for e in sketch.registry.entities if isinstance(e, Bezier))
    assert bezier.cp1 == (30.0, 40.0)
    assert bezier.cp2 == (-30.0, -40.0)


def test_bezier_command_mirrors_cp_from_adjacent_segment():
    """When starting from a smooth waypoint of a previous bezier, the new
    segment's cp1 is mirrored from the previous segment's cp2."""
    sketch = Sketch()
    p1 = sketch.add_point(0.0, 0.0)
    p2 = sketch.add_point(100.0, 0.0)

    first = BezierCommand(
        sketch,
        start_id=p1,
        end_pos=(100.0, 0.0),
        end_pid=p2,
        is_line=False,
        cp1=(30.0, 10.0),
        cp2=(20.0, 15.0),
    )
    first.execute()

    second = BezierCommand(
        sketch,
        start_id=p2,
        end_pos=(200.0, 50.0),
        is_line=False,
        cp1=(-20.0, -15.0),
        cp2=(10.0, -5.0),
    )
    second.execute()

    beziers = [e for e in sketch.registry.entities if isinstance(e, Bezier)]
    assert len(beziers) == 2
    assert beziers[1].cp1 == (-20.0, -15.0)
    assert beziers[1].cp2 == (10.0, -5.0)


def test_bezier_preview_state_defaults():
    state = BezierPreviewState(start_id=1, start_temp=True)
    assert state.get_preview_point_ids() == set()
    assert state.virtual_cp is None
    assert state.get_virtual_cp_absolute(Mock()) is None
