import math
from unittest.mock import MagicMock

import pytest
from sketcher.core import Sketch
from sketcher.core.constraints import RadiusConstraint
from sketcher.core.entities import Circle, Line
from sketcher.ui_gtk.tools import PolygonTool
from sketcher.ui_gtk.tools.base import SketcherKey


@pytest.fixture
def element():
    element = MagicMock()
    element.sketch = Sketch()
    element.hittester.get_hit_data.return_value = (None, None)
    element.hittester.screen_to_model.side_effect = lambda x, y, e: (x, y)
    element.canvas.view_transform.get_scale.return_value = (1.0, 1.0)
    element.snap_engine.query.return_value = MagicMock(
        snapped=False, position=(0.0, 0.0), snap_lines=[]
    )
    element.execute_command.side_effect = lambda cmd: cmd.execute()
    element.remove_point_if_unused.side_effect = (
        element.sketch.remove_point_if_unused
    )
    return element


@pytest.fixture
def tool(element, monkeypatch):
    tool = PolygonTool(element)
    monkeypatch.setattr(
        tool, "query_snap_for_creation", lambda el, x, y, exclude: (x, y)
    )
    monkeypatch.setattr(tool, "get_snapped_point_id", lambda: None)
    return tool


def _outline(sketch):
    return [
        e
        for e in sketch.registry.entities
        if isinstance(e, Line) and not e.construction
    ]


def _type(tool, text):
    for char in text:
        tool.handle_text_input(char)


@pytest.mark.ui
def test_drag_creates_a_hexagon(tool, element):
    tool.on_press(0.0, 0.0, 1)
    tool.on_hover_motion(10.0, 0.0)
    tool.on_release(10.0, 0.0)
    assert tool.get_preview_state() is None
    lines = _outline(element.sketch)
    assert len(lines) == 6
    circles = [
        e for e in element.sketch.registry.entities if isinstance(e, Circle)
    ]
    assert len(circles) == 1 and circles[0].construction


@pytest.mark.ui
def test_two_clicks_create_the_polygon(tool, element):
    tool.on_press(0.0, 0.0, 1)
    tool.on_release(0.5, 0.0)
    assert tool.get_preview_state() is not None
    tool.on_hover_motion(0.0, 8.0)
    tool.on_press(0.0, 8.0, 1)
    assert len(_outline(element.sketch)) == 6


@pytest.mark.ui
def test_first_corner_joins_a_snapped_point(tool, element, monkeypatch):
    target = element.sketch.add_point(0.0, 8.0)
    tool.on_press(0.0, 0.0, 1)
    tool.on_release(0.5, 0.0)
    tool.on_hover_motion(0.0, 8.0)
    monkeypatch.setattr(tool, "get_snapped_point_id", lambda: target)
    tool.on_press(0.0, 8.0, 1)
    assert _outline(element.sketch)[0].p1_idx == target


@pytest.mark.ui
def test_ctrl_makes_a_star(tool, element):
    tool.on_press(0.0, 0.0, 1)
    tool.on_modifier_change(ctrl=True)
    state = tool.get_preview_state()
    assert state is not None and state.star_ratio == 0.5
    tool.on_hover_motion(10.0, 0.0)
    tool.on_release(10.0, 0.0)
    assert len(_outline(element.sketch)) == 12


@pytest.mark.ui
def test_typed_values_set_radius_corners_and_star(tool, element):
    tool.on_press(0.0, 0.0, 1)
    tool.on_release(0.5, 0.0)
    tool.on_hover_motion(3.0, 4.0)
    _type(tool, "20")
    tool.handle_key_event(SketcherKey.TAB)
    _type(tool, "5")
    tool.handle_key_event(SketcherKey.TAB)
    _type(tool, "0.4")
    tool.handle_key_event(SketcherKey.RETURN)

    sketch = element.sketch
    assert tool.get_preview_state() is None
    assert len(_outline(sketch)) == 10
    (radius,) = [
        c for c in sketch.constraints if isinstance(c, RadiusConstraint)
    ]
    assert radius.value == 20.0
    first = sketch.registry.get_point(_outline(sketch)[0].p1_idx)
    assert math.hypot(first.x, first.y) == pytest.approx(20.0)
    assert math.atan2(first.y, first.x) == pytest.approx(math.atan2(4, 3))


@pytest.mark.ui
def test_invalid_corner_count_is_ignored(tool, element):
    tool.on_press(0.0, 0.0, 1)
    tool.on_release(0.5, 0.0)
    tool.on_hover_motion(10.0, 0.0)
    _type(tool, "10")
    tool.handle_key_event(SketcherKey.TAB)
    tool.handle_text_input("1")
    tool.handle_key_event(SketcherKey.TAB)
    state = tool.get_preview_state()
    assert state is not None and state.sides == 6
    assert state.locked_radius == 10.0


@pytest.mark.ui
def test_deactivate_removes_the_preview(tool, element):
    points_before = len(element.sketch.registry.points)
    tool.on_press(0.0, 0.0, 1)
    tool.on_hover_motion(10.0, 0.0)
    tool.on_deactivate()
    assert element.sketch.registry.entities == []
    assert len(element.sketch.registry.points) == points_before


@pytest.mark.ui
def test_corner_count_is_remembered(tool, element):
    tool.on_press(0.0, 0.0, 1)
    tool.on_release(0.5, 0.0)
    tool.on_hover_motion(10.0, 0.0)
    _type(tool, "10")
    tool.handle_key_event(SketcherKey.TAB)
    _type(tool, "8")
    tool.handle_key_event(SketcherKey.TAB)
    tool.on_press(10.0, 0.0, 1)
    tool.on_press(30.0, 0.0, 1)
    state = tool.get_preview_state()
    assert state is not None and state.sides == 8


@pytest.mark.ui
def test_registered_in_the_tool_registry():
    from sketcher.ui_gtk.tools import TOOL_REGISTRY

    assert TOOL_REGISTRY["polygon"] is PolygonTool
