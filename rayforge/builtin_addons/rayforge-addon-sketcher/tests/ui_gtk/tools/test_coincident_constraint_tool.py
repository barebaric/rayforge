from unittest.mock import MagicMock

import pytest
from sketcher.core.commands import AddItemsCommand
from sketcher.core.constraints import (
    CoincidentConstraint,
    PointOnCurveConstraint,
    PointOnLineConstraint,
)
from sketcher.core.params import ParameterContext
from sketcher.core.selection import SketchSelection
from sketcher.core.sketch import Sketch
from sketcher.ui_gtk.tools import CoincidentConstraintTool


@pytest.fixture
def bezier_env():
    sketch = Sketch()
    start = sketch.add_point(0, 0, fixed=True)
    end = sketch.add_point(10, 0, fixed=True)
    bezier_id = sketch.registry.add_bezier(
        start_idx=start, end_idx=end, cp1=(0.0, 10.0), cp2=(0.0, 10.0)
    )
    return sketch, start, end, bezier_id


@pytest.fixture
def mock_element(bezier_env):
    sketch, _start, _end, _bezier_id = bezier_env
    element = MagicMock()
    element.sketch = sketch
    element.selection = SketchSelection()
    element.execute_command = MagicMock()
    return element


@pytest.fixture
def tool(mock_element):
    return CoincidentConstraintTool(mock_element)


def select_point_and_entity(element, point_id, entity_id):
    element.selection.point_ids = [point_id]
    element.selection.entity_ids = [entity_id]


def test_is_available_for_point_and_bezier(tool, mock_element, bezier_env):
    sketch, _start, _end, bezier_id = bezier_env
    point = sketch.add_point(5, 3)
    select_point_and_entity(mock_element, point, bezier_id)
    assert tool.is_available(None, None) is True


def test_is_available_false_without_selection(tool, mock_element):
    assert tool.is_available(None, None) is False


def test_is_available_false_when_point_is_bezier_endpoint(
    tool, mock_element, bezier_env
):
    _sketch, start, _end, bezier_id = bezier_env
    select_point_and_entity(mock_element, start, bezier_id)
    assert tool.is_available(None, None) is False


def test_is_available_false_when_already_constrained(
    tool, mock_element, bezier_env
):
    sketch, _start, _end, bezier_id = bezier_env
    point = sketch.add_point(5, 3)
    sketch.constraints.append(PointOnCurveConstraint(point, bezier_id))
    select_point_and_entity(mock_element, point, bezier_id)
    assert tool.is_available(None, None) is False


def test_bezier_unavailable_for_line_selection(tool, mock_element, bezier_env):
    sketch, start, end, _bezier_id = bezier_env
    point = sketch.add_point(5, 5)
    line_id = sketch.registry.add_line(start, end)
    select_point_and_entity(mock_element, point, line_id)
    assert tool._has_point_on_curve_constraint(sketch, point, line_id) is False
    assert tool.is_available(None, None) is True


def test_on_activate_adds_point_on_curve(tool, mock_element, bezier_env):
    sketch, _start, _end, bezier_id = bezier_env
    point = sketch.add_point(5, 3)
    select_point_and_entity(mock_element, point, bezier_id)

    tool.on_activate()

    mock_element.execute_command.assert_called_once()
    cmd = mock_element.execute_command.call_args[0][0]
    assert isinstance(cmd, AddItemsCommand)
    assert len(cmd.constraints) == 1
    constr = cmd.constraints[0]
    assert isinstance(constr, PointOnCurveConstraint)
    assert constr.point_id == point
    assert constr.shape_id == bezier_id
    mock_element.set_tool.assert_called_once_with("select")


def test_two_point_selection_still_creates_point_pair(
    tool, mock_element, bezier_env
):
    sketch, start, _end, _bezier_id = bezier_env
    point = sketch.add_point(5, 5)
    mock_element.selection.point_ids = [point, start]

    tool.on_activate()

    cmd = mock_element.execute_command.call_args[0][0]
    constr = cmd.constraints[0]
    assert isinstance(constr, CoincidentConstraint)
    assert {constr.p1, constr.p2} == {point, start}


def test_point_and_line_still_creates_point_on_line(
    tool, mock_element, bezier_env
):
    sketch, start, end, _bezier_id = bezier_env
    point = sketch.add_point(5, 5)
    line_id = sketch.registry.add_line(start, end)
    select_point_and_entity(mock_element, point, line_id)

    tool.on_activate()

    cmd = mock_element.execute_command.call_args[0][0]
    constr = cmd.constraints[0]
    assert isinstance(constr, PointOnLineConstraint)
    assert constr.point_id == point
    assert constr.shape_id == line_id


def test_end_to_end_point_moves_onto_curve(tool, mock_element, bezier_env):
    sketch, _start, _end, bezier_id = bezier_env
    point = sketch.add_point(5, 6)
    select_point_and_entity(mock_element, point, bezier_id)
    mock_element.execute_command.side_effect = lambda cmd: cmd.execute()

    tool.on_activate()

    constr = sketch.constraints[-1]
    assert isinstance(constr, PointOnCurveConstraint)

    sketch.solve()

    assert constr.error(sketch.registry, ParameterContext()) < 1e-4
    moved = sketch.registry.get_point(point)
    assert moved.y == pytest.approx(7.5, abs=1e-2)
    assert moved.x == pytest.approx(5.0, abs=1e-2)
