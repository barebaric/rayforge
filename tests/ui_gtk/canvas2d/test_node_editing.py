from raygeo.geo import Geometry

from rayforge.ui_gtk.canvas2d.elements.node_editing import (
    Handle,
    contour_span,
    handle_positions,
    nearest,
    node_positions,
    visible_handles,
    visible_nodes,
)


def _two_contours() -> Geometry:
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(1, 0)
    geo.bezier_to(1, 1, 1.2, 0.2, 1.2, 0.8)
    geo.move_to(5, 5)
    geo.line_to(6, 5)
    return geo


def test_contour_span():
    data = _two_contours().data
    assert contour_span(data, 2) == (0, 2)
    assert contour_span(data, 3) == (3, 4)
    assert contour_span(data, 4) == (3, 4)


def test_nodes_of_the_selected_contour_only():
    data = _two_contours().data
    assert visible_nodes(data, {1}, None) == [0, 1, 2]
    assert visible_nodes(data, set(), 4) == [3, 4]
    assert visible_nodes(data, set(), None) == []


def test_handles_of_selected_beziers_and_the_selected_node():
    data = _two_contours().data
    handles = visible_handles(data, {2}, None)
    assert [(h.index, h.which) for h in handles] == [(2, 1), (2, 2)]
    assert handles[0].anchor == (1.0, 0.0)
    assert handles[1].anchor == (1.0, 1.0)
    assert visible_handles(data, {1}, None) == []
    at_node = visible_handles(data, set(), 1)
    assert [(h.index, h.which) for h in at_node] == [(2, 1), (2, 2)]


def test_nearest_uses_a_per_axis_tolerance():
    data = _two_contours().data
    nodes = node_positions(data, [0, 1, 2])
    assert nearest(nodes, (1.04, 0.0), (0.05, 0.01)) == 1
    assert nearest(nodes, (1.0, 0.04), (0.05, 0.01)) is None
    assert nearest(nodes, (3.0, 3.0), (0.05, 0.05)) is None


def test_handles_win_from_nodes_when_asked_first():
    data = _two_contours().data
    handles = visible_handles(data, {2}, None)
    hit = nearest(handle_positions(handles), (1.2, 0.2), (0.05, 0.05))
    assert isinstance(hit, Handle)
    assert (hit.index, hit.which) == (2, 1)
