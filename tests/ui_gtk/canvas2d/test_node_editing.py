from raygeo.geo import Geometry

from rayforge.ui_gtk.canvas2d.node_editing import (
    Handle,
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


def _square() -> Geometry:
    geo = Geometry()
    geo.move_to(0, 0)
    for x, y in ((1, 0), (1, 1), (0, 1), (0, 0)):
        geo.line_to(x, y)
    return geo


def test_nodes_of_the_selected_contour_only():
    geo = _two_contours()
    assert visible_nodes(geo, {1}, None) == [0, 1, 2]
    assert visible_nodes(geo, set(), 4) == [3, 4]
    assert visible_nodes(geo, set(), None) == []


def test_the_seam_of_a_closed_contour_is_one_node():
    geo = _square()
    assert visible_nodes(geo, {1}, None) == [0, 1, 2, 3]
    assert visible_nodes(geo, {3}, None) == [0, 1, 2, 3]


def test_an_open_contour_shows_its_end_node():
    geo = _two_contours()
    assert visible_nodes(geo, {2}, None) == [0, 1, 2]


def test_handles_of_selected_beziers_and_the_selected_node():
    geo = _two_contours()
    handles = visible_handles(geo, {2}, None)
    assert [(h.index, h.which) for h in handles] == [(2, 1), (2, 2)]
    assert handles[0].anchor == (1.0, 0.0)
    assert handles[1].anchor == (1.0, 1.0)
    assert visible_handles(geo, {1}, None) == []
    at_node = visible_handles(geo, set(), 1)
    assert [(h.index, h.which) for h in at_node] == [(2, 1), (2, 2)]


def test_the_seam_node_shows_the_handles_of_both_sides():
    geo = _square()
    geo = geo.set_segment_curve(1, True)[0]
    geo = geo.set_segment_curve(3, True)[0]
    geo = geo.set_segment_curve(4, True)[0]
    handles = visible_handles(geo, set(), 0)
    assert [(h.index, h.which) for h in handles] == [
        (1, 1),
        (1, 2),
        (4, 1),
        (4, 2),
    ]


def test_nearest_uses_a_per_axis_tolerance():
    data = _two_contours().data
    nodes = node_positions(data, [0, 1, 2])
    assert nearest(nodes, (1.04, 0.0), (0.05, 0.01)) == 1
    assert nearest(nodes, (1.0, 0.04), (0.05, 0.01)) is None
    assert nearest(nodes, (3.0, 3.0), (0.05, 0.05)) is None


def test_handles_win_from_nodes_when_asked_first():
    geo = _two_contours()
    handles = visible_handles(geo, {2}, None)
    hit = nearest(handle_positions(handles), (1.2, 0.2), (0.05, 0.05))
    assert isinstance(hit, Handle)
    assert (hit.index, hit.which) == (2, 1)
