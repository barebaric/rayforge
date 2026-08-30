# flake8: noqa: E402
"""Tests for the pie menu's layout math and submenu behavior."""

import math

import gi

gi.require_version("Gtk", "4.0")

import pytest
from gi.repository import Gtk

from rayforge.ui_gtk.shared.piemenu import (
    PieMenu,
    PieMenuItem,
    angle_in_span,
    compute_pie_layout,
    index_in_span,
    submenu_span,
)

pytestmark = pytest.mark.ui


def make_item(label, children=None):
    return PieMenuItem("icon", label, data=label, children=children)


def polar(menu, angle, dist):
    """Widget coordinates at the given polar offset from the pie center."""
    center = menu.total_radius
    return (
        center + math.cos(angle) * dist,
        center + math.sin(angle) * dist,
    )


@pytest.fixture
def menu():
    pie_menu = PieMenu(Gtk.Box())
    yield pie_menu
    pie_menu.unparent()


def test_layout_flat_when_space_is_available():
    items = [
        make_item("a"),
        make_item("b", children=[make_item("b1"), make_item("b2")]),
        make_item("c"),
    ]
    collapsed, inner = compute_pie_layout(items, 10)
    assert not collapsed
    assert [item.label for item in inner] == ["a", "b1", "b2", "c"]


def test_layout_collapsed_when_inner_ring_is_tight():
    group = make_item("b", children=[make_item("b1"), make_item("b2")])
    items = [make_item(f"i{i}") for i in range(9)] + [group]
    collapsed, inner = compute_pie_layout(items, 10)
    assert collapsed
    assert len(inner) == 10
    assert inner[-1].label == "b"
    assert inner[-1].children == group.children


def test_layout_drops_empty_groups_when_collapsed():
    hidden = make_item("hidden")
    hidden.visible = False
    items = [
        make_item("a"),
        make_item("empty", children=[hidden]),
        make_item("g", children=[make_item("c1")]),
    ]
    collapsed, inner = compute_pie_layout(items, 2)
    assert collapsed
    assert [item.label for item in inner] == ["a", "c1"]


def test_layout_hoists_single_child_groups_when_collapsed():
    group = make_item("g", children=[make_item("c1")])
    items = [make_item(f"i{i}") for i in range(10)] + [group]
    collapsed, inner = compute_pie_layout(items, 10)
    assert collapsed
    assert [item.label for item in inner[-2:]] == ["i9", "c1"]
    assert not inner[-1].has_children


def test_angle_in_span():
    assert angle_in_span(0.5, 0.2, 0.8)
    assert not angle_in_span(0.9, 0.2, 0.8)
    assert angle_in_span(6.2, 6.0, 0.5)
    assert angle_in_span(0.2, 6.0, 0.5)
    assert not angle_in_span(3.0, 6.0, 0.5)


def test_submenu_span_expands_only_when_needed():
    start, end, step = submenu_span(1.0, 0.3, 2, 100.0, 50.0)
    assert math.isclose(end - start, 1.0)
    assert math.isclose(step, 0.5)
    assert math.isclose((start + end) / 2, 1.0)

    start, end, step = submenu_span(1.0, 0.4, 2, 100.0, 20.0)
    assert math.isclose(end - start, 0.8)
    assert math.isclose(step, 0.4)
    assert math.isclose((start + end) / 2, 1.0)


def test_index_in_span():
    assert index_in_span(0.25, 0.0, 0.5, 4) == 0
    assert index_in_span(0.75, 0.0, 0.5, 4) == 1
    assert index_in_span(1.9, 0.0, 0.5, 4) == 3


def test_default_inner_capacity(menu):
    assert menu._get_max_inner_items() == 10


def test_hover_on_flat_menu(menu):
    menu.set_items([make_item(f"i{i}") for i in range(6)])
    mid_angle = (1 + 0.5) * (2 * math.pi / 6)
    menu._on_motion(None, *polar(menu, mid_angle, 52))
    assert menu._active_index == 1
    assert menu._active_child_index == -1

    menu._on_motion(None, *polar(menu, 0, 0))
    assert menu._active_index == -1


def test_submenu_opens_and_reaches_children(menu):
    group = make_item("group", children=[make_item("c0"), make_item("c1")])
    menu.set_items([make_item(f"i{i}") for i in range(9)] + [group])
    assert menu._collapsed

    parent_index = 9
    step = 2 * math.pi / len(menu._inner_items)
    mid_angle = (parent_index + 0.5) * step
    menu._on_motion(None, *polar(menu, mid_angle, 52))
    assert menu._active_index == parent_index
    assert menu._is_submenu_open()

    start, _end, child_step = menu._submenu_geometry()
    child_angle = start + 0.5 * child_step
    child_dist = (menu.sub_radius_inner + menu.sub_radius_outer) / 2
    menu._on_motion(None, *polar(menu, child_angle, child_dist))
    assert menu._active_index == parent_index
    assert menu._active_child_index == 0


def test_corridor_prevents_accidental_submenu_switch(menu):
    group = make_item("group", children=[make_item("c0"), make_item("c1")])
    menu.set_items([make_item(f"i{i}") for i in range(9)] + [group])

    parent_index = 9
    step = 2 * math.pi / len(menu._inner_items)
    mid_angle = (parent_index + 0.5) * step
    menu._on_motion(None, *polar(menu, mid_angle, 52))
    assert menu._is_submenu_open()

    # Slightly inside the neighboring wedge but still within the
    # corridor around the parent wedge.
    drift_angle = parent_index * step - 0.06
    menu._on_motion(None, *polar(menu, drift_angle, 52))
    assert menu._active_index == parent_index
    assert menu._active_child_index == -1

    # Well into the neighboring wedge the selection must switch and
    # the submenu must close.
    far_angle = (8 + 0.5) * step
    menu._on_motion(None, *polar(menu, far_angle, 52))
    assert menu._active_index == 8
    assert not menu._is_submenu_open()


def test_release_on_child_activates_it(menu):
    activated = []
    child = make_item("c0")
    child.on_click.connect(
        lambda sender: activated.append(sender.label), weak=False
    )
    group = make_item("group", children=[child, make_item("c1")])
    menu.set_items([make_item(f"i{i}") for i in range(9)] + [group])

    parent_index = 9
    step = 2 * math.pi / len(menu._inner_items)
    menu._on_motion(None, *polar(menu, (parent_index + 0.5) * step, 52))
    start, _end, child_step = menu._submenu_geometry()
    child_angle = start + 0.5 * child_step
    child_dist = (menu.sub_radius_inner + menu.sub_radius_outer) / 2

    x, y = polar(menu, child_angle, child_dist)
    inner_index, child_index = menu._resolve_target(x, y)
    menu._activate_target(inner_index, child_index)
    assert activated == ["c0"]


def test_release_on_group_reveals_submenu(menu):
    activated = []
    group = make_item("group", children=[make_item("c0"), make_item("c1")])
    group.on_click.connect(
        lambda sender: activated.append(sender.label), weak=False
    )
    menu.set_items([make_item(f"i{i}") for i in range(9)] + [group])

    parent_index = 9
    menu._activate_target(parent_index, -1)
    assert activated == []
    assert menu._active_index == parent_index
    assert menu._is_submenu_open()
