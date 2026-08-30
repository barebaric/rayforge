# flake8: noqa: E402
"""Tests for grouping sketcher tools into pie menu submenus."""

import gi

gi.require_version("Gtk", "4.0")

import pytest
from gi.repository import Gtk
from sketcher.ui_gtk.piemenu import SketchPieMenu
from sketcher.ui_gtk.tools import PIE_GROUPS, TOOL_REGISTRY

from rayforge.ui_gtk.shared.piemenu import PieMenuItem

pytestmark = pytest.mark.ui


def make_item(label):
    return PieMenuItem("icon", label, data=label)


class FakeTool:
    PIE_GROUP: str | None

    def __init__(self, group):
        self.PIE_GROUP = group


def test_group_items_collapse_into_parents():
    pie_menu = SketchPieMenu(Gtk.Box())
    try:
        entries = [
            (FakeTool(None), make_item("a")),
            (FakeTool("array"), make_item("c1")),
            (FakeTool("array"), make_item("c2")),
            (FakeTool("other"), make_item("b")),
        ]
        roots = pie_menu._group_items(entries)
        assert [item.label for item in roots] == ["a", "Array", "b"]
        array_parent = roots[1]
        assert array_parent.has_children
        assert [c.label for c in array_parent.children] == ["c1", "c2"]
    finally:
        pie_menu.unparent()


def test_all_pie_groups_have_metadata():
    for tool_cls in TOOL_REGISTRY.values():
        group = tool_cls.PIE_GROUP
        if group is not None:
            assert group in PIE_GROUPS, f"missing metadata for {group}"
