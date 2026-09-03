# flake8: noqa: E402
"""UI tests for the shared StatusBar widget."""

import gi

gi.require_version("Gtk", "4.0")

from gi.repository import Gtk

import pytest

from rayforge.ui_gtk.shared.status_bar import StatusBar


def _measure_width(widget: Gtk.Widget) -> int:
    minimum, _, _, _ = widget.measure(Gtk.Orientation.HORIZONTAL, -1)
    return minimum


def _measure_height(widget: Gtk.Widget) -> int:
    minimum, _, _, _ = widget.measure(Gtk.Orientation.VERTICAL, -1)
    return minimum


def _count_children(box: Gtk.Box) -> int:
    count = 0
    child = box.get_first_child()
    while child is not None:
        count += 1
        child = child.get_next_sibling()
    return count


@pytest.mark.ui
def test_entries_are_added_to_content_row():
    bar = StatusBar()
    bar.add_shortcut_entry(["Space"], "Pan", separator="")
    bar.add_shortcut_entry(["Shift", "Tab"], "Constrain to Axis")
    assert _count_children(bar._content) == 2


@pytest.mark.ui
def test_clear_removes_all_entries():
    bar = StatusBar()
    bar.add_shortcut_entry(["Space"], "Pan", separator="")
    bar.add_separator()
    bar.add_shortcut_entry(["Shift"], "Constrain to Axis")
    assert _count_children(bar._content) == 3

    bar.clear()
    assert _count_children(bar._content) == 0


@pytest.mark.ui
def test_min_width_does_not_follow_content():
    """
    The status bar must not propagate the width of its shortcut entries
    into the window's size request. Entry changes during sketch drags
    would otherwise resize the whole window (issue #385).
    """
    bar = StatusBar()
    empty_width = _measure_width(bar)

    bar.add_shortcut_entry(["Space"], "Pan", separator="")
    idle_width = _measure_width(bar)

    bar.add_shortcut_entry(
        ["Shift", "Doubleclick"],
        "Select Connected with a very long description",
    )
    bar.add_shortcut_entry(
        ["Ctrl", "Shift", "Alt", "Super"],
        "Yet another very long shortcut description",
    )
    assert _count_children(bar._content) == 3
    assert idle_width == empty_width
    assert _measure_width(bar) == idle_width


@pytest.mark.ui
def test_height_follows_content():
    bar = StatusBar()
    empty_height = _measure_height(bar)

    bar.add_shortcut_entry(["Space"], "Pan", separator="")
    idle_height = _measure_height(bar)
    assert idle_height > empty_height

    bar.clear()
    assert _measure_height(bar) == empty_height
