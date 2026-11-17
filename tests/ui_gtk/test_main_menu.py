# flake8: noqa: E402
import os
import sys
from typing import cast

import pytest

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip(
        "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
        allow_module_level=True,
    )

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from rayforge.ui_gtk.main_menu import MainMenu


class _RecentInfo:
    """Minimal stand-in for Gtk.RecentInfo."""

    def __init__(self, display_name: str, uri: str):
        self._display_name = display_name
        self._uri = uri

    def get_display_name(self) -> str:
        return self._display_name

    def get_uri(self) -> str:
        return self._uri


def _pump_events() -> None:
    context = GLib.MainContext.default()
    while context.pending():
        context.iteration(False)


def _rendered_labels(widget: Gtk.Widget) -> list[str]:
    labels: list[str] = []
    if isinstance(widget, Gtk.Label):
        layout = widget.get_layout()
        if layout is not None:
            labels.append(layout.get_text())
    child = widget.get_first_child()
    while child is not None:
        labels.extend(_rendered_labels(child))
        child = child.get_next_sibling()
    return labels


@pytest.mark.ui
def test_recent_filenames_render_literally():
    menu = MainMenu()
    recent_infos = cast(
        "list[Gtk.RecentInfo]",
        [
            _RecentInfo("bookmark_one.ryp", "file:///tmp/bookmark_one.ryp"),
            _RecentInfo("x&y.ryp", "file:///tmp/x%26y.ryp"),
        ],
    )
    menu.update_recent_files_menu(recent_infos)

    popover = Gtk.PopoverMenu.new_from_model(menu.recent_files_menu)
    window = Gtk.Window()
    window.set_child(popover)
    window.present()
    _pump_events()

    try:
        labels = _rendered_labels(popover)
    finally:
        window.close()

    assert "bookmark_one.ryp" in labels
    assert "x&y.ryp" in labels
    assert not any("&amp;" in label for label in labels)
