# flake8: noqa: E402
import os
import sys
from types import SimpleNamespace
from typing import cast

import pytest

if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip(
        "DISPLAY not set on Linux, skipping UI tests. Run with xvfb-run.",
        allow_module_level=True,
    )

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gio, GLib, Gtk

from rayforge.machine.models.machine import Machine
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


def _machine(name: str, machine_id: str) -> Machine:
    """Creates the minimal machine shape the menu builder consumes."""
    return cast(Machine, SimpleNamespace(name=name, id=machine_id))


def _menu_items(menu: Gio.Menu) -> list[dict[str, str | None]]:
    """Returns label/action/target attributes for every menu item."""
    items: list[dict[str, str | None]] = []
    for index in range(menu.get_n_items()):
        item: dict[str, str | None] = {}
        for attribute in ("label", "action", "target"):
            value = menu.get_item_attribute_value(index, attribute, None)
            item[attribute] = value.get_string() if value else None
        items.append(item)
    return items


def _find_submenu(menu: Gio.Menu, label: str) -> Gio.Menu | None:
    """Returns the submenu whose label matches, if present."""
    for index in range(menu.get_n_items()):
        value = menu.get_item_attribute_value(index, "label", None)
        if value is None or value.get_string() != label:
            continue
        return cast(
            "Gio.Menu | None",
            menu.get_item_link(index, Gio.MENU_LINK_SUBMENU),
        )
    return None


def _find_model_buttons(widget: Gtk.Widget) -> list[Gtk.Widget]:
    """Finds the private GtkModelButton widgets inside a popover."""
    buttons: list[Gtk.Widget] = []
    properties = {prop.name for prop in widget.list_properties()}
    if {"active", "menu-name"} <= properties:
        buttons.append(widget)
    child = widget.get_first_child()
    while child is not None:
        buttons.extend(_find_model_buttons(child))
        child = child.get_next_sibling()
    return buttons


@pytest.mark.ui
def test_machine_menu_lists_all_machines():
    menu = MainMenu()
    menu.update_machines_menu(
        [_machine("Alpha", "alpha-id"), _machine("Beta", "beta-id")]
    )

    items = _menu_items(menu.dynamic_machines_section)
    labels = [item["label"] for item in items]
    targets = [item["target"] for item in items]
    actions = [item["action"] for item in items]

    assert labels == ["Alpha", "Beta"]
    assert targets == ["alpha-id", "beta-id"]
    assert actions == ["win.select-machine", "win.select-machine"]


@pytest.mark.ui
def test_machine_menu_has_select_machine_submenu():
    menu = MainMenu()
    machine_menu = _find_submenu(menu, "_Machine")
    assert machine_menu is not None
    assert _find_submenu(machine_menu, "Select Machine") is (
        menu.machines_submenu
    )


@pytest.mark.ui
def test_machine_menu_management_entries():
    menu = MainMenu()
    machine_menu = _find_submenu(menu, "_Machine")
    assert machine_menu is not None

    popover = Gtk.PopoverMenu.new_from_model(machine_menu)
    window = Gtk.Window()
    window.set_child(popover)
    window.present()
    _pump_events()

    try:
        labels = _rendered_labels(popover)
    finally:
        window.close()

    assert "Add Machine…" not in labels
    assert labels.index("Manage Machines…") < labels.index(
        "Current Machine Settings"
    )


@pytest.mark.ui
def test_machine_menu_empty_state_is_disabled():
    menu = MainMenu()
    menu.update_machines_menu([])

    items = _menu_items(menu.dynamic_machines_section)
    assert len(items) == 1
    assert items[0]["label"] == "(No Machines)"
    assert items[0]["action"] is None


@pytest.mark.ui
def test_machine_menu_escapes_underscores():
    menu = MainMenu()
    menu.update_machines_menu([_machine("my_machine", "id")])

    assert _menu_items(menu.dynamic_machines_section)[0]["label"] == (
        "my__machine"
    )


@pytest.mark.ui
def test_machine_menu_marks_active_machine():
    menu = MainMenu()
    menu.update_machines_menu(
        [_machine("Alpha", "alpha-id"), _machine("Beta", "beta-id")]
    )

    popover = Gtk.PopoverMenu.new_from_model(menu.dynamic_machines_section)
    window = Gtk.Window()
    action = Gio.SimpleAction.new_stateful(
        "select-machine",
        GLib.VariantType.new("s"),
        GLib.Variant.new_string("beta-id"),
    )
    action_group = Gio.SimpleActionGroup()
    action_group.add_action(action)
    window.insert_action_group("win", action_group)
    window.set_child(popover)
    window.present()
    _pump_events()

    try:
        buttons = _find_model_buttons(popover)
        assert [_rendered_labels(button) for button in buttons] == [
            ["Alpha"],
            ["Beta"],
        ]
        assert [button.get_property("active") for button in buttons] == [
            False,
            True,
        ]
    finally:
        window.close()
