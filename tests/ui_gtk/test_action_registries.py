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
gi.require_version("Adw", "1")
from gi.repository import Gio, Gtk

from rayforge.ui_gtk.action_registry import ActionRegistry
from rayforge.ui_gtk.actions import ActionExtensionRegistry, ActionManager

pytestmark = pytest.mark.ui


class StubWindow:
    def __init__(self):
        self.added: list[Gio.SimpleAction] = []
        self.removed: list[Gio.SimpleAction] = []

    def add_action(self, action):
        self.added.append(action)

    def remove_action(self, action):
        self.removed.append(action)


def _make_handler():
    def _register_actions(action_manager):
        pass

    return _register_actions


def _make_manager() -> ActionManager:
    return ActionManager.__new__(ActionManager)


def test_unregister_all_from_addon_with_same_handler_names():
    registry = ActionExtensionRegistry()
    sketcher_handler = _make_handler()
    print_and_cut_handler = _make_handler()
    registry.register_setup(sketcher_handler, "sketcher")
    registry.register_setup(print_and_cut_handler, "print_and_cut")

    assert registry.unregister_all_from_addon("sketcher") == 1
    assert registry._setup_handlers == [print_and_cut_handler]

    assert registry.unregister_all_from_addon("print_and_cut") == 1
    assert registry._setup_handlers == []


def test_register_setup_is_idempotent():
    registry = ActionExtensionRegistry()
    handler = _make_handler()
    registry.register_setup(handler, "addon")
    registry.register_setup(handler, "addon")

    assert registry._setup_handlers == [handler]
    assert registry.unregister_all_from_addon("addon") == 1


def test_register_state_update_is_idempotent():
    registry = ActionExtensionRegistry()

    def _update_action_states(action_manager):
        pass

    registry.register_state_update(_update_action_states, "addon")
    registry.register_state_update(_update_action_states, "addon")

    assert registry._state_update_handlers == [_update_action_states]


def test_register_stores_enabled_when():
    registry = ActionRegistry()
    registry.set_window(cast(Gtk.ApplicationWindow, StubWindow()))
    action = Gio.SimpleAction.new("test-action", None)

    def enabled_when(action_manager):
        return True

    registry.register(
        action_name="test-action",
        action=action,
        addon_name="test",
        enabled_when=enabled_when,
    )

    info = registry.get("test-action")
    assert info is not None and info.enabled_when is enabled_when
    registry.unregister("test-action")


def test_update_predicated_actions_evaluates_predicate(monkeypatch):
    registry = ActionRegistry()
    registry.set_window(cast(Gtk.ApplicationWindow, StubWindow()))
    monkeypatch.setattr("rayforge.ui_gtk.actions.action_registry", registry)
    action = Gio.SimpleAction.new("predicated-test-action", None)
    action.set_enabled(False)
    received = []

    def enabled_when(action_manager):
        received.append(action_manager)
        return True

    registry.register(
        action_name="predicated-test-action",
        action=action,
        addon_name="test",
        enabled_when=enabled_when,
    )

    manager = _make_manager()
    manager._update_predicated_actions()
    assert action.get_enabled() is True
    assert received == [manager]


def test_update_predicated_actions_isolates_failing_predicate(monkeypatch):
    registry = ActionRegistry()
    registry.set_window(cast(Gtk.ApplicationWindow, StubWindow()))
    monkeypatch.setattr("rayforge.ui_gtk.actions.action_registry", registry)
    bad = Gio.SimpleAction.new("predicated-bad-action", None)
    good = Gio.SimpleAction.new("predicated-good-action", None)
    good.set_enabled(False)

    def failing(action_manager):
        raise RuntimeError("boom")

    registry.register(
        action_name="predicated-bad-action",
        action=bad,
        addon_name="test",
        enabled_when=failing,
    )
    registry.register(
        action_name="predicated-good-action",
        action=good,
        addon_name="test",
        enabled_when=lambda am: True,
    )

    _make_manager()._update_predicated_actions()
    assert good.get_enabled() is True
