# flake8: noqa: E402
"""Tests for the gesture registry."""

import gi

gi.require_version("Gtk", "4.0")

import pytest

pytestmark = pytest.mark.ui

from gi.repository import Gdk

from rayforge.ui_gtk.gestures.model import (
    BUTTON_MIDDLE,
    GestureContext,
    GestureKind,
    GestureSlot,
    GestureSpec,
)
from rayforge.ui_gtk.gestures.registry import GestureRegistry


class _FakeConfig:
    def __init__(self, bindings=None):
        self.gesture_bindings = bindings or {}


def _context(context_id="ctx"):
    return GestureContext(id=context_id, label="Context")


def _slot(context_id="ctx", slot_id="pan", default=None):
    return GestureSlot(
        id=slot_id,
        context_id=context_id,
        label="Slot",
        default_binding=default
        if default is not None
        else GestureSpec(GestureKind.DRAG, button=BUTTON_MIDDLE),
    )


class TestRegistration:
    def test_register_context_and_get_slots(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot())
        assert [c.id for c in registry.get_contexts()] == ["ctx"]
        assert [s.id for s in registry.get_slots("ctx")] == ["pan"]

    def test_reregistering_identical_entries_is_noop(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot())
        calls = []
        registry.changed.connect(
            lambda sender, **kw: calls.append(sender), weak=False
        )
        registry.register_context(_context())
        registry.register_slot(_slot())
        assert calls == []

    def test_unregister_all_from_addon(self):
        registry = GestureRegistry()
        registry.register_context(_context(), "addon_a")
        registry.register_slot(_slot(), "addon_a")
        registry.register_context(_context("other"), "addon_b")

        removed = registry.unregister_all_from_addon("addon_a")

        assert removed == 2
        assert [c.id for c in registry.get_contexts()] == ["other"]
        assert registry.get_slots("ctx") == []

    def test_get_slot_returns_none_for_unknown(self):
        registry = GestureRegistry()
        assert registry.get_slot("ctx", "pan") is None
        assert registry.get_context("ctx") is None


class TestResolveBinding:
    def test_missing_entry_falls_back_to_default(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot())
        default = GestureSpec(GestureKind.DRAG, button=BUTTON_MIDDLE)
        assert registry.resolve_binding(_FakeConfig(), "ctx", "pan") == (
            default
        )

    def test_stored_binding_overrides_default(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot())
        config = _FakeConfig({"ctx": {"pan": "click+primary"}})
        assert registry.resolve_binding(config, "ctx", "pan") == (
            GestureSpec(GestureKind.CLICK, button=1)
        )

    def test_none_stored_value_unassigns_slot(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot())
        config = _FakeConfig({"ctx": {"pan": None}})
        assert registry.resolve_binding(config, "ctx", "pan") is None

    def test_invalid_stored_value_falls_back_to_default(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot())
        default = GestureSpec(GestureKind.DRAG, button=BUTTON_MIDDLE)
        config = _FakeConfig({"ctx": {"pan": "bogus+entry"}})
        assert registry.resolve_binding(config, "ctx", "pan") == default

    def test_unknown_slot_resolves_to_none(self):
        registry = GestureRegistry()
        assert registry.resolve_binding(_FakeConfig(), "x", "y") is None

    def test_unregistered_context_with_modifier_spec(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(
            _slot(
                default=GestureSpec(
                    GestureKind.DRAG,
                    button=BUTTON_MIDDLE,
                    modifiers=Gdk.ModifierType.SHIFT_MASK,
                )
            )
        )
        config = _FakeConfig({"ctx": {"pan": "drag+shift+middle"}})
        assert registry.resolve_binding(config, "ctx", "pan") == (
            GestureSpec(
                GestureKind.DRAG,
                button=BUTTON_MIDDLE,
                modifiers=Gdk.ModifierType.SHIFT_MASK,
            )
        )


class TestConflictDetection:
    def test_finds_conflicting_slot(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot(slot_id="pan"))
        registry.register_slot(_slot(slot_id="zoom"))
        config = _FakeConfig({"ctx": {"pan": "scroll"}})
        conflict = registry.find_binding_conflict(
            config, "ctx", GestureSpec(GestureKind.SCROLL)
        )
        assert conflict is not None
        assert conflict.id == "pan"

    def test_excluded_slot_is_ignored(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot(slot_id="pan"))
        config = _FakeConfig({"ctx": {"pan": "scroll"}})
        assert (
            registry.find_binding_conflict(
                config,
                "ctx",
                GestureSpec(GestureKind.SCROLL),
                exclude_slot_id="pan",
            )
            is None
        )

    def test_free_gesture_has_no_conflict(self):
        registry = GestureRegistry()
        registry.register_context(_context())
        registry.register_slot(_slot(slot_id="pan"))
        config = _FakeConfig({"ctx": {"pan": "scroll"}})
        assert (
            registry.find_binding_conflict(
                config, "ctx", GestureSpec(GestureKind.CLICK, button=2)
            )
            is None
        )
