# flake8: noqa: E402
"""Tests for the GestureRouter dispatch logic."""

import gi

gi.require_version("Gtk", "4.0")

import pytest

pytestmark = pytest.mark.ui

from gi.repository import Gdk, Gtk

from rayforge.ui_gtk.gestures import gesture_registry
from rayforge.ui_gtk.gestures.model import (
    BUTTON_MIDDLE,
    BUTTON_PRIMARY,
    BUTTON_SECONDARY,
    GestureKind,
    GestureSlot,
    GestureSpec,
)
from rayforge.ui_gtk.gestures.router import GestureRouter

_ZERO_STATE = Gdk.ModifierType(0)


class _FakeConfig:
    def __init__(self, bindings=None):
        self.gesture_bindings = bindings or {}


class _FakeGesture:
    """Minimal GestureDrag/GestureClick stand-in."""

    def __init__(self, button=BUTTON_MIDDLE, state=_ZERO_STATE):
        self._button = button
        self._state = state
        self.states = []

    def get_current_button(self):
        return self._button

    def get_current_event_state(self):
        return self._state

    def get_start_point(self):
        return True, 10.0, 20.0

    def set_state(self, state):
        self.states.append(state)


class _FakeScrollController:
    """Minimal EventControllerScroll stand-in."""

    def __init__(self, state=_ZERO_STATE):
        self._state = state

    def get_current_event_state(self):
        return self._state


@pytest.fixture
def click_and_scroll_slots():
    """Registers non-drag test slots for the canvas2d context."""
    gesture_registry.register_slot(
        GestureSlot(
            id="context_menu",
            context_id="canvas2d",
            label="Context menu",
            default_binding=GestureSpec(
                GestureKind.CLICK, button=BUTTON_SECONDARY
            ),
        ),
        "test",
    )
    gesture_registry.register_slot(
        GestureSlot(
            id="zoom",
            context_id="canvas2d",
            label="Zoom",
            default_binding=GestureSpec(GestureKind.SCROLL),
        ),
        "test",
    )
    yield
    gesture_registry.unregister_all_from_addon("test")


def _defaults():
    return {
        "canvas2d": {
            "pan": "drag+middle",
            "zoom": "scroll",
            "context_menu": "click+secondary",
        }
    }


def _recorder():
    calls = []
    return calls


class TestDragDispatch:
    def test_matching_drag_is_claimed_and_dispatched(self):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_drag(
            "pan",
            begin=lambda g, x, y: calls.append(("begin", x, y)),
            update=lambda g, dx, dy: calls.append(("update", dx, dy)),
            end=lambda g, dx, dy: calls.append(("end", dx, dy)),
        )
        gesture = _FakeGesture()
        router._on_drag_begin(gesture, 10.0, 20.0)
        router._on_drag_update(gesture, 50.0, 60.0)
        router._on_drag_update(gesture, 55.0, 66.0)
        router._on_drag_end(gesture, 58.0, 70.0)

        assert calls == [
            ("begin", 10.0, 20.0),
            ("update", 0.0, 0.0),
            ("update", 5.0, 6.0),
            ("end", 8.0, 10.0),
        ]
        assert gesture.states == [Gtk.EventSequenceState.CLAIMED]

    def test_drag_below_threshold_never_dispatches(self):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_drag(
            "pan", update=lambda g, dx, dy: calls.append("update")
        )
        gesture = _FakeGesture()
        router._on_drag_begin(gesture, 10.0, 20.0)
        router._on_drag_update(gesture, 3.0, 4.0)
        router._on_drag_end(gesture, 3.0, 4.0)

        assert calls == []
        assert gesture.states == []

    def test_unmatched_drag_is_denied(self):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_drag(
            "pan", begin=lambda g, x, y: calls.append("begin")
        )
        gesture = _FakeGesture(button=BUTTON_PRIMARY)
        router._on_drag_begin(gesture, 1.0, 2.0)

        assert calls == []
        assert gesture.states == [Gtk.EventSequenceState.DENIED]

    def test_modifier_distinguishes_bindings(self):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas3d",
            widget,
            config_provider=lambda: _FakeConfig({}),
        )
        calls = _recorder()
        router.register_drag(
            "orbit", begin=lambda g, x, y: calls.append("orbit")
        )
        router.register_drag("pan", begin=lambda g, x, y: calls.append("pan"))
        pan_gesture = _FakeGesture(
            button=BUTTON_MIDDLE, state=Gdk.ModifierType.SHIFT_MASK
        )
        router._on_drag_begin(pan_gesture, 0.0, 0.0)
        router._on_drag_update(pan_gesture, 30.0, 0.0)
        router._on_drag_end(pan_gesture, 30.0, 0.0)
        orbit_gesture = _FakeGesture(button=BUTTON_MIDDLE)
        router._on_drag_begin(orbit_gesture, 0.0, 0.0)
        router._on_drag_update(orbit_gesture, 30.0, 0.0)

        assert calls == ["pan", "orbit"]

    def test_unassigned_slot_never_matches(self):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig({"canvas2d": {"pan": None}}),
        )
        calls = _recorder()
        router.register_drag(
            "pan", begin=lambda g, x, y: calls.append("begin")
        )
        router._on_drag_begin(_FakeGesture(), 0.0, 0.0)
        assert calls == []

    def test_drag_update_without_active_slot_is_ignored(self):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_drag(
            "pan", update=lambda g, dx, dy: calls.append("update")
        )
        router._on_drag_update(_FakeGesture(), 5.0, 5.0)
        assert calls == []


class TestClickDispatch:
    def test_matching_click_invokes_handler(self, click_and_scroll_slots):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_click(
            "context_menu",
            invoke=lambda g, n, x, y: calls.append((n, x, y)),
        )
        gesture = _FakeGesture(button=BUTTON_SECONDARY)
        router._on_click_pressed(gesture, 1, 30.0, 40.0)

        assert calls == [(1, 30.0, 40.0)]
        assert gesture.states == [Gtk.EventSequenceState.CLAIMED]

    def test_unmatched_click_is_denied(self, click_and_scroll_slots):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_click(
            "context_menu",
            invoke=lambda g, n, x, y: calls.append("invoke"),
        )
        gesture = _FakeGesture(button=BUTTON_PRIMARY)
        router._on_click_pressed(gesture, 1, 0.0, 0.0)

        assert calls == []
        assert gesture.states == [Gtk.EventSequenceState.DENIED]

    def test_click_with_unexpected_modifier_is_denied(
        self, click_and_scroll_slots
    ):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_click(
            "context_menu",
            invoke=lambda g, n, x, y: calls.append("invoke"),
        )
        gesture = _FakeGesture(
            button=BUTTON_SECONDARY, state=Gdk.ModifierType.CONTROL_MASK
        )
        router._on_click_pressed(gesture, 1, 0.0, 0.0)
        assert calls == []


class TestScrollDispatch:
    def test_matching_scroll_invokes_handler(self, click_and_scroll_slots):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig(_defaults()),
        )
        calls = _recorder()
        router.register_scroll(
            "zoom", scroll=lambda c, dx, dy: calls.append((dx, dy))
        )
        router._on_scroll(_FakeScrollController(), 0.0, -1.0)
        assert calls == [(0.0, -1.0)]

    def test_scroll_unassigned_is_ignored(self, click_and_scroll_slots):
        widget = Gtk.Box()
        router = GestureRouter(
            "canvas2d",
            widget,
            config_provider=lambda: _FakeConfig({"canvas2d": {"zoom": None}}),
        )
        calls = _recorder()
        router.register_scroll(
            "zoom", scroll=lambda c, dx, dy: calls.append("scroll")
        )
        router._on_scroll(_FakeScrollController(), 0.0, -1.0)
        assert calls == []


class TestRegistryIntegration:
    def test_builtin_contexts_are_registered(self):
        assert gesture_registry.get_context("canvas2d") is not None
        assert gesture_registry.get_context("canvas3d") is not None
        canvas2d_ids = {s.id for s in gesture_registry.get_slots("canvas2d")}
        assert canvas2d_ids == {"pan"}
        canvas3d_ids = {s.id for s in gesture_registry.get_slots("canvas3d")}
        assert canvas3d_ids == {"orbit", "pan", "z_rotate"}

    def test_default_bindings_resolve_for_canvas2d(self):
        binding = gesture_registry.resolve_binding(
            _FakeConfig(), "canvas2d", "pan"
        )
        assert binding == GestureSpec(GestureKind.DRAG, button=BUTTON_MIDDLE)
