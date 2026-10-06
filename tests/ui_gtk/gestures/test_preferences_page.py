# flake8: noqa: E402
"""Smoke tests for the gesture preferences page."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

pytestmark = pytest.mark.ui

from gi.repository import Adw, Gdk

from rayforge.ui_gtk.gestures import (
    BUTTON_MIDDLE,
    BUTTON_SECONDARY,
    gesture_registry,
)
from rayforge.ui_gtk.gestures.model import (
    GestureContext,
    GestureKind,
    GestureSlot,
    GestureSpec,
)
from rayforge.ui_gtk.settings.gesture_preferences_page import (
    GesturePreferencesPage,
)


def _selected_label(page, context_id, slot_id):
    row = page._rows.get((context_id, slot_id))
    assert row is not None
    item = row.get_selected_item()
    assert item is not None
    return item.get_string()


class TestGesturePreferencesPage:
    def test_builds_groups_for_builtin_contexts(self, ui_context_initializer):
        page = GesturePreferencesPage()
        labels = [group.get_title() for group in page._groups]
        assert "2D Canvas" in labels
        assert "3D Canvas" in labels

    def test_rows_are_dropdowns_of_the_current_binding(
        self, ui_context_initializer
    ):
        page = GesturePreferencesPage()
        row = page._rows.get(("canvas2d", "pan"))
        assert isinstance(row, Adw.ComboRow)
        assert "Middle" in _selected_label(page, "canvas2d", "pan")

    def test_row_reflects_config_change(self, ui_context_initializer):
        config = ui_context_initializer.config
        page = GesturePreferencesPage()
        try:
            config.set_gesture_binding("canvas2d", "pan", "drag+secondary")
            assert "Right" in _selected_label(page, "canvas2d", "pan")
            config.reset_gesture_binding("canvas2d", "pan")
            assert "Middle" in _selected_label(page, "canvas2d", "pan")
        finally:
            config.reset_gesture_binding("canvas2d", "pan")

    def test_stale_binding_is_still_shown(self, ui_context_initializer):
        """A stored binding outside the slot's buttons stays selectable."""
        config = ui_context_initializer.config
        page = GesturePreferencesPage()
        try:
            config.set_gesture_binding("canvas2d", "pan", "drag+primary")
            assert "Left" in _selected_label(page, "canvas2d", "pan")
        finally:
            config.reset_gesture_binding("canvas2d", "pan")

    def test_selecting_option_updates_config(self, ui_context_initializer):
        config = ui_context_initializer.config
        page = GesturePreferencesPage()
        try:
            row = page._rows[("canvas2d", "pan")]
            options = page._options[("canvas2d", "pan")]
            right_drag = GestureSpec(
                GestureKind.DRAG, button=Gdk.BUTTON_SECONDARY
            )
            row.set_selected(options.index(right_drag))
            assert config.gesture_bindings["canvas2d"]["pan"] == (
                "drag+secondary"
            )
        finally:
            config.reset_gesture_binding("canvas2d", "pan")

    def test_does_not_offer_primary_button_for_pan(
        self, ui_context_initializer
    ):
        page = GesturePreferencesPage()
        buttons = {
            option.button
            for option in page._options[("canvas2d", "pan")]
            if option is not None
        }
        assert buttons == {BUTTON_MIDDLE, BUTTON_SECONDARY}

    def test_selecting_default_clears_custom_binding(
        self, ui_context_initializer
    ):
        config = ui_context_initializer.config
        page = GesturePreferencesPage()
        try:
            config.set_gesture_binding("canvas2d", "pan", "drag+secondary")
            row = page._rows[("canvas2d", "pan")]
            options = page._options[("canvas2d", "pan")]
            middle_drag = GestureSpec(GestureKind.DRAG, button=BUTTON_MIDDLE)
            row.set_selected(options.index(middle_drag))
            assert "pan" not in config.gesture_bindings.get("canvas2d", {})
        finally:
            config.reset_gesture_binding("canvas2d", "pan")

    def test_offers_unassigned_option(self, ui_context_initializer):
        page = GesturePreferencesPage()
        assert None in page._options[("canvas2d", "pan")]

    def test_excludes_bindings_of_other_slots(self, ui_context_initializer):
        page = GesturePreferencesPage()
        orbit_default = GestureSpec(GestureKind.DRAG, button=BUTTON_MIDDLE)
        assert orbit_default not in page._options[("canvas3d", "pan")]
        pan_default = GestureSpec(
            GestureKind.DRAG,
            button=BUTTON_MIDDLE,
            modifiers=Gdk.ModifierType.SHIFT_MASK,
        )
        assert pan_default not in page._options[("canvas3d", "orbit")]

    def test_registry_change_triggers_rebuild(self, ui_context_initializer):
        page = GesturePreferencesPage()
        groups_before = len(page._groups)
        try:
            gesture_registry.register_context(
                GestureContext(id="testctx", label="Test"), "test"
            )
            gesture_registry.register_slot(
                GestureSlot(id="pan", context_id="testctx", label="Test pan"),
                "test",
            )
            assert len(page._groups) == groups_before + 1
        finally:
            gesture_registry.unregister_all_from_addon("test")
