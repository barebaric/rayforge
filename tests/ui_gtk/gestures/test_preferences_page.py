# flake8: noqa: E402
"""Smoke tests for the gesture preferences page."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

import pytest

pytestmark = pytest.mark.ui


from rayforge.ui_gtk.gestures import gesture_registry
from rayforge.ui_gtk.gestures.model import (
    GestureContext,
    GestureSlot,
)
from rayforge.ui_gtk.settings.gesture_preferences_page import (
    GesturePreferencesPage,
)


class TestGesturePreferencesPage:
    def test_builds_groups_for_builtin_contexts(self, ui_context_initializer):
        page = GesturePreferencesPage()
        labels = [group.get_title() for group in page._groups]
        assert "2D Canvas" in labels
        assert "3D Canvas" in labels

    def test_rows_show_binding_labels(self, ui_context_initializer):
        page = GesturePreferencesPage()
        button = page._buttons.get(("canvas2d", "pan"))
        assert button is not None
        assert "Middle" in button.get_label()

    def test_row_reflects_config_change(self, ui_context_initializer):
        config = ui_context_initializer.config
        page = GesturePreferencesPage()
        button = page._buttons.get(("canvas2d", "pan"))
        assert button is not None
        try:
            config.set_gesture_binding("canvas2d", "pan", "drag+primary")
            assert "Left" in button.get_label()
            config.reset_gesture_binding("canvas2d", "pan")
            assert "Middle" in button.get_label()
        finally:
            config.reset_gesture_binding("canvas2d", "pan")

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
