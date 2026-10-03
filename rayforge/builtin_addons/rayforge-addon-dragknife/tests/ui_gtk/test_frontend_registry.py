# flake8: noqa: E402
"""Verify the dragknife frontend registers pages via the hook."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from dragknife.frontend import register_step_settings_pages

from rayforge.ui_gtk.doceditor.step_settings.page_registry import (
    StepSettingsPageRegistry,
)


def test_frontend_registers_pages():
    registry = StepSettingsPageRegistry()
    register_step_settings_pages(registry)
    assert registry.get("drag_knife") is not None
    assert registry.get("tangential_knife") is not None
