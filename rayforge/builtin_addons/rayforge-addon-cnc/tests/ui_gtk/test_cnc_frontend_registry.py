# flake8: noqa: E402
"""Verify the cnc_essentials frontend registers pages via the new hook."""

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from cnc_essentials.frontend import register_step_settings_pages

from rayforge.ui_gtk.doceditor.step_settings.page_registry import (
    StepSettingsPageRegistry,
)


def test_frontend_registers_pages():
    registry = StepSettingsPageRegistry()
    register_step_settings_pages(registry)
    assert registry.get("adaptive_clearing") is not None
    assert registry.get("profile_outer") is not None
    assert registry.get("slot") is not None
    assert registry.get("helix") is not None
