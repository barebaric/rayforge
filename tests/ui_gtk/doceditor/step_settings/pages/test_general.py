# flake8: noqa: E402
"""Tests for the GeneralStepSettingsPage fallback."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.ui_gtk.doceditor.step_settings.pages import (
    GeneralStepSettingsPage,
    StepSettingsPage,
)


@pytest.mark.ui
def test_general_page_is_a_step_settings_page(editor, step):
    page = GeneralStepSettingsPage(editor, step)
    assert isinstance(page, StepSettingsPage)
