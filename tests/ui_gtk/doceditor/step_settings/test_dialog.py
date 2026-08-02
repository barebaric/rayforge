# flake8: noqa: E402
"""Tests for the StepSettingsDialog."""

import gi
import pytest

gi.require_version("Gtk", "4.0")
gi.require_version("Adw", "1")

from rayforge.ui_gtk.doceditor.step_settings.dialog import StepSettingsDialog
from rayforge.ui_gtk.doceditor.step_settings.pages import (
    GeneralStepSettingsPage,
    StepSettingsPage,
)


@pytest.mark.ui
def test_dialog_uses_fallback_page_for_unknown_step(editor, step):
    dialog = StepSettingsDialog(editor, step)
    assert isinstance(dialog.general_view, GeneralStepSettingsPage)
    dialog.close()


@pytest.mark.ui
def test_set_step_settings_page(editor, step):
    dialog = StepSettingsDialog(editor, step)
    page = StepSettingsPage(editor, step)
    dialog.set_step_settings_page(page)
    assert dialog.general_view is page
    dialog.close()
