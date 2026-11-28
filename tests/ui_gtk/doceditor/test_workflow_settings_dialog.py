"""UI tests for the workflow settings dialog."""

import pytest
from gi.repository import Gdk

from rayforge.core.layer import Layer
from rayforge.ui_gtk.doceditor.workflow_settings_dialog import (
    WorkflowSettingsDialog,
)


def _make_dialog(color: str | None = None):
    workflow = Layer(name="l").workflows[0]
    if color is not None:
        workflow.set_color(color)
    dialog = WorkflowSettingsDialog(
        workflow=workflow, transient_for=None, editor=None
    )
    return dialog, workflow


@pytest.mark.ui
def test_shows_workflow_name_color_and_enabled(ui_context_initializer):
    dialog, workflow = _make_dialog("#123456")

    assert dialog.name_row.get_text() == workflow.name
    rgba = dialog.color_button.get_rgba()
    assert (
        round(rgba.red * 255),
        round(rgba.green * 255),
        round(rgba.blue * 255),
    ) == (0x12, 0x34, 0x56)
    assert dialog.enabled_row.get_active() == workflow.enabled
    dialog.destroy()


@pytest.mark.ui
def test_edits_name_color_and_enabled(ui_context_initializer):
    dialog, workflow = _make_dialog()

    dialog.name_row.set_text("New name")
    assert workflow.name == "New name"

    rgba = Gdk.RGBA()
    rgba.parse("#abcdef")
    dialog.color_button.set_rgba(rgba)
    assert workflow.color == "#abcdef"

    dialog.enabled_row.set_active(False)
    assert workflow.enabled is False
    dialog.destroy()
