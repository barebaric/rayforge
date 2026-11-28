"""UI tests for the workflow property provider."""

from unittest.mock import MagicMock

import pytest
from gi.repository import Gtk

from rayforge.core.layer import Layer
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.property_providers.workflow import (
    WorkflowPropertyProvider,
)


def _setup():
    layer = Layer("Layer")
    first = WorkPiece("First")
    second = WorkPiece("Second")
    layer.add_child(first)
    layer.add_child(second)
    workflow = layer.workflows[0]
    provider = WorkflowPropertyProvider()
    provider.create_widgets()
    editor = MagicMock()
    provider.update_widgets(editor, [workflow])
    return provider, workflow, editor, first, second


def _chips(provider):
    chips = []
    child = provider.parts_box.get_first_child()
    while child is not None:
        button = child.get_child()
        if isinstance(button, Gtk.ToggleButton):
            chips.append(button)
        child = child.get_next_sibling()
    return chips


@pytest.mark.ui
def test_shows_name_steps_and_part_chips(ui_context_initializer):
    provider, workflow, _editor, _first, _second = _setup()

    assert provider.name_row.get_text() == workflow.name
    assert provider.steps_row.get_subtitle() == "No steps"
    chips = _chips(provider)
    assert [chip.get_label() for chip in chips] == ["First", "Second"]
    assert all(chip.get_active() for chip in chips)


@pytest.mark.ui
def test_toggling_a_chip_limits_the_workflow(ui_context_initializer):
    provider, workflow, editor, _first, second = _setup()
    chips = _chips(provider)

    chips[0].set_active(False)

    editor.workflow.set_workflow_workpieces.assert_called_once_with(
        workflow, [second.uid]
    )


@pytest.mark.ui
def test_applying_the_name_renames_the_workflow(ui_context_initializer):
    provider, workflow, editor, _first, _second = _setup()

    provider.name_row.set_text("Renamed")
    provider.name_row.emit("apply")

    editor.workflow.rename_workflow.assert_called_once_with(
        workflow, "Renamed"
    )
