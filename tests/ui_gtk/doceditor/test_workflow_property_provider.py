"""UI tests for the workflow property provider."""

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from gi.repository import Gtk

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.ui_gtk.doceditor.property_providers.transform import (
    TransformPropertyProvider,
)
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
def test_shows_name_and_part_chips(ui_context_initializer):
    provider, workflow, _editor, _first, _second = _setup()

    assert provider.name_row.get_text() == workflow.name
    chips = _chips(provider)
    assert [chip.get_label() for chip in chips] == ["First", "Second"]
    assert all(chip.get_active() for chip in chips)
    assert provider._workflow_view is not None
    assert provider._workflow_view.workflow is workflow


@pytest.mark.ui
def test_shows_the_standard_workflow_step_list(ui_context_initializer):
    layer = Layer("Layer")
    workflow = layer.workflows[0]
    step = Step(typelabel="Contour", name="Cut")
    workflow.add_step(step)
    doc = Doc()
    doc.set_children([])
    doc.add_layer(layer)
    provider = WorkflowPropertyProvider()
    provider.create_widgets()

    provider.update_widgets(MagicMock(), [workflow])

    assert provider._workflow_view is not None
    rows = list(provider._workflow_view.draglist)
    assert len(rows) == 1
    assert cast(Any, rows[0]).stepbox.step is step


@pytest.mark.ui
def test_workflow_has_no_transform_properties(ui_context_initializer):
    workflow = Layer("Layer").workflows[0]
    provider = TransformPropertyProvider()

    assert provider.can_handle([workflow]) is False


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
