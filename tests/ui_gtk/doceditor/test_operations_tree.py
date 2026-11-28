"""UI tests for the Operations tree interactions."""

import pytest
from gi.repository import Gtk

from rayforge.core.doc import Doc
from rayforge.core.layer import Layer
from rayforge.core.workflow import Workflow
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.tree.add_workflow_button import WorkflowBar
from rayforge.doceditor.tree.operations_tree import OperationsTree


def _make_tree() -> tuple[OperationsTree, Layer]:
    layer = Layer("Layer")
    layer.add_child(Workflow("Second Workflow"))
    first = WorkPiece("First")
    second = WorkPiece("Second")
    layer.add_child(first)
    layer.add_child(second)
    doc = Doc()
    doc.set_children([])
    doc.add_layer(layer)
    tree = OperationsTree()
    tree.set_doc(doc)
    return tree, layer


@pytest.mark.ui
def test_eye_toggle_disables_the_workflow(ui_context_initializer):
    tree, layer = _make_tree()
    workflow = layer.workflows[0]
    events = []

    def on_toggled(sender, workflow, enabled):
        events.append((workflow, enabled))

    tree.enabled_toggled.connect(on_toggled, weak=False)
    row = tree.find_row(workflow.uid)
    assert row is not None
    tree.on_eye_toggled(row.spec)

    assert events == [(workflow, False)]


@pytest.mark.ui
def test_disabled_workflow_is_struck_through(ui_context_initializer):
    tree, layer = _make_tree()
    workflow = layer.workflows[0]

    workflow.set_enabled(False)

    row = tree.find_row(workflow.uid)
    assert row is not None
    assert row.has_css_class("rf-workflow-disabled")

    workflow.set_enabled(True)

    row = tree.find_row(workflow.uid)
    assert row is not None
    assert not row.has_css_class("rf-workflow-disabled")


@pytest.mark.ui
def test_workflow_drop_reorders_workflows(ui_context_initializer):
    tree, layer = _make_tree()
    first, second = layer.workflows
    events = []

    def on_reorder(sender, layer, workflows):
        events.append((layer, workflows))

    tree.workflow_reorder_requested.connect(on_reorder, weak=False)
    source = tree.find_row(first.uid)
    target = tree.find_row(second.uid)
    assert source is not None and target is not None

    tree._on_workflow_drop(None, source, 0, 10, target)

    assert events == [(layer, [second, first])]


@pytest.mark.ui
def test_part_drop_reorders_parts(ui_context_initializer):
    tree, layer = _make_tree()
    workflow = layer.workflows[0]
    first, second = workflow.workpieces
    tree.expand(workflow.uid)
    events = []

    def on_parts(sender, workflow, workpiece_uids):
        events.append((workflow, workpiece_uids))

    tree.workflow_parts_changed.connect(on_parts, weak=False)
    source = tree.find_row(first.uid)
    target = tree.find_row(second.uid)
    assert source is not None and target is not None

    tree._on_part_drop(None, source, 0, 10, target)

    assert events == [(workflow, [second.uid, first.uid])]


@pytest.mark.ui
def test_shows_a_separator_before_each_layer(ui_context_initializer):
    tree, _layer = _make_tree()
    assert tree._doc is not None
    tree._doc.add_layer(Layer("Second"))

    separators = [
        row.spec.label
        for row in tree.rows()
        if row.spec.kind == "layer-separator"
    ]
    assert separators == ["Layer", "Second"]


@pytest.mark.ui
def test_workflow_bar_offers_three_colors_and_custom(ui_context_initializer):
    bar = WorkflowBar()
    events = []

    def on_requested(sender, color):
        events.append(color)

    bar.workflow_requested.connect(on_requested, weak=False)
    buttons = [child for child in bar if isinstance(child, Gtk.Button)]
    assert len(buttons) == 4

    buttons[0].emit("clicked")
    assert events and events[0].startswith("#")
