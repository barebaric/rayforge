"""Tests for workflow editing commands."""

import pytest

from rayforge.core.layer import Layer
from rayforge.core.workflow import Workflow
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.workflow_cmd import WorkflowCmd


@pytest.fixture
def workflow_cmd(doc_editor):
    return WorkflowCmd(doc_editor)


def test_set_workflow_enabled_invalidates_the_job_output(
    workflow_cmd, monkeypatch
):
    """Toggling a workflow must drop any cached G-code, also on undo."""
    calls = []
    monkeypatch.setattr(
        workflow_cmd._editor.pipeline,
        "invalidate_job_output",
        lambda: calls.append(True),
    )
    layer = Layer("Layer")
    workflow = layer.workflows[0]

    workflow_cmd.set_workflow_enabled(workflow, False)
    assert workflow.enabled is False
    workflow_cmd._editor.history_manager.undo()
    assert workflow.enabled is True
    assert len(calls) == 2


def test_set_workflow_workpieces(workflow_cmd):
    layer = Layer("Layer")
    workflow = layer.workflows[0]
    first = WorkPiece("First")
    second = WorkPiece("Second")
    layer.add_child(first)
    layer.add_child(second)

    workflow_cmd.set_workflow_workpieces(workflow, [first.uid])

    assert workflow.workpiece_uids == [first.uid]
    workflow_cmd._editor.history_manager.undo()
    assert workflow.workpiece_uids == []


def test_set_workflow_workpieces_no_change(workflow_cmd):
    layer = Layer("Layer")
    workflow = layer.workflows[0]

    workflow_cmd.set_workflow_workpieces(workflow, [])

    assert len(workflow_cmd._editor.history_manager.undo_stack) == 0


def test_reorder_workflows(workflow_cmd):
    layer = Layer("Layer")
    first = layer.workflows[0]
    second = Workflow("Second")
    layer.add_child(second)

    workflow_cmd.reorder_workflows(layer, [second, first])

    assert layer.workflows == [second, first]
    workflow_cmd._editor.history_manager.undo()
    assert layer.workflows == [first, second]


def test_workflow_parts_follow_explicit_order():
    layer = Layer("Layer")
    first = WorkPiece("First")
    second = WorkPiece("Second")
    layer.add_child(first)
    layer.add_child(second)
    workflow = layer.workflows[0]

    workflow.set_workpiece_uids([second.uid, first.uid])

    assert workflow.workpieces == [second, first]
