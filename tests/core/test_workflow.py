import pytest
from unittest.mock import MagicMock
from rayforge.core.doc import Doc
from rayforge.core.workflow import Workflow
from rayforge.core.step import Step


@pytest.fixture
def workflow():
    """Provides a real, but detached, Workflow for unit testing."""
    mock_doc = MagicMock(spec=Doc)
    # When using spec_set, we must explicitly list instance attributes
    # that are set in the constructor.
    mock_layer = MagicMock(
        spec_set=[
            "doc",
            "workflow",
            "name",
            "changed",
            "descendant_added",
            "descendant_removed",
            "descendant_updated",
        ]
    )
    mock_layer.doc = mock_doc
    # A real workflow needs a layer to get the doc
    wf = Workflow(mock_layer, "Test Workflow")
    mock_layer.workflow = wf
    return wf


def test_add_step_to_workflow(workflow):
    assert not workflow.steps
    step = Step(workflow, "Test Step")

    workflow.add_step(step)

    assert len(workflow.steps) == 1
    assert workflow.steps[0] is step
    assert step.workflow is workflow


def test_add_step_fires_changed_signal(workflow):
    """Test that the workflow's own changed signal fires."""
    workflow_changed_handler = MagicMock()
    workflow.changed.connect(workflow_changed_handler)

    step = Step(workflow, "Test Step")
    workflow.add_step(step)

    workflow_changed_handler.assert_called_once_with(workflow)


def test_add_step_fires_descendant_added_signal(workflow):
    """Test adding a step fires the descendant_added signal."""
    handler = MagicMock()
    workflow.descendant_added.connect(handler)

    step = Step(workflow, "Test Step")
    workflow.add_step(step)

    handler.assert_called_once_with(workflow, origin=step)


def test_remove_step_fires_descendant_removed_signal(workflow):
    """Test removing a step fires the descendant_removed signal."""
    step = Step(workflow, "Test Step")
    workflow.add_step(step)

    handler = MagicMock()
    workflow.descendant_removed.connect(handler)
    workflow.remove_step(step)

    handler.assert_called_once_with(workflow, origin=step)


def test_step_change_fires_workflow_changed_signal(workflow):
    """Test that a child step's change signal bubbles to the workflow."""
    step = Step(workflow, "Test Step")
    workflow.add_step(step)

    workflow_changed_handler = MagicMock()
    workflow.changed.connect(workflow_changed_handler)

    # Act
    step.set_power(500)

    # Assert
    workflow_changed_handler.assert_called_once_with(workflow)


def test_step_change_fires_workflow_descendant_updated_signal(workflow):
    """Test a step change fires the workflow's descendant_updated signal."""
    step = Step(workflow, "Test Step")
    workflow.add_step(step)

    handler = MagicMock()
    workflow.descendant_updated.connect(handler)

    # Act
    step.set_power(500)

    # Assert
    handler.assert_called_once_with(workflow, origin=step)
