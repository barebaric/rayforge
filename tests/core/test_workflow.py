import pytest
from unittest.mock import MagicMock
from rayforge.core.workflow import Workflow
from rayforge.core.step import Step


@pytest.fixture
def workflow():
    """Provides a real, but detached, Workflow for unit testing."""
    return Workflow("Test Workflow")


def test_add_step_to_workflow(workflow):
    assert not workflow.steps
    step = Step("Test Step")

    workflow.add_step(step)

    assert len(workflow.steps) == 1
    assert workflow.steps[0] is step
    assert step.parent is workflow


def test_add_step_fires_descendant_added_signal(workflow):
    """Test adding a step fires the descendant_added signal."""
    handler = MagicMock()
    workflow.descendant_added.connect(handler)

    step = Step("Test Step")
    workflow.add_step(step)

    handler.assert_called_once_with(workflow, origin=step)


def test_remove_step_fires_descendant_removed_signal(workflow):
    """Test removing a step fires the descendant_removed signal."""
    step = Step("Test Step")
    workflow.add_step(step)

    handler = MagicMock()
    workflow.descendant_removed.connect(handler)
    workflow.remove_step(step)

    handler.assert_called_once_with(workflow, origin=step)


def test_step_change_fires_workflow_descendant_updated_signal(workflow):
    """Test a step change fires the workflow's descendant_updated signal."""
    step = Step("Test Step")
    workflow.add_step(step)

    handler = MagicMock()
    workflow.descendant_updated.connect(handler)

    # Act
    step.set_power(500)

    # Assert
    handler.assert_called_once_with(workflow, origin=step)
