import pytest
from unittest.mock import MagicMock, patch
from rayforge.core.workflow import Workflow
from rayforge.core.step import Step
from rayforge.core.matrix import Matrix


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
    step.set_power(0.5)

    # Assert
    handler.assert_called_once_with(workflow, origin=step)


def test_workflow_to_dict_serialization():
    """Tests serializing a Workflow to a dictionary."""
    workflow = Workflow("Test Workflow")
    workflow.matrix = Matrix.translation(5, 10) @ Matrix.rotation(45)

    step1 = Step("Step 1")
    step2 = Step("Step 2")
    workflow.add_step(step1)
    workflow.add_step(step2)

    data = workflow.to_dict()

    expected_matrix = Matrix.translation(5, 10) @ Matrix.rotation(45)

    assert data["type"] == "workflow"
    assert data["name"] == "Test Workflow"
    assert data["matrix"] == expected_matrix.to_list()
    assert "children" in data
    assert len(data["children"]) == 2


def test_workflow_from_dict_deserialization():
    """Tests deserializing a Workflow from a dictionary."""
    workflow_dict = {
        "uid": "test-workflow-uid",
        "type": "workflow",
        "name": "Deserialized Workflow",
        "matrix": [[1, 0, 10], [0, 1, 20], [0, 0, 1]],
        "children": [],
    }

    workflow = Workflow.from_dict(workflow_dict)

    assert isinstance(workflow, Workflow)
    assert workflow.uid == "test-workflow-uid"
    assert workflow.name == "Deserialized Workflow"
    assert workflow.matrix == Matrix.translation(10, 20)


def test_workflow_from_dict_with_no_children():
    """Tests deserializing a Workflow with no children."""
    workflow_dict = {
        "uid": "empty-workflow-uid",
        "type": "workflow",
        "name": "Empty Workflow",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "children": [],
    }

    workflow = Workflow.from_dict(workflow_dict)

    assert isinstance(workflow, Workflow)
    assert workflow.uid == "empty-workflow-uid"
    assert workflow.name == "Empty Workflow"
    assert len(workflow.steps) == 0


def test_workflow_from_dict_ignores_non_step_children():
    """Tests that from_dict ignores children that are not steps."""
    workflow_dict = {
        "uid": "mixed-workflow-uid",
        "type": "workflow",
        "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "children": [
            {
                "uid": "step1-uid",
                "type": "step",
            },
            {
                "uid": "workpiece-uid",
                "type": "workpiece",
            },
        ],
    }

    with patch("rayforge.core.step.Step.from_dict") as mock_step_from_dict:
        mock_step = MagicMock()
        mock_step_from_dict.return_value = mock_step

        workflow = Workflow.from_dict(workflow_dict)

        assert isinstance(workflow, Workflow)
        assert len(workflow.children) == 1
        assert workflow.children[0] is mock_step
        mock_step_from_dict.assert_called_once_with(
            {
                "uid": "step1-uid",
                "type": "step",
            }
        )


def test_workflow_roundtrip_serialization():
    """Tests that to_dict() and from_dict() produce equivalent objects."""
    # Create a workflow with steps
    original = Workflow("Roundtrip Workflow")
    original.matrix = Matrix.translation(5, 10) @ Matrix.rotation(30)

    step1 = Step("Step 1")
    step2 = Step("Step 2")
    original.add_step(step1)
    original.add_step(step2)

    # Serialize and deserialize
    data = original.to_dict()
    restored = Workflow.from_dict(data)

    # Check that the restored object has the same properties
    assert restored.uid == original.uid
    assert restored.name == original.name
    assert restored.matrix == original.matrix
    assert len(restored.steps) == len(original.steps)
