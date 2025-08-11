import pytest
from unittest.mock import MagicMock
from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from rayforge.core.step import Step


@pytest.fixture
def doc():
    return Doc()


@pytest.fixture
def layer(doc):
    """Provides the first layer from a real Doc."""
    return doc.layers[0]


def test_add_workpiece_to_layer_fires_changed(layer):
    layer_changed_handler = MagicMock()
    layer.changed.connect(layer_changed_handler)

    # Using a mock workpiece as its implementation is not relevant here
    mock_workpiece = MagicMock(spec=WorkPiece)
    mock_workpiece.changed = MagicMock()  # Mock the signal attribute
    mock_workpiece.layer = None

    layer.add_workpiece(mock_workpiece)

    layer_changed_handler.assert_called_once_with(layer)
    assert mock_workpiece in layer.workpieces


def test_add_workpiece_fires_descendant_added(layer):
    """Adding a workpiece should fire descendant_added."""
    handler = MagicMock()
    layer.descendant_added.connect(handler)

    mock_workpiece = MagicMock(spec=WorkPiece)
    mock_workpiece.changed = MagicMock()
    mock_workpiece.layer = None

    layer.add_workpiece(mock_workpiece)
    handler.assert_called_once_with(layer, origin=mock_workpiece)


def test_workflow_change_bubbles_up_to_layer(layer):
    """
    Integration test: A change on a Workflow should notify its parent Layer.
    """
    layer_changed_handler = MagicMock()
    layer.changed.connect(layer_changed_handler)

    workflow = layer.workflow
    step = Step(workflow, "Test Step")

    # Act
    workflow.add_step(step)

    # Assert
    layer_changed_handler.assert_called_once_with(layer)


def test_workflow_descendant_added_bubbles_to_layer(layer):
    """A descendant_added signal from a workflow should bubble up."""
    handler = MagicMock()
    layer.descendant_added.connect(handler)

    workflow = layer.workflow
    step = Step(workflow, "Test Step")

    # Act
    workflow.add_step(step)

    # Assert
    handler.assert_called_once_with(layer, origin=step)
