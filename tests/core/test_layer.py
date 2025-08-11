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
    mock_workpiece.transform_changed = MagicMock()
    mock_workpiece.parent = None

    layer.add_workpiece(mock_workpiece)

    layer_changed_handler.assert_called_once_with(layer)
    assert mock_workpiece in layer.workpieces


def test_add_workpiece_fires_descendant_added(layer):
    """Adding a workpiece should fire descendant_added."""
    handler = MagicMock()
    layer.descendant_added.connect(handler)

    mock_workpiece = MagicMock(spec=WorkPiece)
    mock_workpiece.changed = MagicMock()
    mock_workpiece.transform_changed = MagicMock()
    mock_workpiece.parent = None

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


def test_workpiece_data_change_bubbles_up_to_layer(layer):
    """
    A data change on a workpiece (via .changed) should bubble relevant
    signals.
    """
    mock_workpiece = MagicMock(spec=WorkPiece)
    # Fix: The handler accesses .name for logging, so the mock needs it.
    mock_workpiece.name = "Mock WP"
    mock_workpiece.changed = MagicMock()
    mock_workpiece.transform_changed = MagicMock()
    mock_workpiece.parent = None
    layer.add_workpiece(mock_workpiece)

    layer_changed_handler = MagicMock()
    descendant_updated_handler = MagicMock()
    layer.changed.connect(layer_changed_handler)
    layer.descendant_updated.connect(descendant_updated_handler)

    # Act: Simulate the .changed signal being fired, which calls
    # _on_workpiece_changed
    # The connect call returns the handler function itself.
    handler_func = mock_workpiece.changed.connect.call_args.args[0]
    handler_func(mock_workpiece)

    # Assert: Both descendant_updated and changed signals should fire.
    descendant_updated_handler.assert_called_once_with(
        layer, origin=mock_workpiece
    )
    layer_changed_handler.assert_called_once_with(layer)


def test_workpiece_transform_change_bubbles_up_to_layer(layer):
    """
    A transform change on a workpiece (via .transform_changed) should
    bubble a specific signal up.
    """
    mock_workpiece = MagicMock(spec=WorkPiece)
    mock_workpiece.changed = MagicMock()
    mock_workpiece.transform_changed = MagicMock()
    mock_workpiece.parent = None
    layer.add_workpiece(mock_workpiece)

    transform_changed_handler = MagicMock()
    layer.descendant_transform_changed.connect(transform_changed_handler)

    # Act: Simulate the .transform_changed signal firing.
    handler_func = mock_workpiece.transform_changed.connect.call_args.args[0]
    handler_func(mock_workpiece)

    # Assert
    transform_changed_handler.assert_called_once_with(
        layer, origin=mock_workpiece
    )
